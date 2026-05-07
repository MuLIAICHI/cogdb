use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use uuid::Uuid;

/// Metadata stored on every graph edge (one per active SemanticTriple).
#[derive(Debug, Clone)]
pub struct TripleEdge {
    pub triple_id: Uuid,
    pub predicate: String,
    pub confidence: f64,
}

/// In-memory knowledge graph for active SemanticTriples.
///
/// Backed by a petgraph `DiGraph<String, TripleEdge>` where node weights are
/// entity names. Two parallel indexes enable O(1) lookups:
/// - entity name → NodeIndex
/// - triple UUID → EdgeIndex
///
/// This is the in-memory structure that accelerates graph traversal; the
/// authoritative triple data lives in SQLite. On startup the graph is rebuilt
/// from SQLite; on each mutation a WAL record is also appended.
///
/// Not thread-safe on its own — the SemanticStore wraps this in a parking_lot::RwLock.
///
/// # Example
/// ```
/// use cogdb_engine::graph::kg::{KnowledgeGraph, TripleEdge};
/// use uuid::Uuid;
/// let mut kg = KnowledgeGraph::new();
/// let id = Uuid::new_v4();
/// kg.add_edge(id, "Alice", "works_at", "Acme", 0.9);
/// assert_eq!(kg.get_neighbors("Alice"), vec!["Acme"]);
/// ```
pub struct KnowledgeGraph {
    graph: DiGraph<String, TripleEdge>,
    /// entity name → NodeIndex
    entity_index: HashMap<String, NodeIndex>,
    /// triple UUID → EdgeIndex (for O(1) targeted removal)
    edge_index: HashMap<Uuid, EdgeIndex>,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            entity_index: HashMap::new(),
            edge_index: HashMap::new(),
        }
    }

    /// Add a directed edge subject→object labelled with the triple.
    ///
    /// If either node does not exist it is created. If an edge with the same
    /// `triple_id` already exists it is first removed.
    pub fn add_edge(
        &mut self,
        triple_id: Uuid,
        subject: &str,
        predicate: &str,
        object: &str,
        confidence: f64,
    ) {
        // Remove stale edge if it exists (idempotent upsert)
        if self.edge_index.contains_key(&triple_id) {
            self.remove_edge_by_id(triple_id);
        }

        let src = self.get_or_create_node(subject);
        let dst = self.get_or_create_node(object);

        let edge = TripleEdge { triple_id, predicate: predicate.to_string(), confidence };
        let ei = self.graph.add_edge(src, dst, edge);
        self.edge_index.insert(triple_id, ei);
    }

    /// Remove a triple edge by its UUID.
    ///
    /// Returns `true` if the edge existed, `false` otherwise.
    /// Cleans up orphan nodes (nodes with no remaining edges).
    pub fn remove_edge_by_id(&mut self, triple_id: Uuid) -> bool {
        let Some(ei) = self.edge_index.remove(&triple_id) else {
            return false;
        };

        let (src_ni, dst_ni) = match self.graph.edge_endpoints(ei) {
            Some(pair) => pair,
            None => return false,
        };

        // Save names as Strings before any removal — NodeIndex values become
        // invalid after petgraph's swap-remove, but names stay stable.
        let src_name = self.graph.node_weight(src_ni).cloned();
        let dst_name = self.graph.node_weight(dst_ni).cloned();

        self.graph.remove_edge(ei);

        // For each endpoint, re-look up the current NodeIndex by name (valid
        // after the preceding rebuild), check orphan, remove if so.
        for name in [src_name, dst_name].into_iter().flatten() {
            if let Some(&ni) = self.entity_index.get(&name) {
                if self.is_orphan(ni) {
                    self.entity_index.remove(&name);
                    self.graph.remove_node(ni);
                    // petgraph swap-removes: the last node slides into ni's slot.
                    // Rebuild so the next iteration can look up by name correctly.
                    self.rebuild_entity_index();
                }
            }
        }

        true
    }

    /// BFS traversal from `entity` up to `depth` hops.
    ///
    /// Returns all triple edges reachable within `depth` steps, traversing
    /// both outgoing (subject→object) and incoming (object→subject) edges.
    /// Matches the Python SemanticStore `query_entity` behavior.
    pub fn bfs_edges(&self, entity: &str, depth: usize) -> Vec<&TripleEdge> {
        let Some(&start) = self.entity_index.get(entity) else {
            return vec![];
        };

        let mut visited_nodes: HashSet<NodeIndex> = HashSet::new();
        let mut visited_edges: HashSet<EdgeIndex> = HashSet::new();
        let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();
        let mut results: Vec<&TripleEdge> = Vec::new();

        queue.push_back((start, 0));
        visited_nodes.insert(start);

        while let Some((node, current_depth)) = queue.pop_front() {
            if current_depth >= depth {
                continue;
            }

            // Outgoing edges (node is subject)
            for ei in self.graph.edges(node).map(|e| e.id()).collect::<Vec<_>>() {
                if visited_edges.insert(ei) {
                    results.push(self.graph.edge_weight(ei).unwrap());
                    let (_, dst) = self.graph.edge_endpoints(ei).unwrap();
                    if visited_nodes.insert(dst) {
                        queue.push_back((dst, current_depth + 1));
                    }
                }
            }

            // Incoming edges (node is object) — bidirectional traversal
            for ei in self.graph.edges_directed(node, petgraph::Direction::Incoming)
                .map(|e| e.id())
                .collect::<Vec<_>>()
            {
                if visited_edges.insert(ei) {
                    results.push(self.graph.edge_weight(ei).unwrap());
                    let (src, _) = self.graph.edge_endpoints(ei).unwrap();
                    if visited_nodes.insert(src) {
                        queue.push_back((src, current_depth + 1));
                    }
                }
            }
        }

        results
    }

    /// Return the direct neighbors of `entity`.
    ///
    /// Includes both outgoing targets (subject→**object**) and incoming
    /// sources (**subject**→object), matching Python `get_neighbors`.
    pub fn get_neighbors(&self, entity: &str) -> Vec<String> {
        let Some(&ni) = self.entity_index.get(entity) else {
            return vec![];
        };
        let mut neighbors: Vec<String> = Vec::new();

        // Outgoing: node is subject
        for e in self.graph.edges(ni) {
            if let Some(name) = self.graph.node_weight(e.target()) {
                neighbors.push(name.clone());
            }
        }

        // Incoming: node is object
        for e in self.graph.edges_directed(ni, petgraph::Direction::Incoming) {
            if let Some(name) = self.graph.node_weight(e.source()) {
                neighbors.push(name.clone());
            }
        }

        neighbors.sort();
        neighbors.dedup();
        neighbors
    }

    /// All entity names currently in the graph.
    pub fn get_entities(&self) -> Vec<String> {
        self.entity_index.keys().cloned().collect()
    }

    /// Number of edges (active triples) in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Number of nodes (entities) in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns true if the graph contains a node for this entity.
    pub fn contains_entity(&self, entity: &str) -> bool {
        self.entity_index.contains_key(entity)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn get_or_create_node(&mut self, name: &str) -> NodeIndex {
        if let Some(&ni) = self.entity_index.get(name) {
            ni
        } else {
            let ni = self.graph.add_node(name.to_string());
            self.entity_index.insert(name.to_string(), ni);
            ni
        }
    }

    fn is_orphan(&self, ni: NodeIndex) -> bool {
        self.graph.edges(ni).count() == 0
            && self.graph.edges_directed(ni, petgraph::Direction::Incoming).count() == 0
    }

    /// Rebuild the entity_index from current graph node weights.
    ///
    /// Called after `remove_node` because petgraph swap-removes nodes,
    /// invalidating the NodeIndex of the last node in the slab.
    fn rebuild_entity_index(&mut self) {
        self.entity_index.clear();
        for ni in self.graph.node_indices() {
            if let Some(name) = self.graph.node_weight(ni) {
                self.entity_index.insert(name.clone(), ni);
            }
        }
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_id() -> Uuid { Uuid::new_v4() }

    #[test]
    fn add_and_count() {
        let mut kg = KnowledgeGraph::new();
        kg.add_edge(make_id(), "Alice", "works_at", "Acme", 0.9);
        kg.add_edge(make_id(), "Bob", "works_at", "Acme", 0.8);
        assert_eq!(kg.edge_count(), 2);
        assert_eq!(kg.node_count(), 3); // Alice, Bob, Acme
    }

    #[test]
    fn get_neighbors_outgoing_and_incoming() {
        let mut kg = KnowledgeGraph::new();
        kg.add_edge(make_id(), "Alice", "works_at", "Acme", 0.9);
        kg.add_edge(make_id(), "Bob", "manages", "Alice", 0.8);

        let alice_neighbors = kg.get_neighbors("Alice");
        assert!(alice_neighbors.contains(&"Acme".to_string()),  "missing outgoing");
        assert!(alice_neighbors.contains(&"Bob".to_string()),   "missing incoming");
    }

    #[test]
    fn bfs_depth_1() {
        let mut kg = KnowledgeGraph::new();
        let id1 = make_id();
        let id2 = make_id();
        kg.add_edge(id1, "Alice", "works_at", "Acme", 0.9);
        kg.add_edge(id2, "Alice", "lives_in", "NYC", 0.7);
        // depth=1 from Alice should return both edges
        let edges = kg.bfs_edges("Alice", 1);
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn bfs_depth_2_reaches_transitive() {
        let mut kg = KnowledgeGraph::new();
        kg.add_edge(make_id(), "Alice", "works_at", "Acme", 0.9);
        kg.add_edge(make_id(), "Acme", "located_in", "NYC", 0.9);
        kg.add_edge(make_id(), "NYC", "country", "USA", 0.9);

        let d1 = kg.bfs_edges("Alice", 1);
        let d2 = kg.bfs_edges("Alice", 2);
        let d3 = kg.bfs_edges("Alice", 3);

        assert_eq!(d1.len(), 1); // works_at
        assert_eq!(d2.len(), 2); // + located_in
        assert_eq!(d3.len(), 3); // + country
    }

    #[test]
    fn bfs_on_unknown_entity_returns_empty() {
        let kg = KnowledgeGraph::new();
        assert!(kg.bfs_edges("Nobody", 2).is_empty());
    }

    #[test]
    fn remove_edge_decrements_counts() {
        let mut kg = KnowledgeGraph::new();
        let id = make_id();
        kg.add_edge(id, "Alice", "works_at", "Acme", 0.9);
        assert_eq!(kg.edge_count(), 1);
        assert!(kg.remove_edge_by_id(id));
        assert_eq!(kg.edge_count(), 0);
    }

    #[test]
    fn remove_edge_cleans_up_orphan_nodes() {
        let mut kg = KnowledgeGraph::new();
        let id = make_id();
        kg.add_edge(id, "Alice", "knows", "Bob", 0.9);
        assert!(kg.contains_entity("Alice"));
        assert!(kg.contains_entity("Bob"));

        kg.remove_edge_by_id(id);

        // Both nodes are now orphans — should be removed
        assert!(!kg.contains_entity("Alice"));
        assert!(!kg.contains_entity("Bob"));
        assert_eq!(kg.node_count(), 0);
    }

    #[test]
    fn remove_shared_node_not_removed() {
        let mut kg = KnowledgeGraph::new();
        let id1 = make_id();
        let id2 = make_id();
        kg.add_edge(id1, "Alice", "works_at", "Acme", 0.9);
        kg.add_edge(id2, "Bob", "works_at", "Acme", 0.8);

        kg.remove_edge_by_id(id1);

        // Acme still has an edge from Bob → should stay
        assert!(kg.contains_entity("Acme"));
        assert!(kg.contains_entity("Bob"));
        // Alice is orphaned → should be removed
        assert!(!kg.contains_entity("Alice"));
    }

    #[test]
    fn remove_nonexistent_returns_false() {
        let mut kg = KnowledgeGraph::new();
        assert!(!kg.remove_edge_by_id(Uuid::new_v4()));
    }

    #[test]
    fn get_entities_lists_all() {
        let mut kg = KnowledgeGraph::new();
        kg.add_edge(make_id(), "Alice", "knows", "Bob", 0.9);
        let entities = kg.get_entities();
        assert_eq!(entities.len(), 2);
        assert!(entities.contains(&"Alice".to_string()));
        assert!(entities.contains(&"Bob".to_string()));
    }

    #[test]
    fn upsert_same_triple_id_replaces_edge() {
        let mut kg = KnowledgeGraph::new();
        let id = make_id();
        kg.add_edge(id, "Alice", "works_at", "Acme", 0.9);
        kg.add_edge(id, "Alice", "works_at", "Globex", 0.5); // same id, new object
        assert_eq!(kg.edge_count(), 1);
        let edges = kg.bfs_edges("Alice", 1);
        assert_eq!(edges[0].triple_id, id);
        // old object (Acme) should be gone
        assert!(!kg.contains_entity("Acme"));
        assert!(kg.contains_entity("Globex"));
    }

    #[test]
    fn bfs_does_not_traverse_same_edge_twice() {
        let mut kg = KnowledgeGraph::new();
        // Diamond: Alice→Acme, Alice→NYC, Acme→NYC
        kg.add_edge(make_id(), "Alice", "works_at", "Acme", 0.9);
        kg.add_edge(make_id(), "Alice", "lives_in", "NYC", 0.9);
        kg.add_edge(make_id(), "Acme", "located_in", "NYC", 0.9);

        let edges = kg.bfs_edges("Alice", 2);
        // All 3 edges reachable, each visited once
        assert_eq!(edges.len(), 3);
    }
}
