use std::collections::HashMap;
use std::fs;
use std::path::Path;

use bincode::config::standard;
use hnsw_rs::prelude::*;
use parking_lot::RwLock;
use uuid::Uuid;

use crate::error::{CogError, Result};

const SNAPSHOT_FILE: &str = "hnsw_vectors.bin";

/// Search result returned by the HNSW index: (id, distance).
#[derive(Debug, Clone)]
pub struct HnswHit {
    pub id: Uuid,
    /// Cosine distance (0 = identical, 2 = opposite). Lower is more similar.
    pub distance: f32,
}

struct HnswState {
    hnsw: Hnsw<'static, f32, DistCosine>,
    /// Parallel copy of all vectors keyed by label — used for snapshots and
    /// brute-force filtered search. Doubles memory; fine for Phase 1 corpus sizes.
    vectors: HashMap<usize, Vec<f32>>,
}

/// Thread-safe HNSW vector index for episodic memory embeddings.
///
/// Uses cosine distance via `hnsw_rs`. Each vector is keyed by a UUID encoded
/// as the lower 64 bits of its byte representation (usize label).
/// Snapshot save/load serializes the raw vector map and rebuilds the index.
///
/// # Example
/// ```no_run
/// use cogdb_engine::vector::hnsw::HnswIndex;
/// use uuid::Uuid;
/// let idx = HnswIndex::new(384, 16, 200);
/// let id = Uuid::new_v4();
/// idx.insert(id, &vec![0.0f32; 384]).unwrap();
/// let results = idx.search(&vec![0.0f32; 384], 5);
/// ```
pub struct HnswIndex {
    state: RwLock<HnswState>,
    pub dim: usize,
    // Stored for use when rebuilding the index during snapshot load.
    #[allow(dead_code)]
    m: usize,
    #[allow(dead_code)]
    ef_construction: usize,
}

impl HnswIndex {
    /// Create a new empty index.
    ///
    /// # Args
    /// - `dim`: Embedding dimension (e.g. 384 for all-MiniLM-L6-v2)
    /// - `m`: HNSW M parameter — connections per node (default 16)
    /// - `ef_construction`: Construction-time beam width (default 200)
    pub fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        let hnsw = Hnsw::<f32, DistCosine>::new(m, 100_000, 16, ef_construction, DistCosine {});
        Self {
            state: RwLock::new(HnswState { hnsw, vectors: HashMap::new() }),
            dim,
            m,
            ef_construction,
        }
    }

    /// Insert a vector keyed by UUID.
    pub fn insert(&self, id: Uuid, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.dim {
            return Err(CogError::InvalidArg(format!(
                "embedding dim {} != expected {}",
                embedding.len(),
                self.dim
            )));
        }
        let label = uuid_to_label(&id);
        let mut guard = self.state.write();
        guard.hnsw.insert((&embedding.to_vec(), label));
        guard.vectors.insert(label, embedding.to_vec());
        Ok(())
    }

    /// Search for the `top_k` nearest neighbours.
    ///
    /// Returns results sorted by ascending distance (most similar first).
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<HnswHit> {
        if top_k == 0 || query.len() != self.dim {
            return vec![];
        }
        let guard = self.state.read();
        let neighbours = guard.hnsw.search(query, top_k, 50);
        neighbours
            .into_iter()
            .map(|n| HnswHit { id: label_to_uuid(n.d_id), distance: n.distance })
            .collect()
    }

    /// Brute-force cosine search restricted to the given candidate UUIDs.
    ///
    /// Used when the metadata-filtered candidate set is small enough
    /// (< BRUTE_FORCE_THRESHOLD) that scanning beats HNSW overhead.
    pub fn search_filtered(&self, query: &[f32], candidates: &[Uuid], top_k: usize) -> Vec<HnswHit> {
        if candidates.is_empty() || top_k == 0 || query.len() != self.dim {
            return vec![];
        }
        let guard = self.state.read();
        let mut hits: Vec<HnswHit> = candidates
            .iter()
            .filter_map(|id| {
                let label = uuid_to_label(id);
                guard.vectors.get(&label).map(|vec| {
                    let dist = cosine_distance(query, vec);
                    HnswHit { id: *id, distance: dist }
                })
            })
            .collect();

        hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(top_k);
        hits
    }

    /// All (label, vector) pairs stored in the index — used for snapshot restore.
    pub fn vector_pairs(&self) -> Vec<(usize, Vec<f32>)> {
        self.state
            .read()
            .vectors
            .iter()
            .map(|(&label, vec)| (label, vec.clone()))
            .collect()
    }

    /// Number of vectors currently in the index.
    pub fn len(&self) -> usize {
        self.state.read().hnsw.get_nb_point()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialize all vectors to `{dir}/hnsw_vectors.bin`.
    ///
    /// On load, the HNSW is rebuilt by re-inserting all stored vectors.
    pub fn save(&self, dir: &Path) -> Result<()> {
        let guard = self.state.read();
        let pairs: Vec<(usize, Vec<f32>)> = guard.vectors.iter().map(|(k, v)| (*k, v.clone())).collect();
        let bytes = bincode::encode_to_vec(&pairs, standard())?;
        fs::write(dir.join(SNAPSHOT_FILE), &bytes)?;
        Ok(())
    }

    /// Load an index from a snapshot directory created by `save`.
    pub fn load(dir: &Path, dim: usize, m: usize, ef_construction: usize) -> Result<Self> {
        let bytes = fs::read(dir.join(SNAPSHOT_FILE))?;
        let (pairs, _): (Vec<(usize, Vec<f32>)>, _) =
            bincode::decode_from_slice(&bytes, standard())?;

        let hnsw = Hnsw::<f32, DistCosine>::new(m, 100_000.max(pairs.len() + 1), 16, ef_construction, DistCosine {});
        let mut vectors = HashMap::with_capacity(pairs.len());
        for (label, vec) in &pairs {
            hnsw.insert((vec, *label));
            vectors.insert(*label, vec.clone());
        }

        Ok(Self {
            state: RwLock::new(HnswState { hnsw, vectors }),
            dim,
            m,
            ef_construction,
        })
    }
}

// ── UUID ↔ HNSW label encoding ────────────────────────────────────────────────
// hnsw_rs uses `usize` as the data point label (same size as u64 on 64-bit).
// We use the lower 64 bits of the UUID. Collision probability at 10M records
// is ~5×10⁻¹⁰ (birthday bound), acceptable for Phase 1.

pub fn uuid_to_label(id: &Uuid) -> usize {
    let bytes = id.as_bytes();
    let lo = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    lo as usize
}

pub fn label_to_uuid(label: usize) -> Uuid {
    let lo = label as u64;
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&lo.to_le_bytes());
    Uuid::from_bytes(bytes)
}

/// Cosine distance between two vectors: 0 = identical, 2 = opposite.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 1.0; // undefined — treat as maximally distant
    }
    1.0 - dot / (na * nb)
}

/// Cosine similarity (= 1 - cosine_distance).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_distance(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use uuid::Uuid;

    fn random_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut v = Vec::with_capacity(dim);
        let mut x = seed;
        for _ in 0..dim {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            v.push(((x >> 33) as f32) / (u32::MAX as f32) - 0.5);
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 { v.iter_mut().for_each(|x| *x /= norm); }
        v
    }

    #[test]
    fn insert_and_len() {
        let idx = HnswIndex::new(16, 8, 50);
        assert_eq!(idx.len(), 0);
        idx.insert(Uuid::new_v4(), &random_unit_vec(16, 1)).unwrap();
        idx.insert(Uuid::new_v4(), &random_unit_vec(16, 2)).unwrap();
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn wrong_dim_returns_error() {
        let idx = HnswIndex::new(16, 8, 50);
        assert!(idx.insert(Uuid::new_v4(), &[0.1f32; 8]).is_err());
    }

    #[test]
    fn nearest_neighbour_is_self() {
        let idx = HnswIndex::new(32, 8, 100);
        for i in 0..20u64 {
            idx.insert(Uuid::new_v4(), &random_unit_vec(32, i + 1)).unwrap();
        }
        let query = random_unit_vec(32, 1);
        let hits = idx.search(&query, 1);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].distance < 0.01, "distance={}", hits[0].distance);
    }

    #[test]
    fn recall_at_90_percent() {
        let dim = 64;
        let idx = HnswIndex::new(dim, 16, 200);
        let n = 200usize;
        let mut vecs = Vec::with_capacity(n);
        for i in 0..n {
            let v = random_unit_vec(dim, i as u64 + 42);
            idx.insert(Uuid::new_v4(), &v).unwrap();
            vecs.push(v);
        }
        let correct = vecs.iter().filter(|v| {
            let hits = idx.search(v, 1);
            !hits.is_empty() && hits[0].distance < 0.01
        }).count();
        let recall = correct as f64 / n as f64;
        assert!(recall >= 0.90, "recall={recall:.2} < 0.90");
    }

    #[test]
    fn snapshot_roundtrip() {
        let dir = tempdir().unwrap();
        let snap_dir = dir.path().join("snap");
        fs::create_dir_all(&snap_dir).unwrap();
        let dim = 16;
        let idx = HnswIndex::new(dim, 8, 50);
        let id = Uuid::new_v4();
        let vec = random_unit_vec(dim, 99);
        idx.insert(id, &vec).unwrap();

        idx.save(&snap_dir).unwrap();

        let loaded = HnswIndex::load(&snap_dir, dim, 8, 50).unwrap();
        assert_eq!(loaded.len(), 1);
        let hits = loaded.search(&vec, 1);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].distance < 0.01);
    }

    #[test]
    fn search_filtered_brute_force() {
        let dim = 16;
        let idx = HnswIndex::new(dim, 8, 50);
        let target_id = Uuid::new_v4();
        let target_vec = random_unit_vec(dim, 7);
        idx.insert(target_id, &target_vec).unwrap();
        for i in 0..10u64 {
            idx.insert(Uuid::new_v4(), &random_unit_vec(dim, i + 100)).unwrap();
        }
        let hits = idx.search_filtered(&target_vec, &[target_id], 5);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].distance < 0.01);
    }

    #[test]
    fn cosine_similarity_unit_vectors() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        let c = vec![0.0f32, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn search_empty_returns_empty() {
        let idx = HnswIndex::new(16, 8, 50);
        assert!(idx.search(&random_unit_vec(16, 1), 0).is_empty());
        assert!(idx.search_filtered(&random_unit_vec(16, 1), &[], 5).is_empty());
    }
}
