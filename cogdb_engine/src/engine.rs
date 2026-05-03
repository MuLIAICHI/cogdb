use std::path::{Path, PathBuf};
use std::sync::Arc;

use crc32fast::Hasher;

use crate::error::Result;
use crate::stores::episodic::EpisodicStore;
use crate::stores::procedural::ProceduralStore;
use crate::stores::semantic::SemanticStore;
use crate::wal::reader::WalReader;
use crate::wal::record::WalRecord;
use crate::wal::writer::WalWriter;

/// Configuration for the CogDB storage engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Embedding dimension. Must match the model used to produce vectors (default 384).
    pub embedding_dim: usize,
    /// HNSW M parameter — connections per node (default 16).
    pub hnsw_m: usize,
    /// HNSW ef_construction — beam width during index build (default 200).
    pub hnsw_ef_construction: usize,
    /// WAL segment rotation threshold in bytes (default 64 MB).
    pub wal_max_segment_bytes: u64,
    /// Enable contradiction detection in the semantic store (default true).
    pub contradiction_check: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 384,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            wal_max_segment_bytes: 64 * 1024 * 1024,
            contradiction_check: true,
        }
    }
}

/// Top-level storage engine: opens all three stores, handles WAL recovery,
/// and orchestrates checkpoint/close.
///
/// # Directory layout
/// ```text
/// {db_path}/
///   wal/current.wal        ← append-only WAL (HNSW + graph mutations)
///   snapshots/snap-{seq}/  ← HNSW snapshots written at each checkpoint
///   episodic/              ← episodic_meta.db (SQLite, WAL mode)
///   semantic/              ← semantic.db      (SQLite, WAL mode)
///   procedural/            ← procedures.db    (SQLite, WAL mode)
/// ```
///
/// # Example
/// ```no_run
/// use std::path::Path;
/// use cogdb_engine::engine::{Engine, EngineConfig};
/// let engine = Engine::open(Path::new("/tmp/cogdb"), EngineConfig::default()).unwrap();
/// engine.close().unwrap();
/// ```
pub struct Engine {
    pub episodic: Arc<EpisodicStore>,
    pub semantic: Arc<SemanticStore>,
    pub procedural: Arc<ProceduralStore>,
    wal: Arc<WalWriter>,
    db_path: PathBuf,
}

impl Engine {
    /// Open (or create) the engine at `db_path` with WAL recovery.
    ///
    /// Recovery protocol:
    /// 1. SQLite databases auto-recover via their own WAL mode on open.
    /// 2. Read the bespoke WAL to find the last `Checkpoint` record.
    /// 3. Load the HNSW snapshot referenced by that checkpoint.
    /// 4. Replay all `EpisodicUpsert/Delete` WAL records after the checkpoint
    ///    to bring the HNSW up to date.
    /// 5. If no checkpoint exists, replay the entire WAL from the beginning.
    pub fn open(db_path: &Path, config: EngineConfig) -> Result<Self> {
        // ── 1. Create directory structure ────────────────────────────────────
        let wal_dir = db_path.join("wal");
        let snap_root = db_path.join("snapshots");
        let ep_dir = db_path.join("episodic");
        let sem_dir = db_path.join("semantic");
        let proc_dir = db_path.join("procedural");
        for dir in &[&wal_dir, &snap_root, &ep_dir, &sem_dir, &proc_dir] {
            std::fs::create_dir_all(dir)?;
        }
        let wal_path = wal_dir.join("current.wal");

        // ── 2. Read existing WAL (if any) ────────────────────────────────────
        let (old_records, next_seq) = if wal_path.exists() {
            match WalReader::open(&wal_path) {
                Ok(mut reader) => {
                    let records = reader.read_all()?;
                    let max_seq = records.iter().map(|r| r.seq()).max().unwrap_or(0);
                    (records, max_seq + 1)
                }
                Err(_) => (vec![], 0), // corrupt WAL header — start fresh
            }
        } else {
            (vec![], 0)
        };

        // ── 3. Identify last checkpoint ──────────────────────────────────────
        let last_checkpoint = old_records.iter().rev().find_map(|r| {
            if let WalRecord::Checkpoint { seq, snapshot_path, .. } = r {
                Some((*seq, snapshot_path.clone()))
            } else {
                None
            }
        });
        // None means no checkpoint found → replay ALL records.
        // Some(s) means replay only records with seq > s.
        let checkpoint_seq: Option<u64> = last_checkpoint.as_ref().map(|(s, _)| *s);

        // ── 4. Create WAL writer (continues appending to the same file) ──────
        let wal = Arc::new(WalWriter::open(&wal_path, next_seq, config.wal_max_segment_bytes)?);

        // ── 5. Open stores ───────────────────────────────────────────────────
        let episodic = Arc::new(EpisodicStore::open(
            &ep_dir,
            config.embedding_dim,
            config.hnsw_m,
            config.hnsw_ef_construction,
            wal.clone(),
        )?);
        let semantic = Arc::new(SemanticStore::open(
            &sem_dir,
            config.contradiction_check,
            wal.clone(),
        )?);
        let procedural = Arc::new(ProceduralStore::open(&proc_dir, wal.clone())?);

        // ── 6. Restore HNSW snapshot (if checkpoint exists) ──────────────────
        if let Some((_, ref snap_name)) = last_checkpoint {
            let snap_dir = snap_root.join(snap_name);
            if snap_dir.exists() {
                episodic.restore_from_snapshot(&snap_dir)?;
            }
        }

        // ── 7. Replay WAL records after the checkpoint ───────────────────────
        for record in &old_records {
            if let Some(cs) = checkpoint_seq {
                if record.seq() <= cs {
                    continue;
                }
            }
            match record {
                WalRecord::EpisodicUpsert { id, embedding, .. } => {
                    // Ignore errors here — if the embedding is already present
                    // (e.g., duplicate WAL replay), HNSW just overwrites.
                    let _ = episodic.apply_wal_upsert(*id, embedding.clone());
                }
                WalRecord::EpisodicDelete { id, .. } => {
                    episodic.apply_wal_delete(*id);
                }
                _ => {} // Semantic/procedural data is fully in SQLite — no replay needed.
            }
        }

        // After WAL replay, ensure the label_map reflects all SQLite rows
        // (handles the case where recovery added HNSW entries without label_map entries).
        episodic.rebuild_label_map_from_db()?;

        Ok(Self { episodic, semantic, procedural, wal, db_path: db_path.to_path_buf() })
    }

    /// Write an HNSW snapshot and append a Checkpoint WAL record.
    ///
    /// Old WAL records before the checkpoint are safe to ignore on next startup.
    pub fn checkpoint(&self) -> Result<()> {
        let seq = self.wal.current_seq();
        let snap_name = format!("snap-{seq}");
        let snap_dir = self.db_path.join("snapshots").join(&snap_name);
        std::fs::create_dir_all(&snap_dir)?;

        self.episodic.save_snapshot(&snap_dir)?;

        let snapshot_crc = Self::dir_crc32(&snap_dir);
        self.wal.append(WalRecord::Checkpoint {
            seq: 0,
            snapshot_path: snap_name,
            snapshot_crc,
        })?;

        Ok(())
    }

    /// Flush a final checkpoint and close all SQLite connections.
    ///
    /// After this call, all file handles are released so the db_path directory
    /// can be deleted (important for Windows test teardown with TemporaryDirectory).
    pub fn close(&self) -> Result<()> {
        self.checkpoint()?;
        // Swap file-backed connections to in-memory ones — releases all file locks.
        self.episodic.close_connections();
        self.semantic.close_connections();
        self.procedural.close_connections();
        Ok(())
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// CRC32 of all files in a directory (for snapshot integrity check).
    fn dir_crc32(dir: &Path) -> u32 {
        let mut hasher = Hasher::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            let mut paths: Vec<_> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
            paths.sort(); // deterministic ordering
            for path in paths {
                if let Ok(bytes) = std::fs::read(&path) {
                    hasher.update(&bytes);
                }
            }
        }
        hasher.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MemoryScope, MemoryType, MemoryUnit};
    use chrono::Utc;
    use tempfile::tempdir;
    use uuid::Uuid;

    fn small_config() -> EngineConfig {
        EngineConfig {
            embedding_dim: 8,
            hnsw_m: 4,
            hnsw_ef_construction: 20,
            wal_max_segment_bytes: 64 * 1024 * 1024,
            contradiction_check: true,
        }
    }

    fn make_unit(agent_id: &str, dim: usize) -> MemoryUnit {
        let emb: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) / (dim as f32)).collect();
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let emb = emb.iter().map(|x| x / norm).collect();
        MemoryUnit {
            id: Uuid::new_v4(),
            content: format!("content for {agent_id}"),
            memory_type: MemoryType::Episodic,
            agent_id: agent_id.to_string(),
            scope: MemoryScope::Private,
            importance: 0.8,
            embedding: Some(emb),
            metadata: serde_json::json!({}),
            created_at: Utc::now(),
            accessed_at: Utc::now(),
            access_count: 0,
            decay_score: 1.0,
            team_id: None,
        }
    }

    #[test]
    fn open_creates_directories() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path(), small_config()).unwrap();
        assert!(dir.path().join("wal").exists());
        assert!(dir.path().join("episodic").exists());
        assert!(dir.path().join("semantic").exists());
        assert!(dir.path().join("procedural").exists());
        engine.close().unwrap();
    }

    #[test]
    fn data_survives_force_close_and_reopen() {
        let dir = tempdir().unwrap();
        let id;

        // First session: write data, drop without checkpoint
        {
            let engine = Engine::open(dir.path(), small_config()).unwrap();
            let unit = make_unit("a1", 8);
            id = unit.id;
            engine.episodic.add(&unit).unwrap();
            // Drop without calling close() — simulates crash after WAL fsync
        }

        // Second session: reopen and verify
        let engine2 = Engine::open(dir.path(), small_config()).unwrap();
        let fetched = engine2.episodic.get(id).unwrap();
        assert!(fetched.is_some(), "record must survive force-close via WAL replay");
        assert_eq!(fetched.unwrap().agent_id, "a1");
        engine2.close().unwrap();
    }

    #[test]
    fn hnsw_rebuilt_after_wal_replay() {
        let dir = tempdir().unwrap();

        {
            let engine = Engine::open(dir.path(), small_config()).unwrap();
            for _ in 0..5 {
                engine.episodic.add(&make_unit("a1", 8)).unwrap();
            }
            // Drop without checkpoint
        }

        let engine2 = Engine::open(dir.path(), small_config()).unwrap();
        // HNSW should have 5 vectors after replay
        assert_eq!(engine2.episodic.hnsw_len(), 5);
        engine2.close().unwrap();
    }

    #[test]
    fn checkpoint_then_reopen_restores_snapshot() {
        let dir = tempdir().unwrap();

        {
            let engine = Engine::open(dir.path(), small_config()).unwrap();
            for _ in 0..3 {
                engine.episodic.add(&make_unit("a1", 8)).unwrap();
            }
            engine.checkpoint().unwrap(); // writes snapshot + Checkpoint WAL record

            // Add 2 more after checkpoint (these go into WAL only, not the snapshot)
            engine.episodic.add(&make_unit("a1", 8)).unwrap();
            engine.episodic.add(&make_unit("a1", 8)).unwrap();
            // Drop without a second checkpoint
        }

        let engine2 = Engine::open(dir.path(), small_config()).unwrap();
        // Should see all 5: 3 from snapshot + 2 from WAL replay
        assert_eq!(engine2.episodic.hnsw_len(), 5);
        assert_eq!(engine2.episodic.count(Some("a1")).unwrap(), 5);
        engine2.close().unwrap();
    }

    #[test]
    fn delete_survives_reopen() {
        let dir = tempdir().unwrap();
        let id;

        {
            let engine = Engine::open(dir.path(), small_config()).unwrap();
            let unit = make_unit("a1", 8);
            id = unit.id;
            engine.episodic.add(&unit).unwrap();
            engine.episodic.delete(id).unwrap();
        }

        let engine2 = Engine::open(dir.path(), small_config()).unwrap();
        assert!(engine2.episodic.get(id).unwrap().is_none());
        assert_eq!(engine2.episodic.count(None).unwrap(), 0);
        engine2.close().unwrap();
    }

    #[test]
    fn close_is_idempotent() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path(), small_config()).unwrap();
        engine.close().unwrap();
        engine.close().unwrap(); // second close should not panic
    }
}
