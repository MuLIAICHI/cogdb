use bincode::{Decode, Encode};
use uuid::Uuid;

/// Every mutation that affects HNSW or petgraph is recorded here.
/// SQLite mutations are protected by SQLite's own WAL mode.
#[derive(Debug, Clone, Encode, Decode)]
pub enum WalRecord {
    EpisodicUpsert {
        seq: u64,
        /// UUID bytes (16 bytes, fixed size)
        id: [u8; 16],
        embedding: Vec<f32>,
        /// JSON blob of all non-vector metadata fields
        metadata_json: String,
    },
    EpisodicDelete {
        seq: u64,
        id: [u8; 16],
    },
    TripleUpsert {
        seq: u64,
        id: [u8; 16],
        subject: String,
        predicate: String,
        object: String,
        /// Unix timestamp milliseconds; None = still active
        valid_until_ms: Option<i64>,
    },
    TripleDelete {
        seq: u64,
        id: [u8; 16],
    },
    /// Fence record — actual procedural data is fully in SQLite.
    ProceduralUpsert {
        seq: u64,
        id: [u8; 16],
    },
    Checkpoint {
        seq: u64,
        /// Relative path from db_path to the snapshot bundle directory.
        snapshot_path: String,
        /// CRC32 of the snapshot bundle (all files concatenated).
        snapshot_crc: u32,
    },
}

impl WalRecord {
    pub fn seq(&self) -> u64 {
        match self {
            WalRecord::EpisodicUpsert { seq, .. } => *seq,
            WalRecord::EpisodicDelete { seq, .. } => *seq,
            WalRecord::TripleUpsert { seq, .. } => *seq,
            WalRecord::TripleDelete { seq, .. } => *seq,
            WalRecord::ProceduralUpsert { seq, .. } => *seq,
            WalRecord::Checkpoint { seq, .. } => *seq,
        }
    }

    pub fn id_bytes_to_uuid(bytes: &[u8; 16]) -> Uuid {
        Uuid::from_bytes(*bytes)
    }

    pub fn uuid_to_id_bytes(id: &Uuid) -> [u8; 16] {
        *id.as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bincode::config::standard;
    use uuid::Uuid;

    fn roundtrip(record: WalRecord) -> WalRecord {
        let encoded = bincode::encode_to_vec(&record, standard()).expect("encode");
        let (decoded, _): (WalRecord, _) =
            bincode::decode_from_slice(&encoded, standard()).expect("decode");
        decoded
    }

    #[test]
    fn episodic_upsert_roundtrip() {
        let id = Uuid::new_v4();
        let rec = WalRecord::EpisodicUpsert {
            seq: 42,
            id: WalRecord::uuid_to_id_bytes(&id),
            embedding: vec![0.1, 0.2, 0.3],
            metadata_json: r#"{"agent_id":"a1"}"#.to_string(),
        };
        let back = roundtrip(rec);
        match back {
            WalRecord::EpisodicUpsert { seq, id: back_id, embedding, .. } => {
                assert_eq!(seq, 42);
                assert_eq!(WalRecord::id_bytes_to_uuid(&back_id), id);
                assert_eq!(embedding, vec![0.1f32, 0.2, 0.3]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn triple_upsert_roundtrip() {
        let id = Uuid::new_v4();
        let rec = WalRecord::TripleUpsert {
            seq: 100,
            id: WalRecord::uuid_to_id_bytes(&id),
            subject: "Alice".to_string(),
            predicate: "works_at".to_string(),
            object: "Acme".to_string(),
            valid_until_ms: Some(1_700_000_000_000),
        };
        let back = roundtrip(rec);
        match back {
            WalRecord::TripleUpsert { seq, subject, valid_until_ms, .. } => {
                assert_eq!(seq, 100);
                assert_eq!(subject, "Alice");
                assert_eq!(valid_until_ms, Some(1_700_000_000_000));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn checkpoint_roundtrip() {
        let rec = WalRecord::Checkpoint {
            seq: 999,
            snapshot_path: "snapshots/snap-999".to_string(),
            snapshot_crc: 0xDEAD_BEEF,
        };
        let back = roundtrip(rec);
        match back {
            WalRecord::Checkpoint { seq, snapshot_path, snapshot_crc } => {
                assert_eq!(seq, 999);
                assert_eq!(snapshot_path, "snapshots/snap-999");
                assert_eq!(snapshot_crc, 0xDEAD_BEEF);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn seq_accessor() {
        let id = [0u8; 16];
        assert_eq!(WalRecord::EpisodicDelete { seq: 7, id }.seq(), 7);
        assert_eq!(
            WalRecord::Checkpoint {
                seq: 55,
                snapshot_path: "x".into(),
                snapshot_crc: 0
            }
            .seq(),
            55
        );
    }
}
