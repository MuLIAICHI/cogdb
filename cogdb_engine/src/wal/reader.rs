use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use bincode::config::standard;
use crc32fast::Hasher;

use crate::error::{CogError, Result};
use super::record::WalRecord;
use super::writer::MAGIC;

/// Reads all valid WAL records from a segment file sequentially.
///
/// Stops at the first truncated or corrupt record (crash-safe: partial writes
/// at the tail are silently dropped).
pub struct WalReader {
    reader: BufReader<File>,
    /// Sequence number of the first record in this segment (from header).
    pub seq_start: u64,
}

impl WalReader {
    /// Open a WAL segment file for sequential replay.
    ///
    /// # Example
    /// ```no_run
    /// use std::path::Path;
    /// use cogdb_engine::wal::reader::WalReader;
    /// let reader = WalReader::open(Path::new("/tmp/test.wal")).unwrap();
    /// ```
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Validate magic
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(CogError::Wal(format!(
                "bad magic in WAL segment {:?}",
                path
            )));
        }

        // Read version (2 bytes) — currently unused but reserved
        let mut version_bytes = [0u8; 2];
        reader.read_exact(&mut version_bytes)?;

        // Skip engine_id (16 bytes)
        let mut engine_id = [0u8; 16];
        reader.read_exact(&mut engine_id)?;

        // Read seq_start (8 bytes)
        let mut seq_bytes = [0u8; 8];
        reader.read_exact(&mut seq_bytes)?;
        let seq_start = u64::from_le_bytes(seq_bytes);

        Ok(Self { reader, seq_start })
    }

    /// Read all valid records from the segment.
    ///
    /// Silently truncates at the first corrupt or incomplete record —
    /// this handles crash-truncated tails safely.
    pub fn read_all(&mut self) -> Result<Vec<WalRecord>> {
        let mut records = Vec::new();
        loop {
            match self.read_one() {
                Ok(Some(record)) => records.push(record),
                Ok(None) => break,          // clean EOF
                Err(_) => break,            // corrupt/truncated tail — stop here
            }
        }
        Ok(records)
    }

    fn read_one(&mut self) -> Result<Option<WalRecord>> {
        // Read record length
        let mut len_bytes = [0u8; 4];
        match self.reader.read_exact(&mut len_bytes) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        let record_len = u32::from_le_bytes(len_bytes) as usize;

        // Read CRC
        let mut crc_bytes = [0u8; 4];
        self.reader.read_exact(&mut crc_bytes)?;
        let expected_crc = u32::from_le_bytes(crc_bytes);

        // Read body
        let mut body = vec![0u8; record_len];
        self.reader.read_exact(&mut body)?;

        // Validate CRC
        let mut h = Hasher::new();
        h.update(&body);
        let actual_crc = h.finalize();
        if actual_crc != expected_crc {
            return Err(CogError::Wal(format!(
                "CRC mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}"
            )));
        }

        // Decode
        let (record, _): (WalRecord, _) =
            bincode::decode_from_slice(&body, standard())?;

        Ok(Some(record))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use uuid::Uuid;
    use crate::wal::writer::WalWriter;
    use crate::wal::record::WalRecord;

    fn write_records(path: &Path, n: usize) -> Vec<u64> {
        let writer = WalWriter::open(path, 0, 64 * 1024 * 1024).unwrap();
        let mut seqs = Vec::new();
        for _ in 0..n {
            let id = WalRecord::uuid_to_id_bytes(&Uuid::new_v4());
            let seq = writer
                .append(WalRecord::EpisodicUpsert {
                    seq: 0,
                    id,
                    embedding: vec![0.1, 0.2, 0.3],
                    metadata_json: r#"{"agent_id":"a1"}"#.to_string(),
                })
                .unwrap();
            seqs.push(seq);
        }
        seqs
    }

    #[test]
    fn write_then_read_all() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("seg.wal");
        let seqs = write_records(&path, 5);
        assert_eq!(seqs, vec![0, 1, 2, 3, 4]);

        let mut reader = WalReader::open(&path).unwrap();
        assert_eq!(reader.seq_start, 0);
        let records = reader.read_all().unwrap();
        assert_eq!(records.len(), 5);
        for (i, r) in records.iter().enumerate() {
            assert_eq!(r.seq(), i as u64);
        }
    }

    #[test]
    fn crash_truncated_tail_is_tolerated() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("trunc.wal");
        write_records(&path, 3);

        // Truncate the file mid-record to simulate a crash
        let original_len = std::fs::metadata(&path).unwrap().len();
        let truncated_len = original_len - 10; // chop 10 bytes off the end
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(truncated_len).unwrap();

        let mut reader = WalReader::open(&path).unwrap();
        let records = reader.read_all().unwrap();
        // Should recover only the complete records (2 of 3 at minimum)
        assert!(records.len() >= 2);
        assert!(records.len() <= 3);
    }

    #[test]
    fn checkpoint_record_survives_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("chk.wal");
        let writer = WalWriter::open(&path, 0, 64 * 1024 * 1024).unwrap();
        writer
            .append(WalRecord::Checkpoint {
                seq: 0,
                snapshot_path: "snapshots/snap-0".to_string(),
                snapshot_crc: 0xABCD_1234,
            })
            .unwrap();

        let mut reader = WalReader::open(&path).unwrap();
        let records = reader.read_all().unwrap();
        assert_eq!(records.len(), 1);
        match &records[0] {
            WalRecord::Checkpoint { snapshot_path, snapshot_crc, .. } => {
                assert_eq!(snapshot_path, "snapshots/snap-0");
                assert_eq!(*snapshot_crc, 0xABCD_1234);
            }
            _ => panic!("expected Checkpoint"),
        }
    }

    #[test]
    fn bad_magic_returns_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.wal");
        std::fs::write(&path, b"BADMAGIC00000000000000000000000000").unwrap();
        let result = WalReader::open(&path);
        assert!(result.is_err());
    }
}
