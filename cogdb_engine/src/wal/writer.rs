use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use bincode::config::standard;
use crc32fast::Hasher;
use parking_lot::Mutex;

use crate::error::Result;
use super::record::WalRecord;

/// Magic bytes written at the start of every WAL segment file.
pub const MAGIC: &[u8; 8] = b"COGDBWAL";
pub const WAL_VERSION: u16 = 1;

/// Writes WAL records to an append-only segment file.
///
/// Each record is framed as: [record_len: u32][crc32: u32][body: bincode bytes].
/// The segment header is written once on creation.
///
/// Thread-safe: the inner writer is protected by a Mutex.
pub struct WalWriter {
    segment_path: PathBuf,
    inner: Mutex<BufWriter<File>>,
    seq: AtomicU64,
    bytes_written: AtomicU64,
    pub max_segment_bytes: u64,
}

impl WalWriter {
    /// Open or create a WAL segment at `path`.
    ///
    /// If the file does not exist, writes the segment header.
    /// If the file exists, appends to it (recovery has already been run).
    ///
    /// # Example
    /// ```no_run
    /// use std::path::Path;
    /// use cogdb_engine::wal::writer::WalWriter;
    /// let writer = WalWriter::open(Path::new("/tmp/test.wal"), 0, 64 * 1024 * 1024).unwrap();
    /// ```
    pub fn open(path: &Path, seq_start: u64, max_segment_bytes: u64) -> Result<Self> {
        let is_new = !path.exists();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let file_len = file.metadata()?.len();
        let mut writer = BufWriter::new(file);

        if is_new {
            // Write segment header
            writer.write_all(MAGIC)?;
            writer.write_all(&WAL_VERSION.to_le_bytes())?;
            // engine_id placeholder (16 zero bytes for now; set by Engine at open time)
            writer.write_all(&[0u8; 16])?;
            writer.write_all(&seq_start.to_le_bytes())?;
            writer.flush()?;
        }

        Ok(Self {
            segment_path: path.to_path_buf(),
            inner: Mutex::new(writer),
            seq: AtomicU64::new(seq_start),
            bytes_written: AtomicU64::new(file_len),
            max_segment_bytes,
        })
    }

    /// Append a WAL record.  Assigns the next sequence number and fsyncs.
    ///
    /// Returns the sequence number assigned to this record.
    pub fn append(&self, record: WalRecord) -> Result<u64> {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);

        // Update seq field in record
        let record = Self::set_seq(record, seq);

        let body = bincode::encode_to_vec(&record, standard())?;
        let crc = {
            let mut h = Hasher::new();
            h.update(&body);
            h.finalize()
        };

        let record_len = body.len() as u32;
        let mut guard = self.inner.lock();
        guard.write_all(&record_len.to_le_bytes())?;
        guard.write_all(&crc.to_le_bytes())?;
        guard.write_all(&body)?;
        guard.flush()?;
        // fsync — guarantees durability before returning
        guard.get_ref().sync_all()?;

        let written = 4 + 4 + body.len() as u64;
        self.bytes_written.fetch_add(written, Ordering::Relaxed);

        Ok(seq)
    }

    /// Returns true if the segment has exceeded the max size threshold.
    pub fn needs_rotation(&self) -> bool {
        self.bytes_written.load(Ordering::Relaxed) >= self.max_segment_bytes
    }

    /// Current sequence counter (next seq that will be assigned).
    pub fn current_seq(&self) -> u64 {
        self.seq.load(Ordering::SeqCst)
    }

    pub fn segment_path(&self) -> &Path {
        &self.segment_path
    }

    fn set_seq(record: WalRecord, seq: u64) -> WalRecord {
        match record {
            WalRecord::EpisodicUpsert { id, embedding, metadata_json, .. } => {
                WalRecord::EpisodicUpsert { seq, id, embedding, metadata_json }
            }
            WalRecord::EpisodicDelete { id, .. } => WalRecord::EpisodicDelete { seq, id },
            WalRecord::TripleUpsert { id, subject, predicate, object, valid_until_ms, .. } => {
                WalRecord::TripleUpsert { seq, id, subject, predicate, object, valid_until_ms }
            }
            WalRecord::TripleDelete { id, .. } => WalRecord::TripleDelete { seq, id },
            WalRecord::ProceduralUpsert { id, .. } => WalRecord::ProceduralUpsert { seq, id },
            WalRecord::Checkpoint { snapshot_path, snapshot_crc, .. } => {
                WalRecord::Checkpoint { seq, snapshot_path, snapshot_crc }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use uuid::Uuid;
    use super::super::record::WalRecord;

    #[test]
    fn write_and_file_exists() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");
        let writer = WalWriter::open(&path, 0, 64 * 1024 * 1024).unwrap();
        let id = Uuid::new_v4();
        writer
            .append(WalRecord::EpisodicUpsert {
                seq: 0,
                id: WalRecord::uuid_to_id_bytes(&id),
                embedding: vec![0.1, 0.2],
                metadata_json: "{}".to_string(),
            })
            .unwrap();
        assert!(path.exists());
        assert!(path.metadata().unwrap().len() > 32); // header + record
    }

    #[test]
    fn seq_increments() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");
        let writer = WalWriter::open(&path, 0, 64 * 1024 * 1024).unwrap();
        let id = [0u8; 16];
        let s1 = writer.append(WalRecord::EpisodicDelete { seq: 0, id }).unwrap();
        let s2 = writer.append(WalRecord::EpisodicDelete { seq: 0, id }).unwrap();
        assert_eq!(s1, 0);
        assert_eq!(s2, 1);
        assert_eq!(writer.current_seq(), 2);
    }

    #[test]
    fn needs_rotation_when_over_limit() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tiny.wal");
        // Set a very small limit so any write triggers rotation
        let writer = WalWriter::open(&path, 0, 1).unwrap();
        let id = [0u8; 16];
        writer.append(WalRecord::EpisodicDelete { seq: 0, id }).unwrap();
        assert!(writer.needs_rotation());
    }
}
