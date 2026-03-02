use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use polars_config::SpillFormat;
use polars_core::prelude::DataFrame;
use polars_io::ipc::{IpcReader, IpcWriter};
use polars_io::path_utils::create_dir_owner_only;
use polars_io::{SerReader, SerWriter};
use slotmap::Key;

use crate::cleaner;
use crate::memory_manager::DfKey;
use crate::token::Token;

/// On-disk layout:
///
/// ```text
/// <spill_dir>/
///   <pid>/                        ← process directory (one per OS process)
///     query_<id>/                 ← query directory (one per QueryGuard)
///       spill_<thread>_<key>.ipc  ← individual spill file (one per Token)
/// ```
pub struct Spiller {
    #[allow(dead_code)]
    format: SpillFormat,
    process_dir: PathBuf,
    active_query_id: AtomicU64,
    /// Escalating spill-to-disk aggressiveness to free memory. Each spill
    /// pass bumps this 0→1→2 (saturates at 2), increasing the fraction of
    /// the budget spilled to disk: 0 = 1/8, 1 = 1/4, 2+ = 1/2. Reset to
    /// 0 on `begin_query()`.
    ///
    /// TODO: The fractions and number of levels are initial guesses.
    /// Needs benchmarking with real workloads to tune.
    spill_level: AtomicU64,
}

impl Spiller {
    pub fn new(format: SpillFormat) -> Self {
        let spill_dir = polars_config::config().ooc_spill_dir();

        create_dir_owner_only(&spill_dir).unwrap_or_else(|e| {
            panic!("failed to create OOC spill directory: {e} (path = {spill_dir:?})")
        });
        cleaner::cleanup_stale_dirs(&spill_dir);

        Self {
            format,
            process_dir: spill_dir.join(std::process::id().to_string()),
            active_query_id: Default::default(),
            spill_level: Default::default(),
        }
    }

    /// Prepare for a new query: set the active query ID, reset spill
    /// escalation, and create the query's spill directory.
    pub fn begin_query(&self, query_id: u64) {
        self.active_query_id.store(query_id, Ordering::Relaxed);
        self.spill_level.store(0, Ordering::Relaxed);
        let query_dir = self.query_dir(query_id);
        create_dir_owner_only(&query_dir).unwrap_or_else(|e| {
            panic!("failed to create query spill directory: {e} (path = {query_dir:?})")
        });
    }

    /// Return the fraction of the budget to free and escalate for the next
    /// pass: 1/8 → 1/4 → 1/2 (stays at 1/2).
    pub fn spill_fraction_and_escalate(&self) -> f64 {
        const FRACTIONS: [f64; 3] = [1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0];
        let level = self.spill_level.fetch_add(1, Ordering::Relaxed).min(2);
        FRACTIONS[level as usize]
    }

    pub(crate) fn query_dir(&self, query_id: u64) -> PathBuf {
        self.process_dir.join(format!("query_{query_id}"))
    }

    fn file_path(&self, thread_idx: u64, key: DfKey) -> PathBuf {
        let qid = self.active_query_id.load(Ordering::Relaxed);
        self.query_dir(qid)
            .join(format!("spill_{}_{}.ipc", thread_idx, key.data().as_ffi()))
    }

    /// Spill a DataFrame to disk.
    pub fn spill(&self, thread_idx: u64, key: DfKey, mut df: DataFrame) {
        let path = self.file_path(thread_idx, key);
        let mut file = std::fs::File::create(&path)
            .unwrap_or_else(|e| panic!("failed to create spill file {path:?}: {e}"));
        IpcWriter::new(&mut file)
            .finish(&mut df)
            .unwrap_or_else(|e| panic!("failed to write spill file {path:?}: {e}"));
    }

    /// Load a previously spilled DataFrame from disk. Removes the file after reading.
    ///
    /// Currently delegates to [`load_blocking`](Self::load_blocking). The async
    /// signature exists so callers in async contexts (e.g. `MemoryManager::take`)
    /// won't need changes when this is replaced with true async I/O (e.g.
    /// `tokio::fs` or io_uring).
    pub async fn load(&self, token: &Token) -> DataFrame {
        self.load_blocking(token)
    }

    /// Blocking variant of [`load`](Self::load). Used from sync contexts
    /// (e.g. `update_state()`, rayon tasks).
    pub fn load_blocking(&self, token: &Token) -> DataFrame {
        let path = self.file_path(token.thread_idx(), token.key);
        let file = std::fs::File::open(&path)
            .unwrap_or_else(|e| panic!("failed to open spill file {path:?}: {e}"));
        let df = IpcReader::new(file)
            .finish()
            .unwrap_or_else(|e| panic!("failed to read spill file {path:?}: {e}"));
        cleaner::delete_file(&path);
        df
    }

    /// Best-effort deletion of a spill file.
    pub fn delete_spill_file(&self, token: &Token) {
        cleaner::delete_file(&self.file_path(token.thread_idx(), token.key));
    }

    /// Best-effort deletion of a query's spill directory on a background thread.
    pub fn delete_query_dir_background(&self, query_id: u64) {
        cleaner::delete_dir_background(self.query_dir(query_id));
    }
}

impl Drop for Spiller {
    fn drop(&mut self) {
        cleaner::delete_dir_background(self.process_dir.clone());
    }
}
