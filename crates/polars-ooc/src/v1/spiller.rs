use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use polars_config::SpillFormat;
use polars_core::prelude::DataFrame;
use polars_io::ipc::{IpcReader, IpcWriter};
use polars_io::path_utils::create_dir_owner_only;
use polars_io::{SerReader, SerWriter};

use super::cleaner;

/// On-disk layout:
///
/// ```text
/// <spill_dir>/
///   <pid>/                            ← process directory (one per OS process)
///     spill_<index>_<gen>_<seq>.ipc   ← individual spill file (unique per spill)
/// ```
pub struct Spiller {
    #[allow(dead_code)]
    format: SpillFormat,
    process_dir: PathBuf,
    /// Escalating spill-to-disk aggressiveness to free memory. Each spill
    /// pass bumps this 0→1→2 (saturates at 2), increasing the fraction of
    /// the budget spilled to disk: 0 = 1/8, 1 = 1/4, 2+ = 1/2.
    ///
    /// TODO: The fractions and number of levels are initial guesses.
    /// Needs benchmarking with real workloads to tune.
    spill_level: AtomicU64,
}

impl Spiller {
    pub fn new(format: SpillFormat) -> Self {
        let spill_dir = polars_config::config().ooc_spill_dir();

        let process_dir = spill_dir.join(std::process::id().to_string());
        create_dir_owner_only(&process_dir).unwrap_or_else(|e| {
            panic!("failed to create spill directory: {e} (path = {process_dir:?})")
        });
        register_atexit();
        cleaner::init();

        Self {
            format,
            process_dir,
            spill_level: Default::default(),
        }
    }

    /// Return the fraction of the budget to free and escalate for the next
    /// pass: 1/8 → 1/4 → 1/2 (stays at 1/2).
    pub fn spill_fraction_and_escalate(&self) -> f64 {
        const FRACTIONS: [f64; 3] = [1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0];
        let level = self.spill_level.fetch_add(1, Ordering::Relaxed).min(2);
        FRACTIONS[level as usize]
    }

    /// Reset spill escalation level back to 0.
    pub fn reset_spill_level(&self) {
        self.spill_level.store(0, Ordering::Relaxed);
    }

    fn file_path(&self, index: u32, generation: u32, seq: u32) -> PathBuf {
        self.process_dir
            .join(format!("spill_{index}_{generation}_{seq}.ipc"))
    }

    /// Spill a DataFrame to disk.
    pub fn spill(&self, index: u32, generation: u32, seq: u32, mut df: DataFrame) {
        let path = self.file_path(index, generation, seq);
        let mut file = std::fs::File::create(&path)
            .unwrap_or_else(|e| panic!("failed to create spill file {path:?}: {e}"));
        IpcWriter::new(&mut file)
            .finish(&mut df)
            .unwrap_or_else(|e| panic!("failed to write spill file {path:?}: {e}"));
    }

    /// Load a previously spilled DataFrame from disk. Removes the file after reading.
    pub fn load_blocking(&self, index: u32, generation: u32, seq: u32) -> DataFrame {
        let path = self.file_path(index, generation, seq);
        let file = std::fs::File::open(&path)
            .unwrap_or_else(|e| panic!("failed to open spill file {path:?}: {e}"));
        let df = IpcReader::new(file)
            .finish()
            .unwrap_or_else(|e| panic!("failed to read spill file {path:?}: {e}"));
        cleaner::delete_file(path);
        df
    }

    /// Best-effort deletion of a spill file.
    pub fn delete_spill_file(&self, index: u32, generation: u32, seq: u32) {
        cleaner::delete_file(self.file_path(index, generation, seq));
    }
}

/// Register an atexit handler that deletes the process spill directory.
/// SIGKILL/OOM are handled by Layer 3 (`cleanup_stale_dirs`) on next startup.
fn register_atexit() {
    extern "C" fn cleanup_on_exit() {
        let dir = polars_config::config()
            .ooc_spill_dir()
            .join(std::process::id().to_string());

        // On Windows, the cleaner thread may already be terminated by the
        // CRT during process shutdown. Sending to its channel would block
        // forever or access invalid memory. Delete directly instead.
        #[cfg(target_os = "windows")]
        {
            let _ = std::fs::remove_dir_all(&dir);
        }

        #[cfg(not(target_os = "windows"))]
        {
            cleaner::delete_directory(dir);
            cleaner::shutdown();
        }
    }

    unsafe {
        libc::atexit(cleanup_on_exit);
    }
}
