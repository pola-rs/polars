//! Spill cleanup has four layers, each a safety net for the one above:
//!
//! 1. deletes the individual spill file (`Token::drop`).
//! 2. deletes the query spill directory (`QueryGuard::drop`).
//! 3. deletes the process spill directory (`Spiller::drop`).
//! 4. on startup, scans for directories left by dead processes
//!    (killed, OOM, power loss, etc.).

use std::path::{Path, PathBuf};

/// Best-effort file deletion.
pub(crate) fn delete_file(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// Best-effort directory deletion on a background thread.
pub(crate) fn delete_dir_background(dir: PathBuf) {
    std::thread::spawn(move || {
        let _ = std::fs::remove_dir_all(&dir);
    });
}

/// Delete spill directories left behind by dead processes.
///
/// Each subdirectory is named by its owning PID. Skips our own PID and
/// any directory whose PID is still alive.
pub(crate) fn cleanup_stale_dirs(spill_dir: &Path) {
    let Ok(entries) = std::fs::read_dir(spill_dir) else {
        return;
    };

    let our_pid = std::process::id();
    entries
        .flatten()
        .filter_map(|e| {
            let pid = e.file_name().to_str()?.parse::<u32>().ok()?;
            (pid != our_pid && !polars_utils::sys::is_process_alive(pid)).then(|| e.path())
        })
        .for_each(delete_dir_background);
}
