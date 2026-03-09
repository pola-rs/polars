//! Spill cleanup has three layers, each a safety net for the one above:
//!
//! 1. deletes the individual spill file (`Token::drop`).
//! 2. deletes the process spill directory (end of query).
//! 3. on startup, scans for directories left by dead processes
//!    (killed, OOM, power loss, etc.).
//!
//! All three layers are non-blocking: callers push a [`CleanRequest`]
//! into a channel and return immediately. A single background thread
//! drains the channel and performs the actual filesystem I/O. Layer 3
//! runs automatically when the cleaner thread starts.

use std::path::PathBuf;
use std::sync::{LazyLock, mpsc};

enum CleanRequest {
    File(PathBuf),
    Directory(PathBuf),
    Flush(mpsc::SyncSender<()>),
}

static CLEANER: LazyLock<mpsc::Sender<CleanRequest>> = LazyLock::new(|| {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("polars-ooc-cleaner".into())
        .spawn(move || run(rx))
        .expect("failed to spawn polars-ooc cleaner thread");
    tx
});

fn run(rx: mpsc::Receiver<CleanRequest>) {
    cleanup_stale_dirs();

    while let Ok(req) = rx.recv() {
        match req {
            CleanRequest::File(p) => {
                let _ = std::fs::remove_file(&p);
            },
            CleanRequest::Directory(p) => {
                let _ = std::fs::remove_dir_all(&p);
            },
            CleanRequest::Flush(done) => {
                let _ = done.send(());
            },
        }
    }
}

/// Non-blocking file deletion.
pub(crate) fn delete_file(path: PathBuf) {
    let _ = CLEANER.send(CleanRequest::File(path));
}

/// Non-blocking recursive directory deletion.
pub(crate) fn delete_directory(path: PathBuf) {
    let _ = CLEANER.send(CleanRequest::Directory(path));
}

/// Ensure the cleaner thread has been spawned.
pub(crate) fn init() {
    LazyLock::force(&CLEANER);
}

/// Block until all pending deletes have been processed.
pub(crate) fn shutdown() {
    let (tx, rx) = mpsc::sync_channel(1);
    let _ = CLEANER.send(CleanRequest::Flush(tx));
    let _ = rx.recv();
}

/// Delete spill directories left behind by dead processes.
///
/// Each subdirectory is named by its owning PID. Skips our own PID and
/// any directory whose PID is still alive.
fn cleanup_stale_dirs() {
    let spill_dir = polars_config::config().ooc_spill_dir();
    let Ok(entries) = std::fs::read_dir(&spill_dir) else {
        return;
    };

    let our_pid = std::process::id();
    entries
        .flatten()
        .filter_map(|e| {
            let pid = e.file_name().to_str()?.parse::<u32>().ok()?;
            (pid != our_pid && !polars_utils::sys::is_process_alive(pid)).then(|| e.path())
        })
        .for_each(|p| {
            let _ = std::fs::remove_dir_all(&p);
        });
}
