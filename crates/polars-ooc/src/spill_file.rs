use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::sync::mpsc::{Receiver, Sender, channel};

use polars_io::create_dir_owner_only;

/// On-disk layout:
///
/// ```text
/// <spill_dir>/
///   <pid>/                     <- process directory (one per OS process)
///     spill-<ctx>-<uuid>.ipc   <- individual spill file (unique per spill)
/// ```

static SPILL_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
    let spill_dir = polars_config::config().ooc_spill_dir();
    let process_dir = spill_dir.join(std::process::id().to_string());
    create_dir_owner_only(&process_dir).unwrap_or_else(|e| {
        panic!("failed to create spill directory: {e} (path = {process_dir:?})")
    });
    process_dir
});

pub struct SpillFile {
    path: PathBuf,
}

impl SpillFile {
    pub fn new(context_id: &str, ext: &str) -> Self {
        let uuid = uuid::Uuid::now_v7();
        Self {
            path: SPILL_DIR
                .join(format!(
                    "spill-{context_id}-{uuid}.{ext}",
                    uuid = uuid.as_hyphenated()
                ))
                .with_extension(ext),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for SpillFile {
    fn drop(&mut self) {
        SPILL_CLEANER
            .send_rq
            .send(CleanRequest::File(core::mem::take(&mut self.path)))
            .unwrap();
    }
}

struct SpillCleaner {
    send_rq: Sender<CleanRequest>,
}

impl SpillCleaner {
    fn run(recv_rq: Receiver<CleanRequest>) {
        cleanup_stale_dirs();

        while let Ok(rq) = recv_rq.recv() {
            match rq {
                CleanRequest::File(p) => {
                    if let Err(e) = std::fs::remove_file(&p) {
                        if polars_config::config().verbose() {
                            eprintln!("Error while removing spill file '{}': {e}", p.display());
                        }
                    }
                },
                CleanRequest::Directory(p) => {
                    if let Err(e) = std::fs::remove_dir_all(&p) {
                        if polars_config::config().verbose() {
                            eprintln!(
                                "Error while removing spill directory '{}': {e}",
                                p.display()
                            );
                        }
                    }
                },
                CleanRequest::Flush(ack) => drop(ack.send(())),
            }
        }
    }
}

static SPILL_CLEANER: LazyLock<SpillCleaner> = LazyLock::new(|| {
    let (send_rq, recv_rq) = channel();

    std::thread::Builder::new()
        .name("polars-ooc-cleaner".into())
        .spawn(move || SpillCleaner::run(recv_rq))
        .expect("failed to spawn polars-ooc cleaner thread");

    SpillCleaner { send_rq }
});

enum CleanRequest {
    File(PathBuf),
    #[expect(unused)]
    Directory(PathBuf),
    Flush(Sender<()>),
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
    let dead_pid_dirs = entries.flatten().filter_map(|e| {
        let pid = e.file_name().to_str()?.parse::<u32>().ok()?;
        (pid != our_pid && !polars_utils::sys::is_process_alive(pid)).then(|| e.path())
    });
    for path in dead_pid_dirs {
        let _ = std::fs::remove_dir_all(&path);
    }
}

/// Ensures the out-of-core cleanup thread is started.
///
/// This will happen automatically when new garbage is created, but if garbage
/// is left over from a previously crashed instance, it can be beneficial to
/// start early.
pub fn init_ooc_cleaner() {
    LazyLock::force(&SPILL_CLEANER);
}

/// Wait for all dead files for out-of-core execution to be cleaned up.
pub fn flush_ooc_cleanup() {
    let (ack_send, ack_recv) = channel();
    if SPILL_CLEANER
        .send_rq
        .send(CleanRequest::Flush(ack_send))
        .is_err()
    {
        return;
    }

    let _ = ack_recv.recv();
}
