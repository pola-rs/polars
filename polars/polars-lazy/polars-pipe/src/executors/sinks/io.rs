use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crossbeam_channel::{bounded, Sender};
use polars_core::prelude::*;
use polars_core::utils::arrow::temporal_conversions::SECONDS_IN_DAY;
use polars_core::POOL;
use polars_io::prelude::*;

pub(in crate::executors::sinks) type DfIter =
    Box<dyn ExactSizeIterator<Item = DataFrame> + Sync + Send>;
// The Option<IdxCa> are the partitions it should be written to, if any
type Payload = (Option<IdxCa>, DfIter);

/// A helper that can be used to spill to disk
pub(crate) struct IOThread {
    sender: Sender<Payload>,
    // ensures the directory is not GC'ed
    _lockfile: LockFile,
    pub(in crate::executors::sinks) dir: PathBuf,
    pub(in crate::executors::sinks) sent: Arc<AtomicUsize>,
    pub(in crate::executors::sinks) total: Arc<AtomicUsize>,
}

fn get_lockfile_path(dir: &Path) -> PathBuf {
    let mut lockfile_path = dir.to_path_buf();
    lockfile_path.push(".lock");
    lockfile_path
}

/// Starts a new thread that will clean up operations of directories that don't
/// have a lockfile (opened with 'w' permissions).
fn gc_thread(operation_name: &'static str) {
    let _ = std::thread::spawn(move || {
        let dir = resolve_homedir(Path::new(&format!("~/.polars/{operation_name}/")));

        // if the directory does not exist, there is nothing to clean
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                let lockfile_path = get_lockfile_path(&path);

                if let Ok(lockfile) = File::open(lockfile_path) {
                    // lockfile can be read
                    if let Ok(time) = lockfile.metadata().unwrap().modified() {
                        let modified_since =
                            SystemTime::now().duration_since(time).unwrap().as_secs();
                        // the lockfile can still exist if a process was canceled
                        // so we also check the modified date
                        // we don't expect queries that run a month
                        if modified_since > (SECONDS_IN_DAY as u64 * 30) {
                            std::fs::remove_dir_all(path).unwrap()
                        }
                    } else {
                        eprintln!("could not modified time on this platform")
                    }
                } else {
                    std::fs::remove_dir_all(path).unwrap()
                }
            }
        }
    });
}

impl IOThread {
    pub(in crate::executors::sinks) fn try_new(
        // Schema of the file that will be dumped to disk
        schema: SchemaRef,
        // Will be used as subdirectory name in `~/.polars/`
        operation_name: &'static str,
    ) -> PolarsResult<Self> {
        let uuid = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        // start a thread that will clean up old dumps.
        // TODO: if we will have more ooc in the future  we will have a dedicated GC thread
        gc_thread(operation_name);

        let dir = resolve_homedir(Path::new(&format!("~/.polars/{operation_name}/{uuid}")));
        std::fs::create_dir_all(&dir)?;

        let lockfile_path = get_lockfile_path(&dir);
        let lockfile = LockFile::new(lockfile_path)?;

        // we need some pushback otherwise we still could go OOM.
        let (sender, receiver) = bounded::<Payload>(POOL.current_num_threads() * 2);

        let sent: Arc<AtomicUsize> = Default::default();
        let total: Arc<AtomicUsize> = Default::default();

        let dir2 = dir.clone();
        let total2 = total.clone();
        std::thread::spawn(move || {
            let mut count = 0usize;
            while let Ok((partitions, iter)) = receiver.recv() {
                if let Some(partitions) = partitions {
                    for (part, df) in partitions.into_no_null_iter().zip(iter) {
                        let mut path = dir2.clone();
                        path.push(format!("{part}"));

                        let _ = std::fs::create_dir(&path);
                        path.push(format!("{count}.ipc"));

                        let file = std::fs::File::create(path).unwrap();
                        let writer = IpcWriter::new(file);
                        let mut writer = writer.batched(&schema).unwrap();
                        writer.write_batch(&df).unwrap();
                        writer.finish().unwrap();
                        count += 1;
                    }
                } else {
                    let mut path = dir2.clone();
                    path.push(format!("{count}.ipc"));

                    let file = std::fs::File::create(path).unwrap();
                    let writer = IpcWriter::new(file);
                    let mut writer = writer.batched(&schema).unwrap();

                    for df in iter {
                        writer.write_batch(&df).unwrap();
                    }
                    writer.finish().unwrap();

                    count += 1;
                }
                total2.store(count, Ordering::Relaxed);
            }
        });

        Ok(Self {
            sender,
            dir,
            sent,
            total,
            _lockfile: lockfile,
        })
    }

    pub(in crate::executors::sinks) fn dump_chunk(&self, df: DataFrame) {
        let iter = Box::new(std::iter::once(df));
        self.dump_iter(None, iter)
    }
    pub(in crate::executors::sinks) fn dump_iter(&self, partition: Option<IdxCa>, iter: DfIter) {
        let add = iter.size_hint().1.unwrap();
        self.sender.send((partition, iter)).unwrap();
        self.sent.fetch_add(add, Ordering::Relaxed);
    }
}

pub(in crate::executors::sinks) fn block_thread_until_io_thread_done(io_thread: &IOThread) {
    // get number sent
    let sent = io_thread.sent.load(Ordering::Relaxed);
    // get number processed
    while io_thread.total.load(Ordering::Relaxed) != sent {
        std::thread::park_timeout(Duration::from_millis(6))
    }
}

struct LockFile {
    path: PathBuf,
}

impl LockFile {
    fn new(path: PathBuf) -> PolarsResult<Self> {
        if File::create(&path).is_ok() {
            Ok(Self { path })
        } else {
            polars_bail!(ComputeError: "could not create lockfile")
        }
    }
}

impl Drop for LockFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).unwrap()
    }
}
