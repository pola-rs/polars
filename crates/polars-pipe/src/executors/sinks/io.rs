use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use crossbeam_channel::{bounded, Sender};
use polars_core::error::ErrString;
use polars_core::prelude::*;
use polars_core::utils::arrow::temporal_conversions::SECONDS_IN_DAY;
use polars_io::prelude::*;

use crate::executors::sinks::get_base_temp_dir;
use crate::pipeline::morsels_per_sink;

pub(in crate::executors::sinks) type DfIter =
    Box<dyn ExactSizeIterator<Item = DataFrame> + Sync + Send>;
// The Option<IdxCa> are the partitions it should be written to, if any
type Payload = (Option<IdxCa>, DfIter);

/// A helper that can be used to spill to disk
pub(crate) struct IOThread {
    sender: Sender<Payload>,
    _lockfile: Arc<LockFile>,
    pub(in crate::executors::sinks) dir: PathBuf,
    pub(in crate::executors::sinks) sent: Arc<AtomicUsize>,
    pub(in crate::executors::sinks) total: Arc<AtomicUsize>,
    pub(in crate::executors::sinks) thread_local_count: Arc<AtomicUsize>,
    schema: SchemaRef,
}

fn get_lockfile_path(dir: &Path) -> PathBuf {
    let mut lockfile_path = dir.to_path_buf();
    lockfile_path.push(".lock");
    lockfile_path
}

fn get_spill_dir(operation_name: &'static str) -> PolarsResult<PathBuf> {
    let uuid = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    let mut dir = std::path::PathBuf::from(get_base_temp_dir());
    dir.push(&format!("polars/{operation_name}/{uuid}"));

    if !dir.exists() {
        fs::create_dir_all(&dir).map_err(|err| {
            PolarsError::ComputeError(ErrString::from(format!(
                "Failed to create spill directory: {}",
                err
            )))
        })?;
    } else if !dir.is_dir() {
        return Err(PolarsError::ComputeError(
            "Specified spill path is not a directory".into(),
        ));
    }

    Ok(dir)
}

/// Starts a new thread that will clean up operations of directories that don't
/// have a lockfile (opened with 'w' permissions).
fn gc_thread(operation_name: &'static str) {
    let _ = std::thread::spawn(move || {
        let mut dir = std::path::PathBuf::from(get_base_temp_dir());
        dir.push(&format!("polars/{operation_name}"));

        // if the directory does not exist, there is nothing to clean
        let rd = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            _ => panic!("cannot find {:?}", dir),
        };

        for entry in rd {
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
        // Will be used as subdirectory name in `~/.base_dir/polars/`
        operation_name: &'static str,
    ) -> PolarsResult<Self> {
        let dir = get_spill_dir(operation_name)?;

        // make sure we create lockfile before we GC
        let lockfile_path = get_lockfile_path(&dir);
        let lockfile = Arc::new(LockFile::new(lockfile_path)?);

        // start a thread that will clean up old dumps.
        // TODO: if we will have more ooc in the future  we will have a dedicated GC thread
        gc_thread(operation_name);

        // we need some pushback otherwise we still could go OOM.
        let (sender, receiver) = bounded::<Payload>(morsels_per_sink() * 2);

        let sent: Arc<AtomicUsize> = Default::default();
        let total: Arc<AtomicUsize> = Default::default();
        let thread_local_count: Arc<AtomicUsize> = Default::default();

        let dir2 = dir.clone();
        let total2 = total.clone();
        let lockfile2 = lockfile.clone();
        let schema2 = schema.clone();
        std::thread::spawn(move || {
            let schema = schema2;
            // this moves the lockfile in the thread
            // we keep one in the thread and one in the `IoThread` struct
            let _keep_hold_on_lockfile = lockfile2;

            let mut count = 0usize;

            // We accept 2 cases. E.g.
            // 1. (None, DfIter):
            //    This will dump to `dir/count.ipc`
            // 2. (Some(partitions), DfIter)
            //    This will dump to `dir/partition/count.ipc`
            while let Ok((partitions, iter)) = receiver.recv() {
                if let Some(partitions) = partitions {
                    for (part, df) in partitions.into_no_null_iter().zip(iter) {
                        let mut path = dir2.clone();
                        path.push(format!("{part}"));

                        let _ = std::fs::create_dir(&path);
                        path.push(format!("{count}.ipc"));

                        let file = File::create(path).unwrap();
                        let writer = IpcWriter::new(file).with_pl_flavor(true);
                        let mut writer = writer.batched(&schema).unwrap();
                        writer.write_batch(&df).unwrap();
                        writer.finish().unwrap();
                        count += 1;
                    }
                } else {
                    let mut path = dir2.clone();
                    path.push(format!("{count}.ipc"));

                    let file = File::create(path).unwrap();
                    let writer = IpcWriter::new(file).with_pl_flavor(true);
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
            thread_local_count,
            schema,
        })
    }

    pub(in crate::executors::sinks) fn dump_chunk(&self, mut df: DataFrame) {
        // if IO thread is blocked
        // we write locally on this thread
        if self.sender.is_full() {
            let mut path = self.dir.clone();
            let count = self.thread_local_count.fetch_add(1, Ordering::Relaxed);
            // thread local name we start with an underscore to ensure we don't get
            // duplicates
            path.push(format!("_{count}.ipc"));

            let file = File::create(path).unwrap();
            let mut writer = IpcWriter::new(file).with_pl_flavor(true);
            writer.finish(&mut df).unwrap();
        } else {
            let iter = Box::new(std::iter::once(df));
            self.dump_iter(None, iter)
        }
    }

    pub(in crate::executors::sinks) fn dump_partition(&self, partition_no: IdxSize, df: DataFrame) {
        let partition = Some(IdxCa::from_vec("", vec![partition_no]));
        let iter = Box::new(std::iter::once(df));
        self.dump_iter(partition, iter)
    }

    pub(in crate::executors::sinks) fn dump_partition_local(
        &self,
        partition_no: IdxSize,
        df: DataFrame,
    ) {
        let count = self.thread_local_count.fetch_add(1, Ordering::Relaxed);
        let mut path = self.dir.clone();
        path.push(format!("{partition_no}"));

        let _ = std::fs::create_dir(&path);
        // thread local name we start with an underscore to ensure we don't get
        // duplicates
        path.push(format!("_{count}.ipc"));
        let file = File::create(path).unwrap();
        let writer = IpcWriter::new(file).with_pl_flavor(true);
        let mut writer = writer.batched(&self.schema).unwrap();
        writer.write_batch(&df).unwrap();
        writer.finish().unwrap();
    }

    pub(in crate::executors::sinks) fn dump_iter(&self, partition: Option<IdxCa>, iter: DfIter) {
        let add = iter.size_hint().1.unwrap();
        self.sender.send((partition, iter)).unwrap();
        self.sent.fetch_add(add, Ordering::Relaxed);
    }
}

impl Drop for IOThread {
    fn drop(&mut self) {
        // we drop the lockfile explicitly as the thread GC will leak.
        std::fs::remove_file(&self._lockfile.path).unwrap();
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
        let _ = std::fs::remove_file(&self.path);
    }
}
