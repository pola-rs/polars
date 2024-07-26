use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
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
    payload_tx: Sender<Payload>,
    cleanup_tx: Sender<PathBuf>,
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
    let id = uuid::Uuid::new_v4();

    let mut dir = std::path::PathBuf::from(get_base_temp_dir());
    dir.push(format!("polars/{operation_name}/{id}"));

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

fn clean_after_delay(time: Option<SystemTime>, secs: u64, path: &Path) {
    if let Some(time) = time {
        let modified_since = SystemTime::now().duration_since(time).unwrap().as_secs();
        if modified_since > secs {
            // This can be fallible if another thread removes this.
            // That is fine.
            let _ = std::fs::remove_dir_all(path);
        }
    } else {
        polars_warn!("could not modified time on this platform")
    }
}

/// Starts a new thread that will clean up operations of directories that don't
/// have a lockfile (opened with 'w' permissions).
fn gc_thread(operation_name: &'static str, rx: Receiver<PathBuf>) {
    let _ = std::thread::spawn(move || {
        // First clean all existing
        let mut dir = std::path::PathBuf::from(get_base_temp_dir());
        dir.push(format!("polars/{operation_name}"));

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
                    if let Ok(md) = lockfile.metadata() {
                        let time = md.modified().ok();
                        // The lockfile can still exist if a process was canceled
                        // so we also check the modified date
                        // we don't expect queries that run a month.
                        clean_after_delay(time, SECONDS_IN_DAY as u64 * 30, &path);
                    }
                } else {
                    // If path already removed, we simply continue.
                    if let Ok(md) = path.metadata() {
                        let time = md.modified().ok();
                        // Wait 15 seconds to ensure we don't remove before lockfile is created
                        // in a `collect_all` contention case
                        clean_after_delay(time, 15, &path);
                    }
                }
            }
        }

        // Clean on receive
        while let Ok(path) = rx.recv() {
            if path.is_file() {
                let res = std::fs::remove_file(path);
                debug_assert!(res.is_ok());
            } else {
                let res = std::fs::remove_dir_all(path);
                debug_assert!(res.is_ok());
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

        let (cleanup_tx, rx) = unbounded::<PathBuf>();
        // start a thread that will clean up old dumps.
        // TODO: if we will have more ooc in the future  we will have a dedicated GC thread
        gc_thread(operation_name, rx);

        // we need some pushback otherwise we still could go OOM.
        let (tx, rx) = bounded::<Payload>(morsels_per_sink() * 2);

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
            while let Ok((partitions, iter)) = rx.recv() {
                if let Some(partitions) = partitions {
                    for (part, mut df) in partitions.into_no_null_iter().zip(iter) {
                        df.shrink_to_fit();
                        let mut path = dir2.clone();
                        path.push(format!("{part}"));

                        let _ = std::fs::create_dir(&path);
                        path.push(format!("{count}.ipc"));

                        let file = File::create(path).unwrap();
                        let writer = IpcWriter::new(file).with_compat_level(CompatLevel::newest());
                        let mut writer = writer.batched(&schema).unwrap();
                        writer.write_batch(&df).unwrap();
                        writer.finish().unwrap();
                        count += 1;
                    }
                } else {
                    let mut path = dir2.clone();
                    path.push(format!("{count}_0_pass.ipc"));

                    let file = File::create(path).unwrap();
                    let writer = IpcWriter::new(file).with_compat_level(CompatLevel::newest());
                    let mut writer = writer.batched(&schema).unwrap();

                    for mut df in iter {
                        df.shrink_to_fit();
                        writer.write_batch(&df).unwrap();
                    }
                    writer.finish().unwrap();

                    count += 1;
                }
                total2.store(count, Ordering::Relaxed);
            }
        });

        Ok(Self {
            payload_tx: tx,
            cleanup_tx,
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
        if self.payload_tx.is_full() {
            df.shrink_to_fit();
            let mut path = self.dir.clone();
            let count = self.thread_local_count.fetch_add(1, Ordering::Relaxed);
            // thread local name we start with an underscore to ensure we don't get
            // duplicates
            path.push(format!("_{count}_full.ipc"));

            let file = File::create(path).unwrap();
            let mut writer = IpcWriter::new(file).with_compat_level(CompatLevel::newest());
            writer.finish(&mut df).unwrap();
        } else {
            let iter = Box::new(std::iter::once(df));
            self.dump_iter(None, iter)
        }
    }

    pub(in crate::executors::sinks) fn clean(&self, path: PathBuf) {
        self.cleanup_tx.send(path).unwrap()
    }

    pub(in crate::executors::sinks) fn dump_partition(&self, partition_no: IdxSize, df: DataFrame) {
        let partition = Some(IdxCa::from_vec("", vec![partition_no]));
        let iter = Box::new(std::iter::once(df));
        self.dump_iter(partition, iter)
    }

    pub(in crate::executors::sinks) fn dump_partition_local(
        &self,
        partition_no: IdxSize,
        mut df: DataFrame,
    ) {
        df.shrink_to_fit();
        let count = self.thread_local_count.fetch_add(1, Ordering::Relaxed);
        let mut path = self.dir.clone();
        path.push(format!("{partition_no}"));

        let _ = std::fs::create_dir(&path);
        // thread local name we start with an underscore to ensure we don't get
        // duplicates
        path.push(format!("_{count}.ipc"));
        let file = File::create(path).unwrap();
        let writer = IpcWriter::new(file).with_compat_level(CompatLevel::newest());
        let mut writer = writer.batched(&self.schema).unwrap();
        writer.write_batch(&df).unwrap();
        writer.finish().unwrap();
    }

    pub(in crate::executors::sinks) fn dump_iter(&self, partition: Option<IdxCa>, iter: DfIter) {
        let add = iter.size_hint().1.unwrap();
        self.payload_tx.send((partition, iter)).unwrap();
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
        match File::create(&path) {
            Ok(_) => Ok(Self { path }),
            Err(e) => {
                polars_bail!(ComputeError: "could not create lockfile: {e}")
            },
        }
    }
}

impl Drop for LockFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
