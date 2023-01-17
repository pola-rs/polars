use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crossbeam_channel::{bounded, Sender};
use polars_core::prelude::*;
use polars_core::{POOL, PROCESS_ID};
use polars_io::prelude::*;

pub(super) type DfIter = Box<dyn ExactSizeIterator<Item = DataFrame> + Sync + Send>;
// The Option<IdxCa> are the partitions it should be written to, if any
type Payload = (Option<IdxCa>, DfIter);

pub(super) struct IOThread {
    sender: Sender<Payload>,
    pub(super) dir: PathBuf,
    pub(super) sent: Arc<AtomicUsize>,
    pub(super) total: Arc<AtomicUsize>,
}

fn clean_up_thread(current_pid: u128) {
    let _ = std::thread::spawn(move || {
        let dir = resolve_homedir(Path::new("~/.polars/sort/"));

        // if the directory does not exist, there is nothing to clean
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                let dirname = path.file_name().unwrap();
                let dir_uuid = dirname.to_string_lossy().parse::<u128>().unwrap();
                if dir_uuid != current_pid {
                    std::fs::remove_dir_all(path).unwrap()
                }
            }
        }
    });
}

impl IOThread {
    pub(super) fn try_new(schema: SchemaRef) -> PolarsResult<Self> {
        let process_id = *PROCESS_ID;
        let uuid = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        // start a thread that will clean up old dumps.
        // TODO: if we will have more ooc in the future  we will have a dedicated GC thread
        clean_up_thread(process_id);

        let dir = resolve_homedir(Path::new(&format!("~/.polars/sort/{process_id}/{uuid}")));
        std::fs::create_dir_all(&dir)?;

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
        })
    }

    pub(super) fn dump_chunk(&self, df: DataFrame) {
        let iter = Box::new(std::iter::once(df));
        self.dump_iter(None, iter)
    }
    pub(super) fn dump_iter(&self, partition: Option<IdxCa>, iter: DfIter) {
        let add = iter.size_hint().1.unwrap();
        self.sender.send((partition, iter)).unwrap();
        self.sent.fetch_add(add, Ordering::Relaxed);
    }
}

pub(super) fn block_thread_until_io_thread_done(io_thread: &IOThread) {
    // get number sent
    let sent = io_thread.sent.load(Ordering::Relaxed);
    // get number processed
    while io_thread.total.load(Ordering::Relaxed) != sent {
        std::thread::park_timeout(Duration::from_millis(6))
    }
}
