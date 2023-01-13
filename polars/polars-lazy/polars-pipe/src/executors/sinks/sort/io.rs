use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use polars_core::prelude::*;
use polars_core::POOL;
use polars_io::prelude::*;
use polars_utils::atomic::SyncCounter;

pub(super) type DfIter = Box<dyn ExactSizeIterator<Item = DataFrame> + Sync + Send>;
// The Option<IdxCa> are the partitions it should be written to, if any
type Payload = (Option<IdxCa>, DfIter);

pub(super) struct IOThread {
    sender: SyncSender<Payload>,
    pub(super) dir: PathBuf,
    pub(super) sent: SyncCounter,
    pub(super) total: SyncCounter,
    pub(super) all_processed: Arc<(Condvar, Mutex<()>)>,
}

impl IOThread {
    pub(super) fn try_new(schema: SchemaRef) -> PolarsResult<Self> {
        let uuid = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = resolve_homedir(Path::new(&format!("~/.polars/sort/{uuid}")));
        std::fs::create_dir_all(&dir)?;

        // we need some pushback otherwise we still could go OOM.
        let (sender, receiver) =
            std::sync::mpsc::sync_channel::<Payload>(POOL.current_num_threads() * 2);

        let sent = SyncCounter::new(0);
        let total = SyncCounter::new(0);

        // we acutally don't really need the mutex
        let all_processed = Arc::new((Condvar::new(), Mutex::new(())));

        let dir2 = dir.clone();
        let total2 = total.clone();
        let all_processed2 = all_processed.clone();
        std::thread::spawn(move || {
            let mut count = 0usize;
            while let Ok((partitions, iter)) = receiver.recv() {
                if let Some(partitions) = partitions {
                    for (part, df) in partitions.into_no_null_iter().zip(iter) {
                        let mut path = dir2.clone();
                        path.push(format!("{part}"));

                        let _ = std::fs::create_dir(&path);
                        path.push(format!("{count}.parquet"));

                        let file = std::fs::File::create(path).unwrap();
                        let mut writer = ParquetWriter::new(file).batched(&schema).unwrap();
                        writer.write_batch(&df).unwrap();
                        writer.finish().unwrap();
                        count += 1;
                    }
                } else {
                    let mut path = dir2.clone();
                    path.push(format!("{count}.parquet"));

                    let file = std::fs::File::create(path).unwrap();
                    let mut writer = ParquetWriter::new(file).batched(&schema).unwrap();

                    for df in iter {
                        writer.write_batch(&df).unwrap();
                    }
                    writer.finish().unwrap();

                    count += 1;
                }
                let total = total2.load(Ordering::Relaxed);
                if total != 0 && total == count {
                    all_processed2.0.notify_one();
                }
            }
        });

        Ok(Self {
            sender,
            dir,
            sent,
            total,
            all_processed,
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
    // set total sent
    io_thread.total.store(sent, Ordering::Relaxed);

    // then the io thread will check if it has written all files, and if it has
    // it will set the condvar so we can continue on this thread

    // we don't really need the mutex for our case, but the condvar needs one
    let cond_lock = io_thread.all_processed.1.lock().unwrap();
    let _lock = io_thread.all_processed.0.wait(cond_lock).unwrap();
}
