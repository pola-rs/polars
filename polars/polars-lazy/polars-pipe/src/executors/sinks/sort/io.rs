use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Sender, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

use polars_core::prelude::*;
use polars_core::POOL;
use polars_io::prelude::*;
use polars_utils::atomic::SyncCounter;

type Payload = (Schema, Box<dyn Iterator<Item = DataFrame> + Send + Sync>);

pub(super) struct IOThread {
    sender: SyncSender<Payload>,
    pub(super) dir: PathBuf,
    pub(super) send: SyncCounter,
    pub(super) all_processed: Arc<Condvar>
}

impl IOThread {
    pub(super) fn try_new() -> PolarsResult<Self> {
        let uuid = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = resolve_homedir(Path::new(&format!("~/.polars/sort/{uuid}")));
        std::fs::create_dir_all(&dir)?;

        // we need some pushback otherwise we still could go OOM.
        let (sender, receiver) =
            std::sync::mpsc::sync_channel::<Payload>(POOL.current_num_threads() * 2);

        let send = SyncCounter::new(0);

        let dir2 = dir.clone();
        let send2 = send.clone();
        std::thread::spawn(move || {
            let mut count = 0usize;
            while let Ok((schema, iter)) = receiver.recv() {
                let mut path = dir2.clone();
                path.push(format!("{count}.parquet"));

                let file = std::fs::File::create(path).unwrap();
                let mut writer = ParquetWriter::new(file).batched(&schema).unwrap();

                for df in iter {
                    writer.write_batch(&df).unwrap();
                }
                writer.finish().unwrap();

                let previous_send = send2.fetch_add(Ordering::Relaxed);

                // read previous_count as we have not updated it yet
                if previous_send == count {

                }
                count += 1;
            }
            eprintln!("kill thread");
        });

        Ok(Self {
            sender,
            dir,
            send
        })
    }

    pub(super) fn dump_chunk(&self, df: DataFrame) {
        let schema = df.schema();
        self.sender
            .send((schema, Box::new(std::iter::once(df))))
            .unwrap();
        self.send.fetch_add(1, Ordering::Relaxed);
    }
}
