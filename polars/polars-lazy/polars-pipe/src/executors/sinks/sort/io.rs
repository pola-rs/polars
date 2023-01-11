use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

use polars_core::prelude::*;
use polars_io::prelude::*;

pub(super) struct IOThread {
    sender: Sender<DataFrame>,
    pub(super) dir: PathBuf,
}

impl IOThread {
    pub(super) fn try_new() -> PolarsResult<Self> {
        let uuid = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = resolve_homedir(Path::new(&format!("~/.polars/sort/{uuid}")));
        std::fs::create_dir_all(&dir)?;

        let (sender, receiver) = std::sync::mpsc::channel::<DataFrame>();

        let dir2 = dir.clone();
        std::thread::spawn(move || {
            let mut count = 0usize;
            while let Ok(mut df) = receiver.recv() {
                let mut path = dir2.clone();
                path.push(format!("{count}.ipc"));

                let file = std::fs::File::create(path).unwrap();
                IpcWriter::new(file).finish(&mut df).unwrap();

                count += 1;
            }
            eprintln!("kill thread");
        });

        Ok(Self { sender, dir })
    }

    pub(super) fn dump_chunk(&self, df: DataFrame) {
        self.sender.send(df).unwrap()
    }
}