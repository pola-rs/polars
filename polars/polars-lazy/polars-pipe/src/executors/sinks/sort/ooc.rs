use std::path::Path;
use polars_core::frame::DataFrame;
use polars_core::prelude::PolarsResult;
use polars_io::prelude::IpcReader;
use polars_io::SerReader;
use crate::executors::sinks::sort::io::IOThread;

pub(super) fn sort_ooc(write_thread: &IOThread) -> PolarsResult<DataFrame> {
    dbg!("start ooc sort");
    let dir = &write_thread.dir;
    loop {
        // we collect as I am not sure that if we write to the same directory the
        // iterator will read those also.
        // We don't want to merge files we just written to disk
        let files = std::fs::read_dir(dir)?.collect::<std::io::Result<Vec<_>>>()?;

        files.chunks_exact(2).try_for_each(|pair| {
            let table_a_entry = &pair[0];
            let table_b_entry = &pair[1];

            let a = table_a_entry.path();
            let b = table_b_entry.path();

            let a= std::fs::File::open(a)?;
            let df_a = IpcReader::new(a).finish()?;

            let b= std::fs::File::open(b)?;
            let df_b = IpcReader::new(b).finish()?;

            dbg!(df_a, df_b);

            PolarsResult::Ok(())
        })?;
    }


    todo!()
}