use std::path::PathBuf;
use std::time::Instant;

use polars_core::prelude::*;
use polars_core::utils::{accumulate_dataframes_vertical_unchecked, split_df};
use polars_core::POOL;
use rayon::prelude::*;

use crate::executors::sinks::io::IOThread;
use crate::executors::sinks::memory::MemTracker;
use crate::executors::sinks::sort::ooc::read_df;
use crate::executors::sinks::sort::sink::sort_accumulated;
use crate::executors::sources::get_source_index;
use crate::operators::{DataChunk, PExecutionContext, Source, SourceResult};

pub struct SortSource {
    files: std::vec::IntoIter<(u32, PathBuf)>,
    n_threads: usize,
    sort_idx: usize,
    descending: bool,
    chunk_offset: IdxSize,
    slice: Option<(i64, usize)>,
    finished: bool,
    io_thread: IOThread,
    memtrack: MemTracker,
    // Start of the Source phase
    source_start: Instant,
    // Start of the OOC sort operation.
    ooc_start: Instant,
}

impl SortSource {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        mut files: Vec<(u32, PathBuf)>,
        sort_idx: usize,
        descending: bool,
        slice: Option<(i64, usize)>,
        verbose: bool,
        io_thread: IOThread,
        memtrack: MemTracker,
        ooc_start: Instant,
    ) -> Self {
        if verbose {
            eprintln!("started sort source phase");
        }

        files.sort_unstable_by_key(|entry| entry.0);

        let n_threads = POOL.current_num_threads();
        let files = files.into_iter();

        Self {
            files,
            n_threads,
            sort_idx,
            descending,
            chunk_offset: get_source_index(1) as IdxSize,
            slice,
            finished: false,
            io_thread,
            memtrack,
            source_start: Instant::now(),
            ooc_start,
        }
    }
    fn finish_batch(&mut self, dfs: Vec<DataFrame>) -> Vec<DataChunk> {
        // TODO: make utility functions to save these allocations
        let chunk_offset = self.chunk_offset;
        self.chunk_offset += dfs.len() as IdxSize;
        dfs.into_iter()
            .enumerate()
            .map(|(i, df)| DataChunk {
                chunk_index: chunk_offset + i as IdxSize,
                data: df,
            })
            .collect()
    }
}

impl Source for SortSource {
    fn get_batches(&mut self, context: &PExecutionContext) -> PolarsResult<SourceResult> {
        // early return
        if self.finished {
            return Ok(SourceResult::Finished);
        }

        match self.files.next() {
            None => {
                if context.verbose {
                    eprintln!("sort source phase took: {:?}", self.source_start.elapsed());
                    eprintln!("full ooc sort took: {:?}", self.ooc_start.elapsed());
                }
                Ok(SourceResult::Finished)
            },
            Some((_, mut path)) => {
                let limit = self.memtrack.get_available() / 3;

                let mut read_size = 0;
                let mut read = vec![];
                loop {
                    let files = std::fs::read_dir(&path)?.collect::<std::io::Result<Vec<_>>>()?;

                    // read the files in a single partition in parallel
                    let dfs = POOL.install(|| {
                        files
                            .par_iter()
                            .map(|entry| {
                                let df = read_df(&entry.path())?;
                                Ok(df)
                            })
                            .collect::<PolarsResult<Vec<DataFrame>>>()
                    })?;
                    let df = accumulate_dataframes_vertical_unchecked(dfs);
                    read_size += df.estimated_size();
                    read.push(df);
                    if read_size > limit {
                        break;
                    }

                    let Some((_, next_path)) = self.files.next() else {
                        break;
                    };
                    path = next_path;
                }
                let df = accumulate_dataframes_vertical_unchecked(read);

                // Sort a single partition
                // We always need to sort again!
                let current_slice = self.slice;

                let mut df = match &mut self.slice {
                    None => sort_accumulated(df, self.sort_idx, self.descending, None),
                    Some((offset, len)) => {
                        let df_len = df.height();
                        assert!(*offset >= 0);
                        let out = if *offset as usize >= df_len {
                            *offset -= df_len as i64;
                            Ok(df.slice(0, 0))
                        } else {
                            let out =
                                sort_accumulated(df, self.sort_idx, self.descending, current_slice);
                            *len = len.saturating_sub(df_len);
                            *offset = 0;
                            out
                        };
                        if *len == 0 {
                            self.finished = true;
                        }
                        out
                    },
                }?;
                self.io_thread.clean(path);

                // convert to chunks
                let dfs = split_df(&mut df, self.n_threads)?;
                Ok(SourceResult::GotMoreData(self.finish_batch(dfs)))
            },
        }
    }

    fn fmt(&self) -> &str {
        "sort_source"
    }
}
