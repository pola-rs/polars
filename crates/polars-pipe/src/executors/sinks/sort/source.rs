use std::iter::Peekable;
use std::path::PathBuf;
use std::time::Instant;

use polars_core::prelude::*;
use polars_core::utils::{accumulate_dataframes_vertical_unchecked, split_df};
use polars_core::POOL;
use rayon::prelude::*;

use crate::executors::sinks::io::IOThread;
use crate::executors::sinks::memory::MemTracker;
use crate::executors::sinks::sort::ooc::{read_df, PartitionSpiller};
use crate::executors::sinks::sort::sink::sort_accumulated;
use crate::executors::sources::get_source_index;
use crate::operators::{DataChunk, PExecutionContext, Source, SourceResult};

pub struct SortSource {
    files: Peekable<std::vec::IntoIter<(u32, PathBuf)>>,
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
    partition_spiller: PartitionSpiller,
    current_part: usize,
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
        partition_spiller: PartitionSpiller,
    ) -> Self {
        if verbose {
            eprintln!("started sort source phase");
        }

        files.sort_unstable_by_key(|entry| entry.0);

        let n_threads = POOL.current_num_threads();
        let files = files.into_iter().peekable();

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
            partition_spiller,
            current_part: 0,
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

    fn finish_from_df(&mut self, df: DataFrame) -> PolarsResult<SourceResult> {
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
                    let out = sort_accumulated(df, self.sort_idx, self.descending, current_slice);
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

        // convert to chunks
        let dfs = split_df(&mut df, self.n_threads)?;
        Ok(SourceResult::GotMoreData(self.finish_batch(dfs)))
    }
    fn print_verbose(&self, verbose: bool) {
        if verbose {
            eprintln!("sort source phase took: {:?}", self.source_start.elapsed());
            eprintln!("full ooc sort took: {:?}", self.ooc_start.elapsed());
        }
    }
}

impl Source for SortSource {
    fn get_batches(&mut self, context: &PExecutionContext) -> PolarsResult<SourceResult> {
        // early return
        if self.finished || self.current_part >= self.partition_spiller.len() {
            self.print_verbose(context.verbose);
            return Ok(SourceResult::Finished);
        }

        let check_in_mem = if let Some((part, _)) = self.files.peek() {
            *part as usize != self.current_part
        } else {
            true
        };

        if check_in_mem {
            let df = self.partition_spiller.get(self.current_part).unwrap();
            self.current_part += 1;
            return self.finish_from_df(df);
        }

        match self.files.next() {
            None => {
                self.print_verbose(context.verbose);
                Ok(SourceResult::Finished)
            },
            Some((partition, mut path)) => {
                let limit = self.memtrack.get_available() / 3;

                let mut read_size = 0;
                let mut read = vec![];

                if let Some(in_mem) = self.partition_spiller.get(partition as usize) {
                    read_size += in_mem.estimated_size();
                    read.push(in_mem)
                }
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
                let out = self.finish_from_df(df);
                self.io_thread.clean(path);
                self.current_part += 1;
                out
            },
        }
    }

    fn fmt(&self) -> &str {
        "sort_source"
    }
}
