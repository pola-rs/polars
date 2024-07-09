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
    nulls_last: bool,
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
        nulls_last: bool,
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
            nulls_last,
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
            None => sort_accumulated(
                df,
                self.sort_idx,
                None,
                SortOptions {
                    descending: self.descending,
                    nulls_last: self.nulls_last,
                    multithreaded: true,
                    maintain_order: false,
                },
            ),
            Some((offset, len)) => {
                let df_len = df.height();
                debug_assert!(*offset >= 0);
                let out = if *offset as usize >= df_len {
                    *offset -= df_len as i64;
                    Ok(df.slice(0, 0))
                } else {
                    let out = sort_accumulated(
                        df,
                        self.sort_idx,
                        current_slice,
                        SortOptions {
                            descending: self.descending,
                            nulls_last: self.nulls_last,
                            multithreaded: true,
                            maintain_order: false,
                        },
                    );
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
        let dfs = split_df(&mut df, self.n_threads, true);
        Ok(SourceResult::GotMoreData(self.finish_batch(dfs)))
    }
    fn print_verbose(&self, verbose: bool) {
        if verbose {
            eprintln!("sort source phase took: {:?}", self.source_start.elapsed());
            eprintln!("full ooc sort took: {:?}", self.ooc_start.elapsed());
        }
    }

    fn get_from_memory(
        &mut self,
        read: &mut Vec<DataFrame>,
        read_size: &mut usize,
        part: usize,
        keep_track: bool,
    ) {
        while self.current_part <= part {
            if let Some(df) = self.partition_spiller.get(self.current_part - 1) {
                if keep_track {
                    *read_size += df.estimated_size();
                }
                read.push(df);
            }
            self.current_part += 1;
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
        self.current_part += 1;
        let mut read_size = 0;
        let mut read = vec![];

        match self.files.next() {
            None => {
                // Ensure we fetch all from memory.
                self.get_from_memory(
                    &mut read,
                    &mut read_size,
                    self.partition_spiller.len(),
                    false,
                );
                if read.is_empty() {
                    self.print_verbose(context.verbose);
                    Ok(SourceResult::Finished)
                } else {
                    self.finished = true;
                    let df = accumulate_dataframes_vertical_unchecked(read);
                    self.finish_from_df(df)
                }
            },
            Some((mut partition, mut path)) => {
                self.get_from_memory(&mut read, &mut read_size, partition as usize, true);
                let limit = self.memtrack.get_available() / 3;

                loop {
                    if let Some(in_mem) = self.partition_spiller.get(partition as usize) {
                        read_size += in_mem.estimated_size();
                        read.push(in_mem)
                    }

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

                    let Some((next_part, next_path)) = self.files.next() else {
                        break;
                    };
                    path = next_path;
                    partition = next_part;
                }
                let df = accumulate_dataframes_vertical_unchecked(read);
                let out = self.finish_from_df(df);
                self.io_thread.clean(path);
                out
            },
        }
    }

    fn fmt(&self) -> &str {
        "sort_source"
    }
}
