use std::fs::DirEntry;
use std::path::PathBuf;

use polars_core::prelude::*;
use polars_core::utils::{accumulate_dataframes_vertical_unchecked, split_df};
use polars_core::POOL;
use rayon::prelude::*;

use crate::executors::sinks::sort::ooc::read_df;
use crate::executors::sinks::sort::sink::sort_accumulated;
use crate::operators::{DataChunk, PExecutionContext, Source, SourceResult};

pub struct SortSource {
    files: std::vec::IntoIter<(u32, PathBuf)>,
    n_threads: usize,
    sort_idx: usize,
    reverse: bool,
    chunk_offset: IdxSize,
    slice: Option<(i64, usize)>,
    finished: bool,

    // The sorted partitions
    // are used check if a directory is already completely sorted
    // if the lower boundary of a partition is equal to the upper
    // boundary, the whole dictionary is already sorted
    // this dictionary may also be very large as in the extreme case
    // we sort a column with a constant value, then the binary search
    // ensures that all files will be written to a single folder
    // in that case we just read the files
    partitions: Series,
    sorted_directory_in_process: Option<std::vec::IntoIter<DirEntry>>,
}

impl SortSource {
    pub(super) fn new(
        mut files: Vec<(u32, PathBuf)>,
        sort_idx: usize,
        reverse: bool,
        slice: Option<(i64, usize)>,
        partitions: Series,
    ) -> Self {
        files.sort_unstable_by_key(|entry| entry.0);

        let n_threads = POOL.current_num_threads();
        let files = files.into_iter();

        Self {
            files,
            n_threads,
            sort_idx,
            reverse,
            chunk_offset: 0,
            slice,
            finished: false,
            partitions,
            sorted_directory_in_process: None,
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
    fn get_batches(&mut self, _context: &PExecutionContext) -> PolarsResult<SourceResult> {
        // early return
        if self.finished {
            return Ok(SourceResult::Finished);
        }

        // this branch processes the directories containing a single sort key
        // e.g. the lower_bound == upper_bound
        if let Some(files) = &mut self.sorted_directory_in_process {
            let read = files
                .take(self.n_threads)
                .map(|entry| read_df(&entry))
                .collect::<PolarsResult<Vec<DataFrame>>>()?;
            let mut df = match (read.len(), &mut self.slice) {
                (0, _) => {
                    // depleted directory, continue with normal sorting
                    self.sorted_directory_in_process = None;
                    return self.get_batches(_context);
                }
                // there is not slice and we got exactly enough files
                // so we return, happy path
                (n, None) if n == self.n_threads => {
                    return Ok(SourceResult::GotMoreData(self.finish_batch(read)))
                }
                // there is a slice, so we concat and apply the slice
                // and then later split over the number of threads
                (_, Some((offset, len))) => {
                    let df = accumulate_dataframes_vertical_unchecked(read);
                    let df_len = df.height();

                    // whole batch can be skipped
                    let out = if *offset as usize >= df_len {
                        *offset -= df_len as i64;
                        return self.get_batches(_context);
                    } else {
                        let out = df.slice(*offset, *len);
                        *len = len.saturating_sub(df_len);
                        *offset = 0;
                        out
                    };
                    if *len == 0 {
                        self.finished = true;
                    }
                    out
                }
                // The number of files read are lower than the number of
                // batches we have to return, so we first accumulate
                // and then split over the number of threads
                (_, None) => accumulate_dataframes_vertical_unchecked(read),
            };
            let batch = split_df(&mut df, self.n_threads)?;
            return Ok(SourceResult::GotMoreData(self.finish_batch(batch)));
        }

        match self.files.next() {
            None => Ok(SourceResult::Finished),
            Some((partition, path)) => {
                let files = std::fs::read_dir(path)?.collect::<std::io::Result<Vec<_>>>()?;

                // both lower and upper can fail.
                // lower can fail because the search_sorted can add the sort idx at the end of the array, which is i == len
                if let (Ok(lower), Ok(upper)) = (
                    self.partitions.get(partition as usize),
                    self.partitions.get(partition as usize + 1),
                ) {
                    if lower == upper && !files.is_empty() {
                        let files = files.into_iter();
                        self.sorted_directory_in_process = Some(files);
                        return self.get_batches(_context);
                    }
                }

                // read the files in a single partition in parallel
                let dfs = POOL.install(|| {
                    files
                        .par_iter()
                        .map(read_df)
                        .collect::<PolarsResult<Vec<DataFrame>>>()
                })?;
                let df = accumulate_dataframes_vertical_unchecked(dfs);
                // sort a single partition
                let current_slice = self.slice;
                let mut df = match &mut self.slice {
                    None => sort_accumulated(df, self.sort_idx, self.reverse, None),
                    Some((offset, len)) => {
                        let df_len = df.height();
                        assert!(*offset >= 0);
                        let out = if *offset as usize >= df_len {
                            *offset -= df_len as i64;
                            Ok(df.slice(0, 0))
                        } else {
                            let out =
                                sort_accumulated(df, self.sort_idx, self.reverse, current_slice);
                            *len = len.saturating_sub(df_len);
                            *offset = 0;
                            out
                        };
                        if *len == 0 {
                            self.finished = true;
                        }
                        out
                    }
                }?;

                // convert to chunks
                let dfs = split_df(&mut df, self.n_threads)?;
                Ok(SourceResult::GotMoreData(self.finish_batch(dfs)))
            }
        }
    }

    fn fmt(&self) -> &str {
        "sort_source"
    }
}
