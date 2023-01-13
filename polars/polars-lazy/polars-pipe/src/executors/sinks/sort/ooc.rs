use std::fs::DirEntry;
use std::path::PathBuf;

use polars_core::prelude::*;
use polars_core::utils::{_split_offsets, accumulate_dataframes_vertical_unchecked, split_df};
use polars_core::POOL;
use polars_io::ipc::IpcReader;
use polars_io::SerReader;
use polars_ops::prelude::*;
use rayon::prelude::*;

use crate::executors::sinks::sort::io::{block_thread_until_io_thread_done, DfIter, IOThread};
use crate::executors::sinks::sort::sink::sort_accumulated;
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Source, SourceResult};

fn read_df(entry: &DirEntry) -> PolarsResult<DataFrame> {
    let path = entry.path();
    let file = std::fs::File::open(path)?;
    IpcReader::new(file).set_rechunk(false).finish()
}

pub(super) fn sort_ooc(
    io_thread: &IOThread,
    partitions: Series,
    idx: usize,
    reverse: bool,
    slice: Option<(i64, usize)>,
) -> PolarsResult<FinalizedSink> {
    let partitions = partitions.to_physical_repr().into_owned();

    // we collect as I am not sure that if we write to the same directory the
    // iterator will read those also.
    // We don't want to merge files we just written to disk
    let dir = &io_thread.dir;
    let files = std::fs::read_dir(dir)?.collect::<std::io::Result<Vec<_>>>()?;

    let offsets = _split_offsets(files.len(), POOL.current_num_threads());
    POOL.install(|| {
        offsets.par_iter().try_for_each(|(offset, len)| {
            let files = &files[*offset..*offset + *len];

            for entry in files {
                let df = read_df(entry)?;

                let sort_col = &df.get_columns()[idx];
                let assigned_parts = det_partitions(sort_col, &partitions, reverse);

                // partition the dataframe into proper buckets
                let (iter, partition) = partition_df(df, &assigned_parts)?;
                io_thread.dump_iter(Some(partition), iter);
            }
            PolarsResult::Ok(())
        })
    })?;

    block_thread_until_io_thread_done(io_thread);

    let files = std::fs::read_dir(dir)?
        .flat_map(|entry| {
            entry
                .map(|entry| {
                    let path = entry.path();
                    if path.is_dir() {
                        let dirname = path.file_name().unwrap();
                        let partition = dirname.to_string_lossy().parse::<u32>().unwrap();
                        Some((partition, path))
                    } else {
                        None
                    }
                })
                .transpose()
        })
        .collect::<std::io::Result<Vec<_>>>()?;

    let source = SortSource::new(files, idx, reverse, slice);
    Ok(FinalizedSink::Source(Box::new(source)))
}

fn det_partitions(s: &Series, partitions: &Series, reverse: bool) -> IdxCa {
    let s = s.to_physical_repr();

    search_sorted(partitions, &s, SearchSortedSide::Any, reverse).unwrap()
}

fn partition_df(df: DataFrame, partitions: &IdxCa) -> PolarsResult<(DfIter, IdxCa)> {
    let groups = partitions.group_tuples(false, false)?;
    let partitions = unsafe { partitions.clone().into_series().agg_first(&groups) };
    let partitions = partitions.idx().unwrap().clone();

    let out = match groups {
        GroupsProxy::Idx(idx) => {
            let iter = idx.into_iter().map(move |(_, group)| {
                // groups are in bounds
                unsafe { df._take_unchecked_slice(&group, false) }
            });
            Box::new(iter) as DfIter
        }
        GroupsProxy::Slice { groups, .. } => {
            let iter = groups
                .into_iter()
                .map(move |[first, len]| df.slice(first as i64, len as usize));
            Box::new(iter) as DfIter
        }
    };
    Ok((out, partitions))
}

pub struct SortSource {
    files: std::vec::IntoIter<(u32, PathBuf)>,
    n_threads: usize,
    sort_idx: usize,
    reverse: bool,
    chunk_offset: IdxSize,
    slice: Option<(i64, usize)>,
    finished: bool,
}

impl SortSource {
    fn new(
        mut files: Vec<(u32, PathBuf)>,
        sort_idx: usize,
        reverse: bool,
        slice: Option<(i64, usize)>,
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
        }
    }
}

impl Source for SortSource {
    fn get_batches(&mut self, _context: &PExecutionContext) -> PolarsResult<SourceResult> {
        match self.files.next() {
            None => Ok(SourceResult::Finished),
            Some((_, path)) => {
                let files = std::fs::read_dir(path)?.collect::<std::io::Result<Vec<_>>>()?;

                // early return
                if self.finished {
                    return Ok(SourceResult::Finished);
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
                        let out = if *offset as usize > df_len {
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
                // TODO: make utility functions to save these allocations
                let chunk_offset = self.chunk_offset;
                let dfs = split_df(&mut df, self.n_threads)?;
                self.chunk_offset += dfs.len() as IdxSize;
                let batch = dfs
                    .into_iter()
                    .enumerate()
                    .map(|(i, df)| DataChunk {
                        chunk_index: chunk_offset + i as IdxSize,
                        data: df,
                    })
                    .collect();

                Ok(SourceResult::GotMoreData(batch))
            }
        }
    }

    fn fmt(&self) -> &str {
        "sort_source"
    }
}
