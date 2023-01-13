use std::collections::VecDeque;
use std::fs::DirEntry;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use polars_core::POOL;

use polars_core::prelude::*;
use polars_core::utils::{accumulate_dataframes_vertical_unchecked, split_df};
use polars_io::parquet::ParquetReader;
use polars_io::prelude::BatchedParquetReader;
use polars_io::SerReader;
use polars_ops::prelude::*;
use polars_plan::prelude::Context::Default;

use crate::executors::sinks::sort::io::{DfIter, IOThread};
use crate::CHUNK_SIZE;
use crate::operators::{DataChunk, FinalizedSink, Operator, OperatorResult, PExecutionContext, Source, SourceResult};

fn read_df(entry: DirEntry) -> PolarsResult<DataFrame> {
    let path = entry.path();
    let file = std::fs::File::open(&path)?;
    ParquetReader::new(file).set_rechunk(false).finish()
}

pub(super) type ChunkFallibleIter = Box<dyn Iterator<Item=PolarsResult<DataChunk>> + Send + Sync>;

pub(super) fn sort_ooc(io_thread: &IOThread, partitions: Series, idx: usize, reverse: bool) -> PolarsResult<FinalizedSink> {

    let partitions = partitions.to_physical_repr().into_owned();

    // let dir = &write_thread.dir;
    // we collect as I am not sure that if we write to the same directory the
    // iterator will read those also.
    // We don't want to merge files we just written to disk
    let dir = &io_thread.dir;
    let files = std::fs::read_dir(dir)?.collect::<std::io::Result<Vec<_>>>()?;
    const BATCH_SIZE: usize = 16;

    for entry in files {
        let df= read_df(entry)?;

        let sort_col = &df.get_columns()[idx];
        let mut assigned_parts = det_partitions(sort_col, &partitions);

        // partition the dataframe into proper buckets
        let (iter, partition) = partition_df(df, &assigned_parts)?;
        io_thread.dump_iter(Some(partition), iter);
    }

    let all_processed = io_thread.all_processed.clone();
    // get number sent
    let sent = io_thread.sent.load(Ordering::Acquire);
    // set total sent
    io_thread.total.store(sent, Ordering::Release);

    // then the io thread will check if it has written all files, and if it has
    // it will set the condvar so we can continue on this thread

    // we don't really need the mutex for our case, but the condvar needs one
    let cond_lock = io_thread.all_processed.1.lock().unwrap();
    all_processed.0.wait(cond_lock).unwrap();

    let mut files = std::fs::read_dir(dir)?.flat_map(|entry| {
        entry.map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                let dirname = path.file_name().unwrap();
                let partition = dirname.to_string_lossy().parse::<u32>().unwrap();
                Some((partition, path))
            } else {
                None
            }
        }).transpose()
    }).collect::<std::io::Result<Vec<_>>>()?;

    let source = SortSource::new(files, idx, reverse) ;
    Ok(FinalizedSink::Source(Box::new(source)))
}

fn det_partitions(s: &Series, partitions: &Series) -> IdxCa {
    let s = s.to_physical_repr();

    search_sorted(partitions, &s, SearchSortedSide::Any).unwrap()

}

fn partition_df(df: DataFrame, partitions: &IdxCa) -> PolarsResult<(DfIter, IdxCa)> {
    let groups = partitions.group_tuples(false, false)?;
    let partitions = unsafe { partitions.clone().into_series().agg_first(&groups) };
    let partitions = partitions.idx().unwrap().clone();

    let out = match groups {
        GroupsProxy::Idx(idx) => {
                let iter = idx
                .into_iter()
                .map(move |(_, group)| {
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
        },
    };
    Ok((out, partitions))
}

pub struct SortSource {
    files: std::vec::IntoIter<(u32, PathBuf)>,
    n_threads: usize,
    sort_idx: usize,
    reverse: bool,
    chunk_offset: IdxSize
}

impl SortSource {
    fn new(mut files: Vec<(u32, PathBuf)>, sort_idx: usize, reverse: bool) -> Self {
        files.sort_unstable_by_key(|entry| {
            entry.0
        });

        let n_threads = POOL.current_num_threads();
        let files = files.into_iter();

        Self {
            files,
            n_threads,
            sort_idx,
            reverse,
            chunk_offset: 0
        }
    }
}

impl Source for SortSource {
    fn get_batches(&mut self, context: &PExecutionContext) -> PolarsResult<SourceResult> {
        match self.files.next() {
            None => Ok(SourceResult::Finished),
            Some((_, path)) => {
                let files = std::fs::read_dir(path)?.collect::<std::io::Result<Vec<_>>>()?;
                let dfs = files.into_iter().map(|entry| {
                    dbg!(entry.path());
                    let df = read_df(entry)?;

                    dbg!(df.get_columns()[0].max::<usize>());
                    Ok(df)


                }).collect::<PolarsResult<Vec<DataFrame>>>()?;
                let df = accumulate_dataframes_vertical_unchecked(dfs);
                let sort_column = df.get_columns()[self.sort_idx].clone();

                let mut df = df.sort_impl(vec![sort_column], vec![self.reverse], false, None, true)?;

                let mut chunk_offset = self.chunk_offset;
                let dfs = split_df(&mut df, self.n_threads)?;
                self.chunk_offset += dfs.len() as IdxSize;
                let batch = dfs.into_iter().enumerate().map(|(i, df)| {
                    DataChunk {
                        chunk_index: chunk_offset + i as IdxSize,
                        data: df
                    }
                }).collect();

                Ok(SourceResult::GotMoreData(batch))
            }
        }
    }

    fn fmt(&self) -> &str {
        "sort_source"
    }
}