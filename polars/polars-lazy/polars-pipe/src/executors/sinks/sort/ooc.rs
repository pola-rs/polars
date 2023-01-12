use std::path::Path;
use std::sync::atomic::Ordering;

use polars_core::prelude::*;
use polars_io::parquet::ParquetReader;
use polars_io::prelude::BatchedParquetReader;
use polars_io::SerReader;
use polars_ops::prelude::*;

use crate::executors::sinks::sort::io::{DfIter, IOThread};
use crate::CHUNK_SIZE;
use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

pub(super) fn sort_ooc(io_thread: &IOThread, partitions: Series, idx: usize, schema: &SchemaRef) -> PolarsResult<DfIter> {
    dbg!("start ooc sort");

    let partitions = partitions.to_physical_repr().into_owned();

    dbg!(&partitions);

    // let dir = &write_thread.dir;
    // we collect as I am not sure that if we write to the same directory the
    // iterator will read those also.
    // We don't want to merge files we just written to disk
    let dir = &io_thread.dir;
    let files = std::fs::read_dir(dir)?.collect::<std::io::Result<Vec<_>>>()?;
    const BATCH_SIZE: usize = 16;

    for entry in files {
        let path = entry.path();
        let file = std::fs::File::open(&path)?;
        let df= ParquetReader::new(file).set_rechunk(false).finish()?;

        let sort_col = &df.get_columns()[idx];
        let mut assigned_parts = det_partitions(sort_col, &partitions);

        // partition the dataframe into proper buckets
        let (iter, partition) = partition_df(df, &assigned_parts)?;
        io_thread.dump_iter(Some(partition), iter);

        // clean up files that are processed
        std::fs::remove_file(path).unwrap()
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

    let mut files = std::fs::read_dir(dir)?.map(|entry| {
        entry.map(|entry| {
            let path = entry.path();
            let dirname = path.file_name().unwrap();
            let partition = dirname.to_string_lossy().parse::<u32>().unwrap();
            (partition, path)
        })
    }).collect::<std::io::Result<Vec<_>>>()?;
    files.sort_unstable_by_key(|entry| {
        entry.0
    });
    let iter = files.into_iter().map(|(_part, path)| {
        let file = std::fs::File::open(&path)?;
        ParquetReader::new(path).set_rechunk(false).finish().unwrap()
    });

    Ok(Box::new(iter) as DfIter)
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

pub struct SortOperator {
    iter: DfIter,
    count: usize,
    total: usize,
}

impl Operator for SortOperator {
    fn execute(&mut self, context: &PExecutionContext, chunk: &DataChunk) -> PolarsResult<OperatorResult> {
        self.count += 1;
        let data = self.iter.next().unwrap();
        let chunk = DataChunk{
            data,
            chunk_index: self.count
        };
        if self.count == self.total {
            Ok(OperatorResult::Finished(chunk))

        }
        todo!()
    }

    fn split(&self, thread_no: usize) -> Box<dyn Operator> {
        todo!()
    }

    fn fmt(&self) -> &str {
        "ooc_sort_operator"
    }
}