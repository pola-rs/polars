use std::fs::DirEntry;

use polars_core::prelude::*;
use polars_core::utils::_split_offsets;
use polars_core::POOL;
use polars_io::ipc::IpcReader;
use polars_io::SerReader;
use polars_ops::prelude::*;
use rayon::prelude::*;

use crate::executors::sinks::sort::io::{block_thread_until_io_thread_done, DfIter, IOThread};
use crate::executors::sinks::sort::source::SortSource;
use crate::operators::FinalizedSink;

pub(super) fn read_df(entry: &DirEntry) -> PolarsResult<DataFrame> {
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
                let (iter, unique_assigned_parts) = partition_df(df, &assigned_parts)?;
                io_thread.dump_iter(Some(unique_assigned_parts), iter);
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

    let source = SortSource::new(files, idx, reverse, slice, partitions);
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
