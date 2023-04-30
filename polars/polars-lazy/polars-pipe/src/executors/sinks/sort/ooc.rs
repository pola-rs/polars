use std::path::Path;

use polars_core::prelude::*;
use polars_core::utils::_split_offsets;
use polars_core::POOL;
use polars_io::ipc::IpcReader;
use polars_io::SerReader;
use polars_ops::prelude::*;
use rayon::prelude::*;

use crate::executors::sinks::io::{DfIter, IOThread};
use crate::executors::sinks::sort::source::SortSource;
use crate::operators::FinalizedSink;

pub(super) fn read_df(path: &Path) -> PolarsResult<DataFrame> {
    let file = std::fs::File::open(path)?;
    IpcReader::new(file).set_rechunk(false).finish()
}

pub(super) fn sort_ooc(
    io_thread: &IOThread,
    partitions: Series,
    idx: usize,
    descending: bool,
    slice: Option<(i64, usize)>,
    verbose: bool,
) -> PolarsResult<FinalizedSink> {
    let partitions = partitions.to_physical_repr().into_owned();

    // we collect as I am not sure that if we write to the same directory the
    // iterator will read those also.
    // We don't want to merge files we just written to disk
    let dir = &io_thread.dir;
    let files = std::fs::read_dir(dir)?.collect::<std::io::Result<Vec<_>>>()?;

    if verbose {
        eprintln!("processing {} files", files.len());
    }

    // here it will split every file into `N` partitions.
    // So this will create approximately M * N files of size M / N
    // this heavily influences performance
    // TODO!
    // check if we can batch the output files per partition before we write them.
    // this way we can write large files and amortize IO cost
    let offsets = _split_offsets(files.len(), POOL.current_num_threads() * 2);
    POOL.install(|| {
        offsets.par_iter().try_for_each(|(offset, len)| {
            let files = &files[*offset..*offset + *len];

            for entry in files {
                let path = entry.path();

                // don't read the lock file
                if path.ends_with(".lock") {
                    continue;
                }
                let df = read_df(&path)?;

                let sort_col = &df.get_columns()[idx];
                let assigned_parts = det_partitions(sort_col, &partitions, descending);

                // partition the dataframe into proper buckets
                let (iter, unique_assigned_parts) = partition_df(df, &assigned_parts)?;
                io_thread.dump_partitioned_thread_local(unique_assigned_parts, iter);
            }
            PolarsResult::Ok(())
        })
    })?;
    if verbose {
        eprintln!("finished partitioning sort files");
    }

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

    let source = SortSource::new(files, idx, descending, slice, verbose);
    Ok(FinalizedSink::Source(Box::new(source)))
}

fn det_partitions(s: &Series, partitions: &Series, descending: bool) -> IdxCa {
    let s = s.to_physical_repr();

    search_sorted(partitions, &s, SearchSortedSide::Any, descending).unwrap()
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
