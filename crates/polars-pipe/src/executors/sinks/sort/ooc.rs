use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crossbeam_queue::SegQueue;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::{
    accumulate_dataframes_vertical_unchecked, accumulate_dataframes_vertical_unchecked_optional,
};
use polars_core::POOL;
use polars_io::ipc::IpcReader;
use polars_io::SerReader;
use polars_ops::prelude::*;
use rayon::prelude::*;

use crate::executors::sinks::io::{DfIter, IOThread};
use crate::executors::sinks::sort::source::SortSource;
use crate::operators::FinalizedSink;

pub(super) fn read_df(path: &Path) -> PolarsResult<DataFrame> {
    let file = polars_utils::open_file(path)?;
    IpcReader::new(file).set_rechunk(false).finish()
}

// Utility to buffer partitioned dataframes
// this ensures we don't write really small dataframes
// and amortize IO cost
#[derive(Default)]
struct PartitionSpillBuf {
    row_count: AtomicU32,
    // keep track of the length
    // that's cheaper than iterating the linked list
    len: AtomicU32,
    size: AtomicU64,
    chunks: SegQueue<DataFrame>,
}

impl PartitionSpillBuf {
    fn push(&self, df: DataFrame) -> Option<DataFrame> {
        debug_assert!(df.height() > 0);
        let acc = self
            .row_count
            .fetch_add(df.height() as u32, Ordering::Relaxed);
        let size = self
            .size
            .fetch_add(df.estimated_size() as u64, Ordering::Relaxed);
        let larger_than_32_mb = size > 1 << 25;
        let len = self.len.fetch_add(1, Ordering::Relaxed);
        self.chunks.push(df);
        if acc > 50_000 || larger_than_32_mb {
            // reset all statistics
            self.row_count.store(0, Ordering::Relaxed);
            self.len.store(0, Ordering::Relaxed);
            self.size.store(0, Ordering::Relaxed);
            // other threads can be pushing while we drain
            // so we pop no more than the current size.
            let pop_max = len;
            let iter = (0..pop_max).flat_map(|_| self.chunks.pop());
            // Due to race conditions, the chunks can already be popped, so we use optional.
            accumulate_dataframes_vertical_unchecked_optional(iter)
        } else {
            None
        }
    }

    fn finish(self) -> Option<DataFrame> {
        if !self.chunks.is_empty() {
            let iter = self.chunks.into_iter();
            Some(accumulate_dataframes_vertical_unchecked(iter))
        } else {
            None
        }
    }
}

struct PartitionSpiller {
    partitions: Vec<PartitionSpillBuf>,
}

impl PartitionSpiller {
    fn new(n_parts: usize) -> Self {
        let mut partitions = vec![];
        partitions.resize_with(n_parts + 1, PartitionSpillBuf::default);
        Self { partitions }
    }

    fn push(&self, partition: usize, df: DataFrame) -> Option<DataFrame> {
        self.partitions[partition].push(df)
    }

    fn spill_all(self, io_thread: &IOThread) {
        let min_len = std::cmp::max(self.partitions.len() / POOL.current_num_threads(), 2);
        POOL.install(|| {
            self.partitions
                .into_par_iter()
                .with_min_len(min_len)
                .enumerate()
                .for_each(|(part, part_buf)| {
                    if let Some(df) = part_buf.finish() {
                        io_thread.dump_partition_local(part as IdxSize, df)
                    }
                })
        })
    }
}

pub(super) fn sort_ooc(
    io_thread: &IOThread,
    // these partitions are the samples
    // these are not yet assigned to a buckets
    samples: Series,
    idx: usize,
    descending: bool,
    slice: Option<(i64, usize)>,
    verbose: bool,
) -> PolarsResult<FinalizedSink> {
    let samples = samples.to_physical_repr().into_owned();

    // we collect as I am not sure that if we write to the same directory the
    // iterator will read those also.
    // We don't want to merge files we just written to disk
    let dir = &io_thread.dir;
    let files = std::fs::read_dir(dir)?.collect::<std::io::Result<Vec<_>>>()?;

    if verbose {
        eprintln!("processing {} files", files.len());
    }

    let partitions_spiller = PartitionSpiller::new(samples.len());

    POOL.install(|| {
        files.par_iter().try_for_each(|entry| {
            let path = entry.path();
            // don't read the lock file
            if path.ends_with(".lock") {
                return PolarsResult::Ok(());
            }
            let df = read_df(&path)?;

            let sort_col = &df.get_columns()[idx];
            let assigned_parts = det_partitions(sort_col, &samples, descending);

            // partition the dataframe into proper buckets
            let (iter, unique_assigned_parts) = partition_df(df, &assigned_parts)?;
            for (part, df) in unique_assigned_parts.into_no_null_iter().zip(iter) {
                if let Some(df) = partitions_spiller.push(part as usize, df) {
                    io_thread.dump_partition_local(part, df)
                }
            }
            PolarsResult::Ok(())
        })
    })?;
    partitions_spiller.spill_all(io_thread);
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
                // groups are in bounds and sorted
                unsafe { df._take_unchecked_slice_sorted(&group, false, IsSorted::Ascending) }
            });
            Box::new(iter) as DfIter
        },
        GroupsProxy::Slice { groups, .. } => {
            let iter = groups
                .into_iter()
                .map(move |[first, len]| df.slice(first as i64, len as usize));
            Box::new(iter) as DfIter
        },
    };
    Ok((out, partitions))
}
