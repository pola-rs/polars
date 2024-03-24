use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

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
use crate::executors::sinks::memory::MemTracker;
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
    // keep track of the length
    // that's cheaper than iterating the linked list
    len: AtomicU32,
    size: AtomicU64,
    chunks: SegQueue<DataFrame>,
}

impl PartitionSpillBuf {
    fn push(&self, df: DataFrame, spill_limit: u64) -> Option<DataFrame> {
        debug_assert!(df.height() > 0);
        let size = self
            .size
            .fetch_add(df.estimated_size() as u64, Ordering::Relaxed);
        let len = self.len.fetch_add(1, Ordering::Relaxed);
        self.chunks.push(df);
        if size > spill_limit {
            // Reset all statistics.
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

    fn finish(&self) -> Option<DataFrame> {
        if !self.chunks.is_empty() {
            let len = self.len.load(Ordering::Relaxed) + 1;
            let mut out = Vec::with_capacity(len as usize);
            while let Some(df) = self.chunks.pop() {
                out.push(df)
            }
            Some(accumulate_dataframes_vertical_unchecked(out))
        } else {
            None
        }
    }
}

pub(crate) struct PartitionSpiller {
    partitions: Vec<PartitionSpillBuf>,
    // Spill limit in bytes.
    spill_limit: u64,
}

impl PartitionSpiller {
    fn new(n_parts: usize, spill_limit: u64) -> Self {
        let mut partitions = vec![];
        partitions.resize_with(n_parts + 1, PartitionSpillBuf::default);
        Self {
            partitions,
            spill_limit,
        }
    }

    fn push(&self, partition: usize, df: DataFrame) -> Option<DataFrame> {
        self.partitions[partition].push(df, self.spill_limit)
    }

    pub(crate) fn get(&self, partition: usize) -> Option<DataFrame> {
        self.partitions[partition].finish()
    }

    pub(crate) fn len(&self) -> usize {
        self.partitions.len()
    }

    #[cfg(debug_assertions)]
    // Used in testing only.
    fn spill_all(&self, io_thread: &IOThread) {
        let min_len = std::cmp::max(self.partitions.len() / POOL.current_num_threads(), 2);
        POOL.install(|| {
            self.partitions
                .par_iter()
                .with_min_len(min_len)
                .enumerate()
                .for_each(|(part, part_buf)| {
                    if let Some(df) = part_buf.finish() {
                        io_thread.dump_partition_local(part as IdxSize, df)
                    }
                })
        });
        eprintln!("PARTITIONED FORCE SPILLED")
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sort_ooc(
    io_thread: IOThread,
    // these partitions are the samples
    // these are not yet assigned to a buckets
    samples: Series,
    idx: usize,
    descending: bool,
    nulls_last: bool,
    slice: Option<(i64, usize)>,
    verbose: bool,
    memtrack: MemTracker,
    ooc_start: Instant,
) -> PolarsResult<FinalizedSink> {
    let now = Instant::now();
    let multithreaded_partition = std::env::var("POLARS_OOC_SORT_PAR_PARTITION").is_ok();
    let spill_size = std::env::var("POLARS_OOC_SORT_SPILL_SIZE")
        .map(|v| v.parse::<usize>().expect("integer"))
        .unwrap_or(1 << 26);
    let samples = samples.to_physical_repr().into_owned();
    let spill_size = std::cmp::min(
        memtrack.get_available_latest() / (samples.len() * 3),
        spill_size,
    );

    // we collect as I am not sure that if we write to the same directory the
    // iterator will read those also.
    // We don't want to merge files we just written to disk
    let dir = &io_thread.dir;
    let files = std::fs::read_dir(dir)?.collect::<std::io::Result<Vec<_>>>()?;

    if verbose {
        eprintln!("spill size: {} mb", spill_size / 1024 / 1024);
        eprintln!("processing {} files", files.len());
    }

    let partitions_spiller = PartitionSpiller::new(samples.len(), spill_size as u64);

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
            let (iter, unique_assigned_parts) =
                partition_df(df, &assigned_parts, multithreaded_partition)?;
            for (part, df) in unique_assigned_parts.into_no_null_iter().zip(iter) {
                if let Some(df) = partitions_spiller.push(part as usize, df) {
                    io_thread.dump_partition_local(part, df)
                }
            }
            io_thread.clean(path);
            PolarsResult::Ok(())
        })
    })?;
    if verbose {
        eprintln!("partitioning sort took: {:?}", now.elapsed());
    }

    // Branch for testing so we hit different parts in the Source phase.
    #[cfg(debug_assertions)]
    {
        if std::env::var("POLARS_SPILL_SORT_PARTITIONS").is_ok() {
            partitions_spiller.spill_all(&io_thread)
        }
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

    let source = SortSource::new(
        files,
        idx,
        descending,
        nulls_last,
        slice,
        verbose,
        io_thread,
        memtrack,
        ooc_start,
        partitions_spiller,
    );
    Ok(FinalizedSink::Source(Box::new(source)))
}

fn det_partitions(s: &Series, partitions: &Series, descending: bool) -> IdxCa {
    let s = s.to_physical_repr();

    search_sorted(partitions, &s, SearchSortedSide::Any, descending).unwrap()
}

fn partition_df(
    df: DataFrame,
    partitions: &IdxCa,
    multithreaded: bool,
) -> PolarsResult<(DfIter, IdxCa)> {
    let groups = partitions.group_tuples(multithreaded, false)?;
    let partitions = unsafe { partitions.clone().into_series().agg_first(&groups) };
    let partitions = partitions.idx().unwrap().clone();

    let out = match groups {
        GroupsProxy::Idx(idx) => {
            let iter = idx.into_iter().map(move |(_, group)| {
                // groups are in bounds and sorted
                unsafe {
                    df._take_unchecked_slice_sorted(&group, multithreaded, IsSorted::Ascending)
                }
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
