use polars_arrow::array::builder::ShareStrategy;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::sort::arg_sort;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_expr::groups::new_hash_grouper;
use polars_expr::hash_keys::HashKeys;
use polars_expr::state::ExecutionState;
use polars_ooc::{
    LeastRecentSpillContext, MostRecentSpillContext, ParameterFreeSpillContext, SpillFrame,
    memory_manager,
};
use polars_utils::IdxSize;
use polars_utils::aliases::PlRandomState;
use polars_utils::hashing::HashPartitioner;
use polars_utils::pl_str::PlSmallStr;
use rayon::prelude::*;

use super::flush::{Flush, OutputRange, OutputUnit};
use super::tuning::{RowSizes, WindowTuning};
use super::{StoredMorsel, WindowOutput, WindowParams};
use crate::hash_shuffle::split::{SplitRows, split};

/// The column with the input row of each row in the partitions.
pub(super) const ROW_ID: PlSmallStr = PlSmallStr::from_static("__POLARS_WINDOW_ROW");
/// A partition that needs too much memory is split again at most this many times.
const MAX_RESPLIT_DEPTH: usize = 3;
/// The most parts a partition is split into at once.
const MAX_RESPLIT_PARTS: usize = 64;
/// Bytes per row assumed for a window column whose type has no fixed width.
const VARIABLE_WIDTH_BYTES: usize = 16;

/// Partitions the input, evaluates the windows of every partition and returns the output.
pub(super) fn evaluate(
    params: &WindowParams,
    tuning: &WindowTuning,
    num_pipelines: usize,
    morsels: Vec<Vec<StoredMorsel>>,
    morsel_ctx: LeastRecentSpillContext,
    state: &ExecutionState,
) -> PolarsResult<Flush> {
    // Morsels can share a sequence number, so ties keep the order of pipelines and arrival.
    let mut morsels: Vec<_> = morsels.into_iter().flatten().collect();
    morsels.sort_by_key(|m| m.seq);
    let mut first_rows = Vec::with_capacity(morsels.len());
    let mut num_rows = 0usize;
    for m in &morsels {
        first_rows.push(num_rows as IdxSize);
        num_rows += m.sf.height();
    }
    let partition_ctx = MostRecentSpillContext::new("window-partitions".into(), None);
    if num_rows == 0 {
        return Ok(Flush::new(
            Vec::new(),
            tuning.output_ahead_bytes(),
            (morsel_ctx, partition_ctx),
        ));
    }

    let keep_morsels = params.output != WindowOutput::Partitions;
    let read_bytes: usize = morsels.iter().map(|m| m.read_bytes).sum();
    let row_id_bytes = if keep_morsels {
        size_of::<IdxSize>()
    } else {
        0
    };
    let mut row_sizes = RowSizes {
        read: read_bytes / num_rows + row_id_bytes,
        window: params
            .exprs
            .iter()
            .map(|(name, _)| {
                let dtype = params.output_schema.get(name).unwrap();
                dtype
                    .byte_width()
                    .map_or(VARIABLE_WIDTH_BYTES, |w| w.ceil() as usize)
            })
            .sum(),
    };
    let num_partitions = tuning.num_partitions(num_rows, row_sizes.work(), num_pipelines);
    let random_state = PlRandomState::default();
    let range_ends = if keep_morsels {
        range_ends(&morsels, row_sizes.window, tuning, num_pipelines)
    } else {
        Vec::new()
    };

    let shuffle = Shuffle {
        params,
        batch_bytes: tuning.batch_bytes(),
        spill_ctx: &partition_ctx,
    };
    let inputs = if keep_morsels {
        morsels
            .iter()
            .zip(&first_rows)
            .map(|(m, first_row)| ShuffleInput::Kept {
                sf: &m.sf,
                first_row: *first_row,
                bytes: m.bytes,
            })
            .collect()
    } else {
        morsels
            .drain(..)
            .map(|m| ShuffleInput::Taken {
                sf: m.sf,
                bytes: m.bytes,
            })
            .collect()
    };
    let partitions = shuffle.run(inputs, num_partitions, &random_state)?;
    let max_work = tuning.max_partition_bytes();
    let partitions = split_large_partitions(&shuffle, partitions, &row_sizes, max_work)?;

    if polars_core::config::verbose() {
        eprintln!(
            "window: {num_rows} rows into {num_partitions} partitions, {} after splitting large ones, {} output ranges",
            partitions.len(),
            range_ends.len()
        );
    }

    let mut morsels = morsels.into_iter();
    let mut start = 0;
    let mut ranges: Vec<_> = range_ends
        .into_iter()
        .map(|end| {
            let range_morsels: Vec<_> = morsels.by_ref().take(end - start).collect();
            let range = OutputRange {
                first_row: first_rows[start],
                num_rows: range_morsels.iter().map(|m| m.sf.height()).sum(),
                bytes: range_morsels.iter().map(|m| m.bytes).sum(),
                morsels: range_morsels.into_iter().map(|m| m.sf).collect(),
                window_frames: Vec::new(),
            };
            start = end;
            range
        })
        .collect();
    let range_starts: Vec<_> = ranges.iter().map(|r| r.first_row).collect();

    let mut outputs = Vec::new();
    let mut partitions = partitions.into_iter().peekable();
    while partitions.peek().is_some() {
        // Evaluate as many partitions together as fit in the working memory, at least one.
        let mut batch = Vec::new();
        let mut batch_work = 0usize;
        while let Some((_, rows)) = partitions.peek() {
            let work = rows.saturating_mul(row_sizes.work());
            if !batch.is_empty() && batch_work.saturating_add(work) > max_work {
                break;
            }
            batch_work = batch_work.saturating_add(work);
            batch.push(partitions.next().unwrap().0);
        }
        let reservation = memory_manager().reserve_blocking(batch_work as u64);

        let results = RAYON.install(|| {
            batch
                .into_par_iter()
                .map(|frames| {
                    evaluate_stored_partition(params, frames, &range_starts, &random_state, state)
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;
        let rows: usize = results.iter().map(|p| p.df.height()).sum();
        let window_bytes: usize = results.iter().map(|p| p.window_bytes).sum();
        row_sizes.window = window_bytes / rows.max(1);

        if keep_morsels {
            write_ranges(&results, &mut ranges, &partition_ctx);
        } else {
            outputs.extend(results.into_iter().map(|p| {
                let bytes = p.df.estimated_size(false);
                let sf = SpillFrame::new_unregistered(p.df);
                partition_ctx.register_no_spill_check(&sf);
                OutputUnit::Frame(sf, bytes)
            }));
        }
        drop(reservation);
        memory_manager().spill_blocking();
    }

    if keep_morsels {
        outputs = ranges.into_iter().map(OutputUnit::Range).collect();
    }
    Ok(Flush::new(
        outputs,
        tuning.output_ahead_bytes(),
        (morsel_ctx, partition_ctx),
    ))
}

/// Cuts the morsels into output ranges of consecutive morsels and returns the end of each range.
fn range_ends(
    morsels: &[StoredMorsel],
    window_bytes_per_row: usize,
    tuning: &WindowTuning,
    num_pipelines: usize,
) -> Vec<usize> {
    let out_bytes = |m: &StoredMorsel| m.bytes + m.sf.height() * window_bytes_per_row;
    let total_bytes = morsels.iter().map(out_bytes).sum();
    let range_bytes = tuning.range_bytes(num_pipelines, total_bytes);
    let mut ends = Vec::new();
    let mut bytes = 0;
    for (i, m) in morsels.iter().enumerate() {
        if i > 0 && bytes + out_bytes(m) > range_bytes {
            ends.push(i);
            bytes = 0;
        }
        bytes += out_bytes(m);
    }
    ends.push(morsels.len());
    ends
}

/// Splits the partitions whose working memory exceeds `max_work` again, with new hashes, and
/// returns the non-empty partitions with their number of rows, the largest first.
fn split_large_partitions(
    shuffle: &Shuffle<'_>,
    partitions: Vec<Vec<SpillFrame>>,
    row_sizes: &RowSizes,
    max_work: usize,
) -> PolarsResult<Vec<(Vec<SpillFrame>, usize)>> {
    let mut todo: Vec<_> = partitions.into_iter().map(|p| (p, 0)).collect();
    let mut out = Vec::with_capacity(todo.len());
    while let Some((frames, depth)) = todo.pop() {
        let rows: usize = frames.iter().map(|sf| sf.height()).sum();
        if rows == 0 {
            continue;
        }
        let work = rows.saturating_mul(row_sizes.work());
        if work <= max_work || depth == MAX_RESPLIT_DEPTH {
            out.push((frames, rows));
            continue;
        }
        let num_parts = work.div_ceil(max_work).clamp(2, MAX_RESPLIT_PARTS);
        let inputs = frames
            .into_iter()
            .map(|sf| {
                let bytes = sf.height() * row_sizes.read;
                ShuffleInput::Partition { sf, bytes }
            })
            .collect();
        // New hashes, so that keys that went to one partition can go to different parts.
        let mut parts = shuffle.run(inputs, num_parts, &PlRandomState::default())?;
        // All rows in one part, as with a single key: splitting again does not help.
        if let Some(i) = parts
            .iter()
            .position(|p| p.iter().map(|sf| sf.height()).sum::<usize>() == rows)
        {
            out.push((parts.swap_remove(i), rows));
            continue;
        }
        todo.extend(parts.into_iter().map(|p| (p, depth + 1)));
    }
    // The largest first, so the small ones fill up the last batches.
    out.sort_by_key(|(_, rows)| std::cmp::Reverse(*rows));
    Ok(out)
}

/// A frame to split into partitions.
enum ShuffleInput<'a> {
    /// An input morsel that is output later: its read columns are split, with the input row of
    /// each row.
    Kept {
        sf: &'a SpillFrame,
        first_row: IdxSize,
        bytes: usize,
    },
    /// An input morsel that is not needed again: its read columns are split.
    Taken { sf: SpillFrame, bytes: usize },
    /// A frame of a partition, split whole.
    Partition { sf: SpillFrame, bytes: usize },
}

impl ShuffleInput<'_> {
    fn bytes(&self) -> usize {
        match self {
            Self::Kept { bytes, .. }
            | Self::Taken { bytes, .. }
            | Self::Partition { bytes, .. } => *bytes,
        }
    }
}

/// A frame loaded to be split.
struct LoadedFrame {
    df: DataFrame,
    /// For a kept morsel, the input row of its first row.
    first_row: Option<IdxSize>,
    /// `None`: all rows, in partition 0.
    split: Option<SplitRows>,
}

impl LoadedFrame {
    /// The rows of partition `p`; `None` for all rows.
    fn partition_rows(&self, p: usize) -> Option<&[IdxSize]> {
        self.split.as_ref().map(|split| split.partition(p))
    }
}

struct Shuffle<'a> {
    params: &'a WindowParams,
    batch_bytes: usize,
    spill_ctx: &'a MostRecentSpillContext,
}

impl Shuffle<'_> {
    /// Splits the rows of `inputs` into `num_partitions` partitions on the hash of their keys.
    /// The inputs are loaded in batches of about `batch_bytes`, and every batch adds at most one
    /// frame to each partition. The rows of a partition keep the order of `inputs`.
    fn run(
        &self,
        inputs: Vec<ShuffleInput<'_>>,
        num_partitions: usize,
        random_state: &PlRandomState,
    ) -> PolarsResult<Vec<Vec<SpillFrame>>> {
        let partitioner = HashPartitioner::new(num_partitions, 0);
        let mut partitions: Vec<Vec<SpillFrame>> =
            (0..num_partitions).map(|_| Vec::new()).collect();

        let mut inputs = inputs.into_iter().peekable();
        while inputs.peek().is_some() {
            let mut batch = Vec::new();
            let mut batch_bytes = 0;
            while let Some(input) = inputs.peek() {
                if !batch.is_empty() && batch_bytes + input.bytes() > self.batch_bytes {
                    break;
                }
                batch_bytes += input.bytes();
                batch.push(inputs.next().unwrap());
            }

            let loaded = RAYON.install(|| {
                batch
                    .into_par_iter()
                    .map(|input| {
                        let (df, first_row) = self.load(input)?;
                        let split = if num_partitions == 1 {
                            None
                        } else {
                            let keys = df.select(self.params.partition_by.iter().cloned())?;
                            let keys = HashKeys::from_df(&keys, random_state.clone(), true, false);
                            split(&keys, None, &partitioner)
                        };
                        PolarsResult::Ok(LoadedFrame {
                            df,
                            first_row,
                            split,
                        })
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            })?;
            // The inputs of one call are of one kind, so the frames have one schema.
            let schema = loaded[0].df.schema().clone();
            let add_row_ids = loaded[0].first_row.is_some();

            RAYON.install(|| {
                partitions
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(p, frames)| {
                        let num_rows: usize = loaded
                            .iter()
                            .map(|f| f.partition_rows(p).map_or(f.df.height(), |rows| rows.len()))
                            .sum();
                        if num_rows == 0 {
                            return;
                        }
                        let mut builder = DataFrameBuilder::new(schema.clone());
                        builder.reserve(num_rows);
                        let mut row_ids =
                            Vec::with_capacity(if add_row_ids { num_rows } else { 0 });
                        for f in &loaded {
                            match f.partition_rows(p) {
                                None => {
                                    builder.extend(&f.df, ShareStrategy::Never);
                                    if let Some(first) = f.first_row {
                                        row_ids.extend(first..first + f.df.height() as IdxSize);
                                    }
                                },
                                Some([]) => {},
                                Some(rows) => {
                                    // SAFETY: the rows were split from this frame.
                                    unsafe {
                                        builder.gather_extend(&f.df, rows, ShareStrategy::Never)
                                    };
                                    if let Some(first) = f.first_row {
                                        row_ids.extend(rows.iter().map(|row| first + row));
                                    }
                                },
                            }
                        }
                        let mut df = builder.freeze();
                        if add_row_ids {
                            let row_ids = IdxCa::from_vec(ROW_ID, row_ids).into_column();
                            // SAFETY: the row ids have the height of `df` and a new name.
                            unsafe { df.hstack_mut_unchecked(&[row_ids]) };
                        }
                        let sf = SpillFrame::new_unregistered(df);
                        self.spill_ctx.register_no_spill_check(&sf);
                        frames.push(sf);
                        memory_manager().spill_blocking();
                    })
            });
        }
        Ok(partitions)
    }

    /// The columns of `input` to split, each in one chunk, and for a kept morsel the input row of
    /// its first row.
    fn load(&self, input: ShuffleInput<'_>) -> PolarsResult<(DataFrame, Option<IdxSize>)> {
        let read_columns = |df: &DataFrame| {
            self.params
                .read_schema
                .iter_names()
                .map(|name| Ok(df.column(name)?.rechunk()))
                .collect::<PolarsResult<Vec<_>>>()
        };
        Ok(match input {
            ShuffleInput::Kept { sf, first_row, .. } => {
                let df = sf.get_blocking();
                // SAFETY: the columns have the same height and unique names.
                let df = unsafe { DataFrame::new_unchecked(df.height(), read_columns(&df)?) };
                (df, Some(first_row))
            },
            ShuffleInput::Taken { sf, .. } => {
                let df = sf.into_df_blocking();
                // SAFETY: the columns have the same height and unique names.
                let df = unsafe { DataFrame::new_unchecked(df.height(), read_columns(&df)?) };
                (df, None)
            },
            ShuffleInput::Partition { sf, .. } => {
                let mut df = sf.into_df_blocking();
                df.rechunk_mut();
                (df, None)
            },
        })
    }
}

/// The evaluated windows of one partition.
struct EvaluatedPartition {
    /// With [`WindowOutput::Partitions`], the sorted rows with the window columns appended.
    /// Otherwise the window columns with the input row of each row.
    df: DataFrame,
    /// The rows of `df` in each output range, `None` if there is one range.
    rows_per_range: Option<SplitRows>,
    /// The estimated size of the window columns.
    window_bytes: usize,
}

impl EvaluatedPartition {
    /// The rows of output range `r`, `None` for all rows.
    fn range_rows(&self, r: usize) -> Option<&[IdxSize]> {
        self.rows_per_range.as_ref().map(|rows| rows.partition(r))
    }
}

/// Loads a partition, evaluates its windows and returns them.
fn evaluate_stored_partition(
    params: &WindowParams,
    frames: Vec<SpillFrame>,
    range_starts: &[IdxSize],
    random_state: &PlRandomState,
    state: &ExecutionState,
) -> PolarsResult<EvaluatedPartition> {
    let dfs: Vec<_> = frames.into_iter().map(|sf| sf.into_df_blocking()).collect();
    let mut df = accumulate_dataframes_vertical_unchecked(dfs);
    df.rechunk_mut();

    if params.output == WindowOutput::Partitions {
        let SortedPartition {
            mut df, windows, ..
        } = evaluate_partition(params, df, random_state, state)?;
        let window_bytes = columns_size(&windows);
        // SAFETY: the window columns have the height of `df` and new names.
        unsafe { df.hstack_mut_unchecked(&windows) };
        return Ok(EvaluatedPartition {
            df,
            rows_per_range: None,
            window_bytes,
        });
    }

    // The sort would reorder the row ids too, so they are kept aside.
    let row_ids = df.drop_in_place(&ROW_ID)?;
    let SortedPartition {
        df: sorted,
        mut windows,
        gathered_row_for_sorted_row,
    } = evaluate_partition(params, df, random_state, state)?;
    drop(sorted);
    let row_ids = row_ids.idx()?.rechunk();
    let gathered_row_ids = row_ids.downcast_as_array().values();

    let rows_per_range = (range_starts.len() > 1).then(|| {
        // The gathered rows are in input order, so their ranges follow in one pass.
        let mut counts: Vec<IdxSize> = vec![0; range_starts.len()];
        let mut r = 0;
        let range_per_gathered_row: Vec<IdxSize> = gathered_row_ids
            .iter()
            .map(|row| {
                while r + 1 < range_starts.len() && range_starts[r + 1] <= *row {
                    r += 1;
                }
                counts[r] += 1;
                r as IdxSize
            })
            .collect();
        let num_rows = range_per_gathered_row.len();
        // SAFETY: the ranges are below `counts.len()`, the sort only permutes the gathered rows,
        // and a permutation keeps the counts.
        unsafe {
            match &gathered_row_for_sorted_row {
                None => SplitRows::from_counts(&counts, num_rows, |i| {
                    *range_per_gathered_row.get_unchecked(i)
                }),
                Some(gathered_rows) => {
                    let gathered_rows = gathered_rows.downcast_as_array().values();
                    SplitRows::from_counts(&counts, num_rows, |i| {
                        let row = *gathered_rows.get_unchecked(i) as usize;
                        *range_per_gathered_row.get_unchecked(row)
                    })
                },
            }
        }
    });
    let row_ids = match gathered_row_for_sorted_row {
        None => row_ids.into_owned(),
        // SAFETY: the sort only permutes the rows.
        Some(gathered_rows) => unsafe { row_ids.take_unchecked(&gathered_rows) },
    };
    let window_bytes = columns_size(&windows);
    windows.push(row_ids.into_column());
    let height = windows[0].len();
    // SAFETY: the columns have the same height and unique names.
    let mut df = unsafe { DataFrame::new_unchecked(height, windows) };
    if rows_per_range.is_some() {
        // Gathering rows needs single-chunk columns.
        df.rechunk_mut();
    }
    Ok(EvaluatedPartition {
        df,
        rows_per_range,
        window_bytes,
    })
}

fn columns_size(columns: &[Column]) -> usize {
    columns.iter().map(|c| c.estimated_size(false)).sum()
}

/// Copies the window rows of `partitions` into one new frame per output range.
fn write_ranges(
    partitions: &[EvaluatedPartition],
    ranges: &mut [OutputRange],
    spill_ctx: &MostRecentSpillContext,
) {
    let schema = partitions[0].df.schema().clone();
    RAYON.install(|| {
        ranges.par_iter_mut().enumerate().for_each(|(r, range)| {
            let num_rows: usize = partitions
                .iter()
                .map(|p| p.range_rows(r).map_or(p.df.height(), |rows| rows.len()))
                .sum();
            if num_rows == 0 {
                return;
            }
            // A copy, so the frame of a range does not keep the whole partition in memory.
            let mut builder = DataFrameBuilder::new(schema.clone());
            builder.reserve(num_rows);
            for p in partitions {
                match p.range_rows(r) {
                    None => builder.extend(&p.df, ShareStrategy::Never),
                    Some([]) => {},
                    // SAFETY: the rows were split from this frame.
                    Some(rows) => unsafe {
                        builder.gather_extend(&p.df, rows, ShareStrategy::Never)
                    },
                }
            }
            let df = builder.freeze();
            range.bytes += df.estimated_size(false);
            let sf = SpillFrame::new_unregistered(df);
            spill_ctx.register_no_spill_check(&sf);
            range.window_frames.push(sf);
            memory_manager().spill_blocking();
        })
    });
}

/// The windows of one partition, evaluated on its rows sorted by partition and order key.
struct SortedPartition {
    /// The sorted rows.
    df: DataFrame,
    /// The window columns for the rows of `df`.
    windows: Vec<Column>,
    /// The row of the gathered partition for each row of `df`. `None` if the order did not
    /// change.
    gathered_row_for_sorted_row: Option<IdxCa>,
}

/// Sorts the rows of one partition by partition and order key and evaluates the windows on them.
fn evaluate_partition(
    params: &WindowParams,
    df: DataFrame,
    random_state: &PlRandomState,
    state: &ExecutionState,
) -> PolarsResult<SortedPartition> {
    let height = df.height();

    // Dense partition ids, in order of first occurrence.
    let keys = df.select(params.partition_by.iter().cloned())?;
    let hash_keys = HashKeys::from_df(&keys, random_state.clone(), true, false);
    let mut grouper = new_hash_grouper(keys.schema().clone());
    let all_rows = (0..height as IdxSize).collect::<Vec<_>>();
    let mut group_idxs = Vec::with_capacity(height);
    // SAFETY: all indices are in-bounds.
    unsafe { grouper.insert_keys_subset(&hash_keys, &all_rows, Some(&mut group_idxs)) };
    let num_groups = grouper.num_groups() as usize;
    drop(all_rows);

    // The rows sorted by the order key, stable if ties must stay in input order.
    let order_idx = match &params.order_by {
        None => None,
        Some((order_by, options)) => {
            let keys = order_by
                .iter()
                .map(|name| df.column(name).cloned())
                .collect::<PolarsResult<Vec<_>>>()?;
            let options = SortMultipleOptions {
                maintain_order: params.ordered_eval,
                ..options.clone()
            };
            let order_idx = arg_sort(&keys, options)?;
            Some(order_idx.rechunk().downcast_as_array().values().clone())
        },
    };

    // Stable counting sort on the partition id, visiting the rows in order key order.
    let mut offsets = vec![0 as IdxSize; num_groups + 1];
    for g in &group_idxs {
        // SAFETY: the group ids are below `num_groups`.
        unsafe { *offsets.get_unchecked_mut(*g as usize + 1) += 1 };
    }
    let mut sum = 0;
    for offset in &mut offsets {
        sum += *offset;
        *offset = sum;
    }
    let gathered_row_for_sorted_row = if num_groups == 1 {
        order_idx.map_or_else(|| (0..height as IdxSize).collect(), |idx| idx.to_vec())
    } else {
        let mut gathered_rows = vec![0 as IdxSize; height];
        let mut next = offsets.clone();
        let mut place = |row: IdxSize| {
            // SAFETY: `row` is below `height` and its group id below `num_groups`. A group gets as
            // many positions as it has rows, so `pos` stays below `height`.
            unsafe {
                let pos = next.get_unchecked_mut(*group_idxs.get_unchecked(row as usize) as usize);
                *gathered_rows.get_unchecked_mut(*pos as usize) = row;
                *pos += 1;
            }
        };
        match &order_idx {
            None => (0..height as IdxSize).for_each(&mut place),
            Some(order_idx) => order_idx.iter().copied().for_each(&mut place),
        }
        gathered_rows
    };
    drop(group_idxs);

    let is_identity = gathered_row_for_sorted_row
        .iter()
        .enumerate()
        .all(|(i, row)| i as IdxSize == *row);
    let (df, gathered_row_for_sorted_row) = if is_identity {
        (df, None)
    } else {
        let gathered_rows = IdxCa::from_vec(PlSmallStr::EMPTY, gathered_row_for_sorted_row);
        // SAFETY: `gathered_rows` is a permutation of the rows of `df`.
        let sorted = unsafe { df.take_unchecked(&gathered_rows) };
        drop(df);
        (sorted, Some(gathered_rows))
    };

    let groups = GroupsType::Slice {
        groups: offsets.windows(2).map(|w| [w[0], w[1] - w[0]]).collect(),
        overlapping: false,
        monotonic: true,
    }
    .into_sliceable();

    let windows = params
        .exprs
        .iter()
        .map(|(name, expr)| {
            let out = expr.evaluate_on_sorted_partitions(&df, groups.clone(), state)?;
            Ok(out.with_name(name.clone()))
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(SortedPartition {
        df,
        windows,
        gathered_row_for_sorted_row,
    })
}
