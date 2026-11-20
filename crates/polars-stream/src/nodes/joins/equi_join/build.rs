use std::sync::atomic::AtomicU64;

use futures::FutureExt;
use polars_arrow::array::builder::ShareStrategy;
use polars_async::parallel::for_each_parallel;
use polars_core::config;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::schema::SchemaRef;
use polars_expr::idx_table::IdxTable;
use polars_ooc::{MemoryGrant, MostRecentSpillContext, memory_manager};
use polars_utils::IdxSize;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::hashing::HashPartitioner;

use super::probe::{ProbePartitions, ProbeState, ProbeTable};
use super::{EquiJoinParams, JoinPartition, select_keys_with_columns, select_payload};
use crate::hash_shuffle::batches::{
    BuildPartition, build_from_batches, make_batches, make_ordered_batches,
};
use crate::hash_shuffle::frame_with_keys::{FrameWithKeys, SpillFrameWithKeys};
use crate::hash_shuffle::split::{NullRows, rows_or_all, unhashed_validity};
use crate::hash_shuffle::store::StoredPartition;
use crate::hash_shuffle::writer::PartitionedWriter;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::joins::BufferedStream;
use crate::nodes::joins::runtime_filter::KeyFilterBuilder;
use crate::nodes::joins::tuning::{BuildStats, JoinTuning};

#[derive(Default)]
pub struct LocalBuilder {
    // The morsels seen by this builder, with their hashed keys.
    frames: Vec<(MorselSeq, SpillFrameWithKeys)>,

    // A cardinality sketch of the keys seen by this builder.
    sketch: CardinalitySketch,

    // The rows and bytes seen by this builder.
    stats: BuildStats,

    // The key of each runtime filter seen by this builder.
    key_filters: Vec<KeyFilterBuilder>,
}

/// The hash table and payload of one partition while they are built.
pub struct PartitionBuilder {
    index: usize,
    table: Box<dyn IdxTable>,
    payload: DataFrameBuilder,
    // Position of each payload row in the full build input, if tracked.
    row_positions: Vec<u64>,
    all_rows: Vec<IdxSize>,
}

impl PartitionBuilder {
    /// Reserves room for a quarter more keys than `estimated_keys`.
    pub fn new(
        index: usize,
        table: &dyn IdxTable,
        payload_schema: SchemaRef,
        estimated_keys: usize,
    ) -> Self {
        let mut table = table.new_empty();
        table.reserve(estimated_keys * 5 / 4);
        Self {
            index,
            table,
            payload: DataFrameBuilder::new(payload_schema),
            row_positions: Vec::new(),
            all_rows: Vec::new(),
        }
    }

    /// Inserts the rows `rows` of `frame`, or all its rows. If `first_row` is given, the position
    /// of the first row of `frame` in the full build input, the positions of the rows are kept.
    pub fn add_rows(
        &mut self,
        frame: &FrameWithKeys,
        rows: Option<&[IdxSize]>,
        track_unmatchable: bool,
        first_row: Option<u64>,
    ) {
        let keys = &frame.keys;
        let rows = rows_or_all(rows, &mut self.all_rows, keys.len());
        if rows.is_empty() {
            return;
        }
        unsafe {
            self.table.insert_keys_subset(keys, rows, track_unmatchable);
            self.payload
                .gather_extend(&frame.frame, rows, ShareStrategy::Never);
        }
        if let Some(first_row) = first_row {
            #[allow(clippy::unnecessary_cast)] // Necessary when IdxSize = u64.
            self.row_positions
                .extend(rows.iter().map(|i| first_row + *i as u64));
        }
    }

    pub fn finish(self) -> ProbeTable {
        ProbeTable {
            hash_table: self.table,
            payload: self.payload.freeze(),
            row_positions: self.row_positions,
        }
    }
}

/// Where the build rows of one partition go: into a table, or into stored frames.
enum BuildTarget {
    Table(PartitionBuilder),
    Stored(StoredPartition),
}

pub struct BuildState {
    pub local_builders: Vec<LocalBuilder>,
    sampled_probe_morsels: BufferedStream,
}

impl BuildState {
    pub fn new(
        num_pipelines: usize,
        params: &EquiJoinParams,
        sampled_probe_morsels: BufferedStream,
    ) -> Self {
        let local_builders = (0..num_pipelines)
            .map(|_| LocalBuilder {
                key_filters: if params.publishes_runtime_filters() {
                    params.runtime_filters.new_builders()
                } else {
                    Vec::new()
                },
                ..Default::default()
            })
            .collect();
        Self {
            local_builders,
            sampled_probe_morsels,
        }
    }

    pub async fn sink(
        mut recv: PortReceiver,
        local: &mut LocalBuilder,
        params: &EquiJoinParams,
        state: &StreamingExecutionState,
        spill_ctx: &MostRecentSpillContext,
    ) -> PolarsResult<()> {
        let (key_selectors, payload_selector);
        if params.left_is_build.unwrap() {
            payload_selector = &params.left_payload_select;
            key_selectors = &params.left_key_selectors;
        } else {
            payload_selector = &params.right_payload_select;
            key_selectors = &params.right_key_selectors;
        };

        let publishes_runtime_filters = params.publishes_runtime_filters();
        while let Ok(morsel) = recv.recv().await {
            // Compute hashed keys and payload.
            let df = morsel.df().await;
            let (hash_keys, keys) =
                select_keys_with_columns(&df, key_selectors, params, &state.in_memory_exec_state)
                    .await?;
            if publishes_runtime_filters {
                params
                    .runtime_filters
                    .extend(&keys, &mut local.key_filters)?;
            }
            let payload = select_payload(df.clone(), payload_selector);

            hash_keys.sketch_cardinality(&mut local.sketch);
            let frame = FrameWithKeys {
                frame: payload,
                keys: hash_keys,
                mask: None,
            };
            local.stats.rows += frame.keys.len() as u64;
            local.stats.null_rows +=
                unhashed_validity(&frame.keys).map_or(0, |v| v.unset_bits()) as u64;
            local.stats.bytes += frame.estimated_size() as u64;
            local.stats.key_bytes += frame.keys.estimated_size() as u64;
            let frame = SpillFrameWithKeys::new(frame, spill_ctx).await;
            local.frames.push((morsel.seq(), frame));
        }
        Ok(())
    }

    /// Whether no row was built, sampled morsels included.
    pub fn is_empty(&self) -> bool {
        self.local_builders
            .iter()
            .all(|b| b.frames.iter().all(|(_, f)| f.num_rows() == 0))
    }

    /// Hand every build key to the runtime filters. Filters set from a sample
    /// keep their value, whichever side is built; filters of a side that was
    /// not built get a predicate that skips nothing.
    pub fn publish_runtime_filters(&mut self, params: &mut EquiJoinParams) {
        if !params.publishes_runtime_filters() {
            params.runtime_filters.publish_nothing();
            return;
        }
        let locals = self
            .local_builders
            .iter_mut()
            .map(|l| std::mem::take(&mut l.key_filters));
        params.runtime_filters.publish_merged(locals);
    }

    /// Takes the memory grant and builds the hash table and payload of every partition, loading
    /// the stored frames one batch at a time. Unordered joins whose tables do not fit in the grant
    /// store the partitions instead, and build the tables of the smallest that fit right away.
    pub fn finalize(
        &mut self,
        params: &EquiJoinParams,
        table: &dyn IdxTable,
        tuning: &JoinTuning,
        state: &StreamingExecutionState,
    ) -> PolarsResult<ProbeState> {
        let left_is_build = params.left_is_build.unwrap();
        let track_unmatchable = params.emit_unmatched_build();
        let payload_schema = params.payload_schema(left_is_build);
        let unordered = !params.preserve_order_probe;

        let mut sketch = CardinalitySketch::new();
        let mut frames = Vec::new();
        let mut stats = BuildStats::default();
        for local in &mut self.local_builders {
            sketch.combine(&local.sketch);
            frames.append(&mut local.frames);
            stats.add(&local.stats);
        }
        stats.estimated_keys = sketch.estimate() as u64;

        // Rows without a hash (null keys) never match. Ordered joins keep them in the tables to
        // emit them as unmatched rows in order; unordered joins keep them as frames of their own.
        let null_rows_in_tables = !unordered && track_unmatchable;
        let null_frames = unordered && track_unmatchable && !params.args.nulls_equal;
        let nulls = if null_frames {
            NullRows::Separate
        } else if null_rows_in_tables {
            NullRows::FirstPartition
        } else {
            NullRows::Skip
        };

        let table_bytes = stats.all_tables_bytes(null_rows_in_tables);
        // Ordered joins keep all their tables in memory, so they ask for all of it.
        let wanted_grant_bytes = if unordered {
            table_bytes.min(tuning.max_grant_bytes())
        } else {
            table_bytes
        };
        let grant = MemoryGrant::take(wanted_grant_bytes, tuning.max_grant_fraction);
        let stored_partitions = if unordered {
            tuning.num_partitions(table_bytes, grant.bytes(), state.num_pipelines)
        } else {
            None
        };
        let num_partitions = stored_partitions.unwrap_or(state.num_pipelines);
        let partitioner = HashPartitioner::new(num_partitions, 0);
        let partition_keys = sketch.estimate() / num_partitions;

        let num_frames = frames.len();
        let mut first_rows = Vec::new();
        if params.preserve_order_build {
            // Morsels can share a sequence number, so ties keep the order of pipelines and arrival.
            frames.sort_by_key(|(seq, _)| *seq);
            if track_unmatchable {
                let mut next_row = 0;
                for (_, frame) in &frames {
                    first_rows.push(next_row);
                    next_row += frame.num_rows() as u64;
                }
            }
        }
        let frames = frames.into_iter().map(|(_, f)| (0, f)).collect();
        let batch_bytes = tuning.batch_bytes();
        let batches = if params.preserve_order_build {
            make_ordered_batches(frames, batch_bytes)
        } else {
            make_batches(frames, batch_bytes)
        };
        let num_batches = batches.len();

        let spill_ctx =
            MostRecentSpillContext::new("equi-join-partition".into(), state.task_metrics.clone());
        // With `NullRows::Separate` the rows without a hash are stored after the last partition.
        let mut targets: Vec<BuildTarget> = (0..num_partitions + usize::from(null_frames))
            .map(|index| {
                if stored_partitions.is_none() && index < num_partitions {
                    let schema = payload_schema.clone();
                    BuildTarget::Table(PartitionBuilder::new(index, table, schema, partition_keys))
                } else {
                    BuildTarget::Stored(StoredPartition::new(index, 2, spill_ctx.clone()))
                }
            })
            .collect();

        let add_batch: &BuildPartition<'_, BuildTarget> = &|target, batch, _| {
            match target {
                BuildTarget::Stored(stored) => stored.write_batch(batch),
                BuildTarget::Table(p) if params.preserve_order_build => {
                    for frame in batch {
                        let rows = frame.partition_rows(p.index);
                        let first_row = first_rows.get(frame.index).copied();
                        p.add_rows(&frame.frame, rows, track_unmatchable, first_row);
                    }
                },
                BuildTarget::Table(p) => {
                    // Partitions start at different frames, so that tasks don't read the same
                    // frame at the same time.
                    let start = p.index * batch.len() / num_partitions;
                    for i in 0..batch.len() {
                        let frame = &batch[(start + i) % batch.len()];
                        let rows = frame.partition_rows(p.index);
                        p.add_rows(&frame.frame, rows, track_unmatchable, None);
                    }
                },
            }
            async { Ok(()) }.boxed()
        };
        let reservation = stored_partitions
            .is_none()
            .then(|| memory_manager().reserve_blocking(table_bytes.saturating_sub(stats.bytes)));
        build_from_batches(batches, &partitioner, &mut targets, nulls, add_batch, state)?;
        drop(reservation);

        let null_build_rows = null_frames.then(|| match targets.pop() {
            Some(BuildTarget::Stored(stored)) => stored,
            _ => unreachable!(),
        });
        let table_rows = if null_rows_in_tables {
            stats.rows
        } else {
            stats.rows - stats.null_rows
        };
        let mut partitions: Vec<JoinPartition> = targets
            .into_iter()
            .map(|target| match target {
                BuildTarget::Table(builder) => {
                    let stored = StoredPartition::new(builder.index, 2, spill_ctx.clone());
                    let table = builder.finish();
                    let share = table.payload.height() as f64 / table_rows.max(1) as f64;
                    JoinPartition {
                        stored,
                        table: Some(table),
                        table_bytes: (table_bytes as f64 * share) as u64,
                    }
                },
                BuildTarget::Stored(stored) => JoinPartition {
                    table_bytes: stats.table_bytes(stored.num_rows(0), stored.input_size(0)),
                    stored,
                    table: None,
                },
            })
            .collect();

        let num_in_memory = if stored_partitions.is_some() {
            let grant_bytes = grant.bytes();
            build_smallest_tables(
                &mut partitions,
                table,
                params,
                &stats,
                tuning,
                grant_bytes,
                state,
            )?
        } else {
            num_partitions
        };
        if config::verbose() {
            eprintln!(
                "[equi-join]: building {num_partitions} partitions ({num_in_memory} in memory) from {} rows ({} bytes, {} estimated keys) in {num_frames} frames, {num_batches} batches, {table_bytes} estimated table bytes, memory grant of {} bytes ({wanted_grant_bytes} wanted)",
                stats.rows,
                stats.bytes,
                stats.estimated_keys,
                grant.bytes(),
            );
        }

        let num_stored = num_partitions - num_in_memory;
        let writers = if num_stored > 0 {
            let frame_bytes = tuning.writer_frame_bytes(state.num_pipelines, num_stored);
            (0..state.num_pipelines)
                .map(|_| PartitionedWriter::new(num_partitions, frame_bytes, spill_ctx.clone()))
                .collect()
        } else {
            Vec::new()
        };
        Ok(ProbeState {
            partitions: ProbePartitions {
                partitions,
                partitioner,
                unordered_morsel_seq: AtomicU64::new(0),
            },
            writers,
            null_build_rows,
            stats,
            grant: Some(grant),
            max_seq_sent: MorselSeq::default(),
            sampled_probe_morsels: core::mem::take(&mut self.sampled_probe_morsels),
        })
    }
}

/// Builds the tables of the stored partitions with the smallest tables in parallel: as many as fit
/// in `grant_bytes`, or as forced. Returns how many.
fn build_smallest_tables(
    partitions: &mut [JoinPartition],
    table: &dyn IdxTable,
    params: &EquiJoinParams,
    stats: &BuildStats,
    tuning: &JoinTuning,
    grant_bytes: u64,
    state: &StreamingExecutionState,
) -> PolarsResult<usize> {
    let track_unmatchable = params.emit_unmatched_build();
    let mut by_size: Vec<&mut JoinPartition> = partitions.iter_mut().collect();
    by_size.sort_by_key(|p| p.table_bytes);
    let sizes: Vec<u64> = by_size.iter().map(|p| p.table_bytes).collect();
    let num_in_memory = tuning.num_in_memory_partitions(&sizes, grant_bytes);
    by_size.truncate(num_in_memory);

    let net_bytes = by_size.iter().map(|p| p.net_table_bytes()).sum();
    let _reservation = memory_manager().reserve_blocking(net_bytes);
    for_each_parallel(by_size, state.task_metrics(), &|p| async move {
        let mut builder = p.table_builder(table, params, stats);
        while let Some((_, frame)) = p.stored.pop_frame() {
            let frame = frame.load().await;
            builder.add_rows(&frame, None, track_unmatchable, None);
            memory_manager().spill().await;
        }
        p.table = Some(builder.finish());
        Ok(())
    })?;
    Ok(num_in_memory)
}
