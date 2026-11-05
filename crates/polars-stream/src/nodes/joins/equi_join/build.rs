use std::sync::atomic::AtomicU64;

use futures::FutureExt;
use polars_arrow::array::builder::ShareStrategy;
use polars_core::config;
use polars_core::frame::builder::DataFrameBuilder;
use polars_expr::idx_table::IdxTable;
use polars_ooc::MostRecentSpillContext;
use polars_utils::IdxSize;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::hashing::HashPartitioner;

use super::probe::{ProbeState, ProbeTable};
use super::{EquiJoinParams, select_keys_with_columns, select_payload};
use crate::hash_shuffle::batches::{
    BuildPartition, LoadedFrame, build_from_batches, make_batches, make_ordered_batches,
};
use crate::hash_shuffle::frame_with_keys::{FrameWithKeys, SpillFrameWithKeys};
use crate::hash_shuffle::split::{NullRows, rows_or_all};
use crate::nodes::compute_node_prelude::*;
use crate::nodes::joins::BufferedStream;
use crate::nodes::joins::runtime_filter::KeyFilterBuilder;

#[derive(Default)]
pub(super) struct LocalBuilder {
    // The morsels seen by this builder, with their hashed keys.
    frames: Vec<(MorselSeq, SpillFrameWithKeys)>,

    // A cardinality sketch of the keys seen by this builder.
    sketch: CardinalitySketch,

    // The key of each runtime filter seen by this builder.
    key_filters: Vec<KeyFilterBuilder>,
}

/// The hash table and payload of one partition while they are built.
struct PartitionBuilder {
    index: usize,
    table: Box<dyn IdxTable>,
    payload: DataFrameBuilder,
    // Position of each payload row in the full build input, if tracked.
    row_positions: Vec<u64>,
    all_rows: Vec<IdxSize>,
}

impl PartitionBuilder {
    /// Inserts the rows of this partition of `frame`. If `first_row` is given, the position of
    /// the first row of `frame` in the full build input, the positions of the rows are kept.
    fn add_rows(&mut self, frame: &LoadedFrame, track_unmatchable: bool, first_row: Option<u64>) {
        let keys = &frame.frame.keys;
        let rows = rows_or_all(
            frame.partition_rows(self.index),
            &mut self.all_rows,
            keys.len(),
        );
        if rows.is_empty() {
            return;
        }
        unsafe {
            self.table.insert_keys_subset(keys, rows, track_unmatchable);
            self.payload
                .gather_extend(&frame.frame.frame, rows, ShareStrategy::Never);
        }
        if let Some(first_row) = first_row {
            #[allow(clippy::unnecessary_cast)] // Necessary when IdxSize = u64.
            self.row_positions
                .extend(rows.iter().map(|i| first_row + *i as u64));
        }
    }
}

pub(super) struct BuildState {
    pub(super) local_builders: Vec<LocalBuilder>,
    sampled_probe_morsels: BufferedStream,
}

impl BuildState {
    pub(super) fn new(
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

    pub(super) async fn sink(
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
            let frame = SpillFrameWithKeys::new(frame, spill_ctx).await;
            local.frames.push((morsel.seq(), frame));
        }
        Ok(())
    }

    /// Whether no row was built, sampled morsels included.
    pub(super) fn is_empty(&self) -> bool {
        self.local_builders
            .iter()
            .all(|b| b.frames.iter().all(|(_, f)| f.num_rows() == 0))
    }

    /// Hand every build key to the runtime filters. Filters set from a sample
    /// keep their value, whichever side is built; filters of a side that was
    /// not built get a predicate that skips nothing.
    pub(super) fn publish_runtime_filters(&mut self, params: &mut EquiJoinParams) {
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

    /// Builds the hash table and payload of every partition, loading the stored frames one batch
    /// of about `batch_bytes` at a time.
    pub(super) fn finalize(
        &mut self,
        params: &EquiJoinParams,
        table: &dyn IdxTable,
        batch_bytes: usize,
        state: &StreamingExecutionState,
    ) -> PolarsResult<ProbeState> {
        let track_unmatchable = params.emit_unmatched_build();
        let payload_schema = if params.left_is_build.unwrap() {
            &params.left_payload_schema
        } else {
            &params.right_payload_schema
        };

        let mut sketch = CardinalitySketch::new();
        let mut frames = Vec::new();
        for local in &mut self.local_builders {
            sketch.combine(&local.sketch);
            frames.append(&mut local.frames);
        }
        let num_rows: usize = frames.iter().map(|(_, f)| f.num_rows()).sum();
        let num_bytes: usize = frames.iter().map(|(_, f)| f.estimated_size()).sum();

        let num_partitions = state.num_pipelines;
        let partitioner = HashPartitioner::new(num_partitions, 0);
        let reserve_keys = sketch.estimate() * 5 / 4 / num_partitions;
        let mut partitions: Vec<PartitionBuilder> = (0..num_partitions)
            .map(|index| {
                let mut table = table.new_empty();
                table.reserve(reserve_keys);
                PartitionBuilder {
                    index,
                    table,
                    payload: DataFrameBuilder::new(payload_schema.clone()),
                    row_positions: Vec::new(),
                    all_rows: Vec::new(),
                }
            })
            .collect();
        // Rows without a hash (null keys) never match, so they are only kept to be emitted as
        // unmatched rows.
        let nulls = if track_unmatchable {
            NullRows::FirstPartition
        } else {
            NullRows::Skip
        };

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
        let batches = if params.preserve_order_build {
            make_ordered_batches(frames, batch_bytes)
        } else {
            make_batches(frames, batch_bytes)
        };
        if config::verbose() {
            eprintln!(
                "[equi-join]: building {num_partitions} partitions from {num_rows} rows ({num_bytes} bytes) in {num_frames} frames, {} batches",
                batches.len()
            );
        }

        let add_batch: &BuildPartition<'_, PartitionBuilder> = &|p, batch, _| {
            if params.preserve_order_build {
                for frame in batch {
                    let first_row = first_rows.get(frame.index).copied();
                    p.add_rows(frame, track_unmatchable, first_row);
                }
            } else {
                // Partitions start at different frames, so that tasks don't read the same frame at
                // the same time.
                let start = p.index * batch.len() / num_partitions;
                for i in 0..batch.len() {
                    p.add_rows(&batch[(start + i) % batch.len()], track_unmatchable, None);
                }
            }
            async { Ok(()) }.boxed()
        };
        build_from_batches(
            batches,
            &partitioner,
            &mut partitions,
            nulls,
            add_batch,
            state,
        )?;

        let table_per_partition = partitions
            .into_iter()
            .map(|p| ProbeTable {
                hash_table: p.table,
                payload: p.payload.freeze(),
                row_positions: p.row_positions,
            })
            .collect();
        Ok(ProbeState {
            table_per_partition,
            max_seq_sent: MorselSeq::default(),
            sampled_probe_morsels: core::mem::take(&mut self.sampled_probe_morsels),
            unordered_morsel_seq: AtomicU64::new(0),
        })
    }
}
