use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use polars_arrow::array::builder::ShareStrategy;
use polars_async::executor;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_core::runtime::{ASYNC, RAYON};
use polars_expr::hash_keys::HashKeys;
use polars_expr::idx_table::IdxTable;
use polars_ooc::{MostRecentSpillContext, SpillFrame};
use polars_utils::IdxSize;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::hashing::HashPartitioner;
use polars_utils::itertools::Itertools;
use polars_utils::priority::Priority;
use polars_utils::sparse_init_vec::SparseInitVec;

use super::probe::{ProbeState, ProbeTable};
use super::{EquiJoinParams, select_keys_with_columns, select_payload};
use crate::nodes::compute_node_prelude::*;
use crate::nodes::joins::BufferedStream;
use crate::nodes::joins::runtime_filter::KeyFilterBuilder;

#[derive(Default)]
pub(super) struct LocalBuilder {
    // The complete list of morsels and their computed hashes seen by this builder.
    morsels: Vec<(MorselSeq, SpillFrame, HashKeys)>,

    // A cardinality sketch per partition for the keys seen by this builder.
    sketch_per_p: Vec<CardinalitySketch>,

    // The key of each runtime filter seen by this builder.
    key_filters: Vec<KeyFilterBuilder>,

    // morsel_idxs_values_per_p[p][start..stop] contains the offsets into morsels[i]
    // for partition p, where start, stop are:
    // let start = morsel_idxs_offsets[i * num_partitions + p];
    // let stop = morsel_idxs_offsets[(i + 1) * num_partitions + p];
    morsel_idxs_values_per_p: Vec<Vec<IdxSize>>,
    morsel_idxs_offsets_per_p: Vec<usize>,
}

pub(super) struct BuildState {
    pub(super) local_builders: Vec<LocalBuilder>,
    sampled_probe_morsels: BufferedStream,
}

impl BuildState {
    pub(super) fn new(
        num_pipelines: usize,
        num_partitions: usize,
        params: &EquiJoinParams,
        sampled_probe_morsels: BufferedStream,
    ) -> Self {
        let local_builders = (0..num_pipelines)
            .map(|_| LocalBuilder {
                morsels: Vec::new(),
                sketch_per_p: vec![CardinalitySketch::default(); num_partitions],
                key_filters: if params.publishes_runtime_filters() {
                    params.runtime_filters.new_builders()
                } else {
                    Vec::new()
                },
                morsel_idxs_values_per_p: vec![Vec::new(); num_partitions],
                morsel_idxs_offsets_per_p: vec![0; num_partitions],
            })
            .collect();
        Self {
            local_builders,
            sampled_probe_morsels,
        }
    }

    pub(super) async fn partition_and_sink(
        mut recv: PortReceiver,
        local: &mut LocalBuilder,
        partitioner: HashPartitioner,
        params: &EquiJoinParams,
        state: &StreamingExecutionState,
        spill_ctx: &MostRecentSpillContext,
    ) -> PolarsResult<()> {
        let track_unmatchable = params.emit_unmatched_build();
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
            // Compute hashed keys and payload. We must rechunk the payload for
            // later gathers.
            let df = morsel.df().await;
            let (hash_keys, keys) =
                select_keys_with_columns(&df, key_selectors, params, &state.in_memory_exec_state)
                    .await?;
            if publishes_runtime_filters {
                params
                    .runtime_filters
                    .extend(&keys, &mut local.key_filters)?;
            }
            let mut payload = select_payload(df.clone(), payload_selector);
            payload.rechunk_mut();

            hash_keys.gen_idxs_per_partition(
                &partitioner,
                &mut local.morsel_idxs_values_per_p,
                &mut local.sketch_per_p,
                None,
                track_unmatchable,
            );

            local
                .morsel_idxs_offsets_per_p
                .extend(local.morsel_idxs_values_per_p.iter().map(|vp| vp.len()));
            let sf = SpillFrame::new(payload, spill_ctx).await;
            local.morsels.push((morsel.seq(), sf, hash_keys));
        }
        Ok(())
    }

    /// Whether no row was built, sampled morsels included.
    pub(super) fn is_empty(&self) -> bool {
        self.local_builders
            .iter()
            .all(|b| b.morsels.iter().all(|(_, _, keys)| keys.is_empty()))
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

    pub(super) fn finalize_ordered(
        &mut self,
        params: &EquiJoinParams,
        table: &dyn IdxTable,
    ) -> ProbeState {
        let track_unmatchable = params.emit_unmatched_build();
        let payload_schema = if params.left_is_build.unwrap() {
            &params.left_payload_schema
        } else {
            &params.right_payload_schema
        };

        let num_partitions = self.local_builders[0].sketch_per_p.len();
        let local_builders = &self.local_builders;
        let probe_tables: SparseInitVec<ProbeTable> = SparseInitVec::with_capacity(num_partitions);

        RAYON.scope(|s| {
            for p in 0..num_partitions {
                let probe_tables = &probe_tables;
                s.spawn(move |_| {
                    // TODO: every thread does an identical linearize, we can do a single parallel one.
                    let mut kmerge = BinaryHeap::with_capacity(local_builders.len());
                    let mut cur_idx_per_loc = vec![0; local_builders.len()];

                    // Compute cardinality estimate and total amount of
                    // payload for this partition, and initialize k-way merge.
                    let mut sketch = CardinalitySketch::new();
                    let mut payload_rows = 0;
                    for (l_idx, l) in local_builders.iter().enumerate() {
                        let Some((seq, _, _)) = l.morsels.first() else {
                            continue;
                        };
                        kmerge.push(Priority(Reverse(seq), l_idx));

                        sketch.combine(&l.sketch_per_p[p]);
                        let offsets_len = l.morsel_idxs_offsets_per_p.len();
                        payload_rows +=
                            l.morsel_idxs_offsets_per_p[offsets_len - num_partitions + p];
                    }

                    // Allocate hash table and payload builder.
                    let mut p_table = table.new_empty();
                    p_table.reserve(sketch.estimate() * 5 / 4);
                    let mut p_payload = DataFrameBuilder::new(payload_schema.clone());
                    p_payload.reserve(payload_rows);

                    let mut p_row_positions = Vec::new();
                    if track_unmatchable {
                        p_row_positions.reserve(payload_rows);
                    }

                    // Linearize and build.
                    unsafe {
                        // Row offset of the current morsel in the full build input.
                        let mut morsel_row_offset = 0u64;
                        while let Some(Priority(Reverse(_seq), l_idx)) = kmerge.pop() {
                            let l = local_builders.get_unchecked(l_idx);
                            let idx_in_l = *cur_idx_per_loc.get_unchecked(l_idx);
                            *cur_idx_per_loc.get_unchecked_mut(l_idx) += 1;
                            if let Some((next_seq, _, _)) = l.morsels.get(idx_in_l + 1) {
                                kmerge.push(Priority(Reverse(next_seq), l_idx));
                            }

                            let (_mseq, sf, keys) = l.morsels.get_unchecked(idx_in_l);
                            let payload = sf.get_blocking();
                            let p_morsel_idxs_start =
                                l.morsel_idxs_offsets_per_p[idx_in_l * num_partitions + p];
                            let p_morsel_idxs_stop =
                                l.morsel_idxs_offsets_per_p[(idx_in_l + 1) * num_partitions + p];
                            let p_morsel_idxs = &l.morsel_idxs_values_per_p[p]
                                [p_morsel_idxs_start..p_morsel_idxs_stop];
                            p_table.insert_keys_subset(keys, p_morsel_idxs, track_unmatchable);
                            p_payload.gather_extend(&payload, p_morsel_idxs, ShareStrategy::Never);

                            if track_unmatchable {
                                #[allow(clippy::unnecessary_cast)] // Necessary when IdxSize = u64.
                                p_row_positions.extend(
                                    p_morsel_idxs.iter().map(|i| morsel_row_offset + *i as u64),
                                );
                            }
                            morsel_row_offset += payload.height() as u64;
                        }
                    }

                    probe_tables
                        .try_set(
                            p,
                            ProbeTable {
                                hash_table: p_table,
                                payload: p_payload.freeze(),
                                row_positions: p_row_positions,
                            },
                        )
                        .ok()
                        .unwrap();
                });
            }
        });

        ProbeState {
            table_per_partition: probe_tables.try_assume_init().ok().unwrap(),
            max_seq_sent: MorselSeq::default(),
            sampled_probe_morsels: core::mem::take(&mut self.sampled_probe_morsels),
            unordered_morsel_seq: AtomicU64::new(0),
        }
    }

    pub(super) fn finalize_unordered(
        &mut self,
        params: &EquiJoinParams,
        table: &dyn IdxTable,
        state: &StreamingExecutionState,
    ) -> ProbeState {
        let track_unmatchable = params.emit_unmatched_build();
        let payload_schema = if params.left_is_build.unwrap() {
            &params.left_payload_schema
        } else {
            &params.right_payload_schema
        };

        // To reduce maximum memory usage we want to drop the morsels
        // as soon as they're processed, so we move into Arcs. The drops might
        // also be expensive, so instead of directly dropping we put that on
        // a work queue.
        let morsels_per_local_builder = self
            .local_builders
            .iter_mut()
            .map(|b| Arc::new(core::mem::take(&mut b.morsels)))
            .collect_vec();
        let (morsel_drop_q_send, morsel_drop_q_recv) =
            async_channel::bounded(morsels_per_local_builder.len());
        let num_partitions = self.local_builders[0].sketch_per_p.len();
        let local_builders = &self.local_builders;
        let probe_tables: SparseInitVec<ProbeTable> = SparseInitVec::with_capacity(num_partitions);

        executor::task_scope(state.task_metrics(), |s| {
            // Wrap in outer Arc to move to each thread, performing the
            // expensive clone on that thread.
            let arc_morsels_per_local_builder = Arc::new(morsels_per_local_builder);
            let mut join_handles = Vec::new();
            for p in 0..num_partitions {
                let arc_morsels_per_local_builder = Arc::clone(&arc_morsels_per_local_builder);
                let morsel_drop_q_send = morsel_drop_q_send.clone();
                let morsel_drop_q_recv = morsel_drop_q_recv.clone();
                let probe_tables = &probe_tables;
                join_handles.push(s.spawn_task(TaskPriority::High, async move {
                    // Extract from outer arc and drop outer arc.
                    let morsels_per_local_builder =
                        Arc::unwrap_or_clone(arc_morsels_per_local_builder);

                    // Compute cardinality estimate and total amount of
                    // payload for this partition.
                    let mut sketch = CardinalitySketch::new();
                    let mut payload_rows = 0;
                    for l in local_builders {
                        sketch.combine(&l.sketch_per_p[p]);
                        let offsets_len = l.morsel_idxs_offsets_per_p.len();
                        payload_rows +=
                            l.morsel_idxs_offsets_per_p[offsets_len - num_partitions + p];
                    }

                    // Allocate hash table and payload builder.
                    let mut p_table = table.new_empty();
                    p_table.reserve(sketch.estimate() * 5 / 4);
                    let mut p_payload = DataFrameBuilder::new(payload_schema.clone());
                    p_payload.reserve(payload_rows);

                    // Build.
                    let mut skip_drop_attempt = false;
                    for (l, l_morsels) in local_builders.iter().zip(morsels_per_local_builder) {
                        // Try to help with dropping the processed morsels.
                        if !skip_drop_attempt {
                            drop(morsel_drop_q_recv.try_recv());
                        }

                        for (i, morsel) in l_morsels.iter().enumerate() {
                            let (_mseq, sf, keys) = morsel;
                            let payload = sf.get().await;
                            unsafe {
                                let p_morsel_idxs_start =
                                    l.morsel_idxs_offsets_per_p[i * num_partitions + p];
                                let p_morsel_idxs_stop =
                                    l.morsel_idxs_offsets_per_p[(i + 1) * num_partitions + p];
                                let p_morsel_idxs = &l.morsel_idxs_values_per_p[p]
                                    [p_morsel_idxs_start..p_morsel_idxs_stop];
                                p_table.insert_keys_subset(keys, p_morsel_idxs, track_unmatchable);
                                p_payload.gather_extend(
                                    &payload,
                                    p_morsel_idxs,
                                    ShareStrategy::Never,
                                );
                            }
                        }

                        if let Some(l) = Arc::into_inner(l_morsels) {
                            // If we're the last thread to process this set of morsels we're probably
                            // falling behind the rest, since the drop can be quite expensive we skip
                            // a drop attempt hoping someone else will pick up the slack.
                            drop(morsel_drop_q_send.try_send(l));
                            skip_drop_attempt = true;
                        } else {
                            skip_drop_attempt = false;
                        }
                    }

                    // We're done, help others out by doing drops.
                    drop(morsel_drop_q_send); // So we don't deadlock trying to receive from ourselves.
                    while let Ok(l_morsels) = morsel_drop_q_recv.recv().await {
                        drop(l_morsels);
                    }

                    probe_tables
                        .try_set(
                            p,
                            ProbeTable {
                                hash_table: p_table,
                                payload: p_payload.freeze(),
                                row_positions: Vec::new(),
                            },
                        )
                        .ok()
                        .unwrap();
                }));
            }

            // Drop outer arc after spawning each thread so the inner arcs
            // can get dropped as soon as they're processed. We also have to
            // drop the drop queue sender so we don't deadlock waiting for it
            // to end.
            drop(arc_morsels_per_local_builder);
            drop(morsel_drop_q_send);

            ASYNC.block_in_place_on(async move {
                for handle in join_handles {
                    handle.await;
                }
            });
        });

        ProbeState {
            table_per_partition: probe_tables.try_assume_init().ok().unwrap(),
            max_seq_sent: MorselSeq::default(),
            sampled_probe_morsels: core::mem::take(&mut self.sampled_probe_morsels),
            unordered_morsel_seq: AtomicU64::new(0),
        }
    }
}
