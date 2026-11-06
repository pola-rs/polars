use std::sync::atomic::{AtomicU64, Ordering};

use polars_arrow::array::builder::ShareStrategy;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::schema::Schema;
use polars_defs::join::JoinType;
use polars_expr::idx_table::IdxTable;
use polars_ops::series::coalesce_columns;
use polars_utils::hashing::HashPartitioner;
use polars_utils::itertools::Itertools;
use polars_utils::{IdxSize, format_pl_smallstr};
use rayon::prelude::*;

use super::{EquiJoinParams, select_keys, select_payload};
use crate::expression::StreamExpr;
use crate::morsel::get_ideal_morsel_size;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::joins::{BufferedStream, emit_morsel_size, send_frames};

/// Where one of the fused predicate's input columns is found at probe time.
struct FusedColumn {
    is_left: bool,
    /// Index into that side's payload.
    index: usize,
}

/// A match condition on top of the equi keys, applied to candidate pairs.
pub(super) struct FusedPredicate {
    expr: StreamExpr,
    /// The columns the predicate reads, in the order it was compiled against.
    columns: Vec<FusedColumn>,
}

impl FusedPredicate {
    pub(super) fn new(
        expr: StreamExpr,
        schema: &Schema,
        left_payload_schema: &Schema,
        right_payload_schema: &Schema,
    ) -> PolarsResult<Self> {
        let columns = schema
            .iter_names()
            .map(|name| {
                // Payload schemas are keyed by output name.
                let column = if let Some(index) = left_payload_schema.index_of(name) {
                    FusedColumn {
                        is_left: true,
                        index,
                    }
                } else if let Some(index) = right_payload_schema.index_of(name) {
                    FusedColumn {
                        is_left: false,
                        index,
                    }
                } else {
                    polars_bail!(
                        ColumnNotFound:
                        "fused predicate reads '{name}', which the join does not output"
                    )
                };
                Ok(column)
            })
            .try_collect_vec()?;

        Ok(Self { expr, columns })
    }

    /// Compacts the candidate pairs the predicate accepts to the front of both slices.
    ///
    /// The two match slices describe the same pairs positionally. Returns how many
    /// survived, which the caller truncates its own buffers to.
    async fn retain_matches(
        &self,
        left_is_build: bool,
        build_payload: &DataFrame,
        build_match: &mut [IdxSize],
        probe_payload: &DataFrame,
        probe_match: &mut [IdxSize],
        state: &ExecutionState,
    ) -> PolarsResult<usize> {
        let n = build_match.len();
        assert_eq!(n, probe_match.len());

        // `probe_subset` can hand back far more candidates than the morsel limit, so the
        // gathered predicate inputs are bounded here instead.
        let batch_size = get_ideal_morsel_size();
        let mut kept = 0;
        let mut start = 0;

        while start < n {
            let end = (start + batch_size).min(n);
            let len = end - start;

            let columns = self
                .columns
                .iter()
                .map(|column| {
                    let (payload, idxs) = if column.is_left == left_is_build {
                        (build_payload, &build_match[start..end])
                    } else {
                        (probe_payload, &probe_match[start..end])
                    };
                    // Payload columns already carry their output name.
                    unsafe { payload.columns()[column.index].take_slice_unchecked(idxs) }
                })
                .collect();
            let df = unsafe { DataFrame::new_unchecked(len, columns) };

            let mask = self
                .expr
                .evaluate_preserve_len_broadcast(&df, state)
                .await?;
            let mask = mask.as_materialized_series().bool()?.rechunk();
            let mask = mask.downcast_as_array();

            // A null is not a match.
            let keep = match mask.validity() {
                Some(validity) => mask.values() & validity,
                None => mask.values().clone(),
            };

            match keep.set_bits() {
                0 => {},
                // The batch survives whole, so it moves as one block.
                set if set == len => {
                    if kept != start {
                        build_match.copy_within(start..end, kept);
                        probe_match.copy_within(start..end, kept);
                    }
                    kept += len;
                },
                // SAFETY: We are in bounds
                _ => {
                    for i in keep.true_idx_iter() {
                        debug_assert!(start + i < end);
                        debug_assert!(kept <= start + i);
                        unsafe {
                            let build = *build_match.get_unchecked(start + i);
                            let probe = *probe_match.get_unchecked(start + i);
                            *build_match.get_unchecked_mut(kept) = build;
                            *probe_match.get_unchecked_mut(kept) = probe;
                        }
                        kept += 1;
                    }
                },
            }

            start = end;
        }

        Ok(kept)
    }
}

/// Rechunks `payload` the first time rows are gathered from it.
fn rechunk_once(payload: &mut DataFrame, rechunked: &mut bool) {
    if !*rechunked {
        payload.rechunk_mut();
        *rechunked = true;
    }
}

/// Fixes names and does coalescing of columns post-join.
fn postprocess_join(df: DataFrame, params: &EquiJoinParams) -> DataFrame {
    if params.args.how == JoinType::Full && params.args.should_coalesce() {
        // TODO: don't do string-based column lookups for each dataframe, pre-compute coalesce indices.
        let new_cols = df
            .columns()
            .iter()
            .filter_map(|c| {
                if let Some(key_idx) = params.left_key_schema.index_of(c.name()) {
                    let other = df
                        .column(&format_pl_smallstr!("__POLARS_COALESCE_KEYCOL_{key_idx}"))
                        .unwrap();
                    return Some(coalesce_columns(&[c.clone(), other.clone()]).unwrap());
                }

                if c.name().starts_with("__POLARS_COALESCE_KEYCOL") {
                    return None;
                }

                Some(c.clone())
            })
            .collect();

        unsafe { DataFrame::new_unchecked(df.height(), new_cols) }
    } else {
        df
    }
}

pub(super) struct ProbeTable {
    pub(super) hash_table: Box<dyn IdxTable>,
    pub(super) payload: DataFrame,
    // Position of each payload row in the full build input. Only filled for
    // ordered joins that emit unmatched build rows.
    pub(super) row_positions: Vec<u64>,
}

pub(super) struct ProbeState {
    pub(super) table_per_partition: Vec<ProbeTable>,
    pub(super) max_seq_sent: MorselSeq,
    pub(super) sampled_probe_morsels: BufferedStream,

    // For unordered joins we relabel output morsels to speed up the linearizer.
    pub(super) unordered_morsel_seq: AtomicU64,
}

impl ProbeState {
    /// Returns the max morsel sequence sent.
    pub(super) async fn partition_and_probe(
        mut recv: PortReceiver,
        mut send: PortSender,
        partitions: &[ProbeTable],
        unordered_morsel_seq: &AtomicU64,
        partitioner: HashPartitioner,
        params: &EquiJoinParams,
        state: &StreamingExecutionState,
    ) -> PolarsResult<MorselSeq> {
        // TODO: shuffle after partitioning and keep probe tables thread-local.
        let mut partition_idxs = vec![Vec::new(); partitioner.num_partitions()];
        let mut probe_partitions = Vec::new();
        let mut materialized_idxsize_range = Vec::new();
        let mut table_match = Vec::new();
        let mut probe_match = Vec::new();
        let mut max_seq = MorselSeq::default();

        let probe_limit = get_ideal_morsel_size() as IdxSize;
        let mark_matches = params.emit_unmatched_build();
        let emit_unmatched = params.emit_unmatched_probe();
        assert!(params.fused_predicate.is_none() || (!mark_matches && !emit_unmatched));

        let (key_selectors, payload_selector, build_payload_schema, probe_payload_schema);
        if params.left_is_build.unwrap() {
            key_selectors = &params.right_key_selectors;
            payload_selector = &params.right_payload_select;
            build_payload_schema = &params.left_payload_schema;
            probe_payload_schema = &params.right_payload_schema;
        } else {
            key_selectors = &params.left_key_selectors;
            payload_selector = &params.left_payload_select;
            build_payload_schema = &params.right_payload_schema;
            probe_payload_schema = &params.left_payload_schema;
        };

        let mut build_out = DataFrameBuilder::new(build_payload_schema.clone());
        let mut probe_out = DataFrameBuilder::new(probe_payload_schema.clone());

        // A simple estimate used to size reserves.
        let mut selectivity_estimate = 1.0;
        let mut selectivity_estimate_confidence = 0.0;

        while let Ok(morsel) = recv.recv().await {
            // Compute hashed keys and payload.
            let (sf, in_seq, src_token, wait_token) = morsel.into_inner();
            let df = sf.into_df().await;

            let df_height = df.height();
            if df_height == 0 {
                continue;
            }

            let hash_keys =
                select_keys(&df, key_selectors, params, &state.in_memory_exec_state).await?;
            let mut payload = select_payload(df, payload_selector);
            let mut payload_rechunked = false;
            let mut total_matches = 0;

            // Use selectivity estimate to reserve for morsel builders.
            let max_match_per_key_est = (selectivity_estimate * 1.2) as usize + 16;
            let out_est_size = ((selectivity_estimate * 1.2 * df_height as f64) as usize)
                .min(probe_limit as usize);
            build_out.reserve(out_est_size + max_match_per_key_est);

            unsafe {
                let mut new_morsel =
                    |build: &mut DataFrameBuilder, probe: &mut DataFrameBuilder| {
                        let mut build_df = build.freeze_reset();
                        let mut probe_df = probe.freeze_reset();
                        let out_df = if params.left_is_build.unwrap() {
                            build_df.hstack_mut_unchecked(probe_df.columns());
                            build_df
                        } else {
                            probe_df.hstack_mut_unchecked(build_df.columns());
                            probe_df
                        };
                        let out_df = postprocess_join(out_df, params);
                        let out_seq = if params.preserve_order_probe {
                            in_seq
                        } else {
                            MorselSeq::new(unordered_morsel_seq.fetch_add(1, Ordering::Relaxed))
                        };
                        max_seq = out_seq;
                        Morsel::new_unregistered(out_df, out_seq, src_token.clone())
                    };

                if params.preserve_order_probe {
                    // To preserve the order we can't do bulk probes per partition and must follow
                    // the order of the probe morsel. We can still group probes that are
                    // consecutively on the same partition.
                    probe_partitions.clear();
                    hash_keys.gen_partitions(&partitioner, &mut probe_partitions, emit_unmatched);

                    let mut probe_group_start = 0;
                    while probe_group_start < probe_partitions.len() {
                        let p_idx = probe_partitions[probe_group_start];
                        let mut probe_group_end = probe_group_start + 1;
                        while probe_partitions.get(probe_group_end) == Some(&p_idx) {
                            probe_group_end += 1;
                        }
                        let Some(p) = partitions.get(p_idx as usize) else {
                            probe_group_start = probe_group_end;
                            continue;
                        };

                        materialized_idxsize_range.extend(
                            materialized_idxsize_range.len() as IdxSize..probe_group_end as IdxSize,
                        );

                        while probe_group_start < probe_group_end {
                            let matches_before_limit = probe_limit - probe_match.len() as IdxSize;
                            table_match.clear();
                            probe_group_start += p.hash_table.probe_subset(
                                &hash_keys,
                                &materialized_idxsize_range[probe_group_start..probe_group_end],
                                &mut table_match,
                                &mut probe_match,
                                mark_matches,
                                emit_unmatched,
                                matches_before_limit,
                            ) as usize;

                            if emit_unmatched {
                                build_out.opt_gather_extend(
                                    &p.payload,
                                    &table_match,
                                    ShareStrategy::Always,
                                );
                            } else {
                                build_out.gather_extend(
                                    &p.payload,
                                    &table_match,
                                    ShareStrategy::Always,
                                );
                            };

                            if probe_match.len() >= probe_limit as usize
                                || probe_group_start == probe_partitions.len()
                            {
                                rechunk_once(&mut payload, &mut payload_rechunked);
                                probe_out.gather_extend(
                                    &payload,
                                    &probe_match,
                                    ShareStrategy::Always,
                                );
                                let out_len = probe_match.len();
                                probe_match.clear();
                                let out_morsel = new_morsel(&mut build_out, &mut probe_out);
                                if send.send(out_morsel).await.is_err() {
                                    return Ok(max_seq);
                                }
                                if probe_group_end != probe_partitions.len() {
                                    // We had enough matches to need a mid-partition flush, let's assume there are a lot of
                                    // matches and just do a large reserve.
                                    let old_est = probe_limit as usize + max_match_per_key_est;
                                    build_out.reserve(old_est.max(out_len + 16));
                                }
                            }
                        }
                    }
                } else {
                    // Partition and probe the tables.
                    for p in partition_idxs.iter_mut() {
                        p.clear();
                    }
                    hash_keys.gen_idxs_per_partition(
                        &partitioner,
                        &mut partition_idxs,
                        &mut [],
                        None,
                        emit_unmatched,
                    );

                    for (p, idxs_in_p) in partitions.iter().zip(&partition_idxs) {
                        let mut offset = 0;
                        while offset < idxs_in_p.len() {
                            let matches_before_limit = probe_limit - probe_match.len() as IdxSize;
                            let probe_start = probe_match.len();
                            table_match.clear();
                            offset += p.hash_table.probe_subset(
                                &hash_keys,
                                &idxs_in_p[offset..],
                                &mut table_match,
                                &mut probe_match,
                                mark_matches,
                                emit_unmatched,
                                matches_before_limit,
                            ) as usize;

                            if let Some(fused_predicate) = &params.fused_predicate
                                && !table_match.is_empty()
                            {
                                rechunk_once(&mut payload, &mut payload_rechunked);
                                let kept = fused_predicate
                                    .retain_matches(
                                        params.left_is_build.unwrap(),
                                        &p.payload,
                                        &mut table_match,
                                        &payload,
                                        &mut probe_match[probe_start..],
                                        &state.in_memory_exec_state,
                                    )
                                    .await?;
                                table_match.truncate(kept);
                                probe_match.truncate(probe_start + kept);
                            }

                            if table_match.is_empty() {
                                continue;
                            }
                            total_matches += table_match.len();

                            if emit_unmatched {
                                build_out.opt_gather_extend(
                                    &p.payload,
                                    &table_match,
                                    ShareStrategy::Always,
                                );
                            } else {
                                build_out.gather_extend(
                                    &p.payload,
                                    &table_match,
                                    ShareStrategy::Always,
                                );
                            };

                            if probe_match.len() >= probe_limit as usize {
                                rechunk_once(&mut payload, &mut payload_rechunked);
                                probe_out.gather_extend(
                                    &payload,
                                    &probe_match,
                                    ShareStrategy::Always,
                                );
                                let out_len = probe_match.len();
                                probe_match.clear();
                                let out_morsel = new_morsel(&mut build_out, &mut probe_out);
                                if send.send(out_morsel).await.is_err() {
                                    return Ok(max_seq);
                                }
                                // We had enough matches to need a mid-partition flush, let's assume there are a lot of
                                // matches and just do a large reserve.
                                let old_est = probe_limit as usize + max_match_per_key_est;
                                build_out.reserve(old_est.max(out_len + 16));
                            }
                        }
                    }
                }

                if !probe_match.is_empty() {
                    rechunk_once(&mut payload, &mut payload_rechunked);
                    probe_out.gather_extend(&payload, &probe_match, ShareStrategy::Always);
                    probe_match.clear();
                    let out_morsel = new_morsel(&mut build_out, &mut probe_out);
                    if send.send(out_morsel).await.is_err() {
                        return Ok(max_seq);
                    }
                }
            }

            drop(wait_token);

            // Move selectivity estimate a bit towards latest value. Allows rapid changes at first.
            // TODO: implement something more re-usable and robust.
            selectivity_estimate = selectivity_estimate_confidence * selectivity_estimate
                + (1.0 - selectivity_estimate_confidence)
                    * (total_matches as f64 / df_height as f64);
            selectivity_estimate_confidence = (selectivity_estimate_confidence + 0.1).min(0.8);
        }

        Ok(max_seq)
    }

    pub(super) fn ordered_unmatched(&mut self, params: &EquiJoinParams) -> DataFrame {
        // TODO: parallelize this operator.

        let build_payload_schema = if params.left_is_build.unwrap() {
            &params.left_payload_schema
        } else {
            &params.right_payload_schema
        };

        let mut unmarked_idxs = Vec::new();
        let mut row_positions = Vec::new();

        unsafe {
            let mut build_out = DataFrameBuilder::new(build_payload_schema.clone());
            for p in &self.table_per_partition {
                p.hash_table
                    .unmarked_keys(&mut unmarked_idxs, 0, IdxSize::MAX);
                row_positions.extend(
                    unmarked_idxs
                        .iter()
                        .map(|i| *p.row_positions.get_unchecked(*i as usize)),
                );
                build_out.gather_extend(
                    &p.payload,
                    &unmarked_idxs,
                    ShareStrategy::Never, // Don't keep entire table alive for unmatched indices.
                );
            }

            let mut perm: Vec<IdxSize> = (0..row_positions.len() as IdxSize).collect();
            perm.sort_unstable_by_key(|i| *row_positions.get_unchecked(*i as usize));

            let mut build_df = build_out.freeze().take_slice_unchecked(&perm);
            let out_df = if params.left_is_build.unwrap() {
                let probe_df =
                    DataFrame::full_null(&params.right_payload_schema, build_df.height());
                build_df.hstack_mut_unchecked(probe_df.columns());
                build_df
            } else {
                let mut probe_df =
                    DataFrame::full_null(&params.left_payload_schema, build_df.height());
                probe_df.hstack_mut_unchecked(build_df.columns());
                probe_df
            };
            postprocess_join(out_df, params)
        }
    }
}

impl Drop for ProbeState {
    fn drop(&mut self) {
        RAYON.install(|| {
            // Parallel drop as the state might be quite big.
            self.table_per_partition.par_drain(..).for_each(drop);
        })
    }
}

pub(super) struct EmitUnmatchedState {
    pub(super) partitions: Vec<ProbeTable>,
    pub(super) active_partition_idx: usize,
    pub(super) offset_in_active_p: usize,
    pub(super) morsel_seq: MorselSeq,
}

impl EmitUnmatchedState {
    pub(super) async fn emit_unmatched(
        &mut self,
        send: PortSender,
        params: &EquiJoinParams,
        num_pipelines: usize,
    ) -> PolarsResult<()> {
        let total_len: usize = self.partitions.iter().map(|p| p.payload.height()).sum();
        let morsel_size = emit_morsel_size(total_len, num_pipelines);

        let mut unmarked_idxs = Vec::new();
        let partitions = &self.partitions;
        let active_partition_idx = &mut self.active_partition_idx;
        let offset_in_active_p = &mut self.offset_in_active_p;
        send_frames(send, &mut self.morsel_seq, || {
            while let Some(p) = partitions.get(*active_partition_idx) {
                // Generate a chunk of unmarked key indices.
                *offset_in_active_p += p.hash_table.unmarked_keys(
                    &mut unmarked_idxs,
                    *offset_in_active_p as IdxSize,
                    morsel_size as IdxSize,
                ) as usize;
                if unmarked_idxs.is_empty() {
                    *active_partition_idx += 1;
                    *offset_in_active_p = 0;
                    continue;
                }

                // Gather and create full-null counterpart.
                let out_df = unsafe {
                    let mut build_df = p.payload.take_slice_unchecked_impl(&unmarked_idxs, false);
                    let len = build_df.height();
                    if params.left_is_build.unwrap() {
                        let probe_df = DataFrame::full_null(&params.right_payload_schema, len);
                        build_df.hstack_mut_unchecked(probe_df.columns());
                        build_df
                    } else {
                        let mut probe_df = DataFrame::full_null(&params.left_payload_schema, len);
                        probe_df.hstack_mut_unchecked(build_df.columns());
                        probe_df
                    }
                };
                return Some(postprocess_join(out_df, params));
            }
            None
        })
        .await
    }
}
