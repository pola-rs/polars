use std::sync::atomic::{AtomicU64, Ordering};

use polars_arrow::array::builder::ShareStrategy;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::schema::Schema;
use polars_defs::join::JoinType;
use polars_expr::idx_table::IdxTable;
use polars_ooc::MemoryGrant;
use polars_ops::series::coalesce_columns;
use polars_utils::hashing::HashPartitioner;
use polars_utils::itertools::Itertools;
use polars_utils::{IdxSize, format_pl_smallstr};
use rayon::prelude::*;

use super::{EquiJoinParams, JoinPartition, select_keys, select_payload};
use crate::expression::StreamExpr;
use crate::hash_shuffle::frame_with_keys::FrameWithKeys;
use crate::hash_shuffle::split::{NullRows, rows_or_all, split};
use crate::hash_shuffle::store::StoredPartition;
use crate::hash_shuffle::writer::PartitionedWriter;
use crate::morsel::get_ideal_morsel_size;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::joins::BufferedStream;
use crate::nodes::joins::tuning::BuildStats;

/// Where one of the fused predicate's input columns is found at probe time.
struct FusedColumn {
    is_left: bool,
    /// Index into that side's payload.
    index: usize,
}

/// A match condition on top of the equi keys, applied to candidate pairs.
pub struct FusedPredicate {
    expr: StreamExpr,
    /// The columns the predicate reads, in the order it was compiled against.
    columns: Vec<FusedColumn>,
}

impl FusedPredicate {
    pub fn new(
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

/// `df`, rows of the left side or else the right side, with null columns for the other side.
pub fn with_nulls_for_other_side(
    df: DataFrame,
    df_is_left: bool,
    params: &EquiJoinParams,
) -> DataFrame {
    let nulls = DataFrame::full_null(params.payload_schema(!df_is_left), df.height());
    let (mut left, right) = if df_is_left { (df, nulls) } else { (nulls, df) };
    unsafe { left.hstack_mut_unchecked(right.columns()) };
    postprocess_join(left, params)
}

pub struct ProbeTable {
    pub hash_table: Box<dyn IdxTable>,
    pub payload: DataFrame,
    // Position of each payload row in the full build input. Only filled for
    // ordered joins that emit unmatched build rows.
    pub row_positions: Vec<u64>,
}

/// The output of a probe until it is sent: the build rows of the matches, the probe rows of the
/// matches in earlier probe frames, and the indices of the matched rows of the current one.
pub struct ProbeOutput {
    build: DataFrameBuilder,
    probe: DataFrameBuilder,
    table_match: Vec<IdxSize>,
    probe_match: Vec<IdxSize>,
    unmatched_probe: Vec<IdxSize>,
    left_is_build: bool,
    mark_matches: bool,
    emit_unmatched: bool,
    /// The number of rows of a full output.
    limit: usize,
    /// The build columns without rows, to gather nulls from.
    no_build_rows: DataFrame,
    null_idxs: Vec<IdxSize>,
}

impl ProbeOutput {
    pub fn new(params: &EquiJoinParams, left_is_build: bool) -> Self {
        let build_schema = params.payload_schema(left_is_build);
        Self {
            build: DataFrameBuilder::new(build_schema.clone()),
            probe: DataFrameBuilder::new(params.payload_schema(!left_is_build).clone()),
            table_match: Vec::new(),
            probe_match: Vec::new(),
            unmatched_probe: Vec::new(),
            left_is_build,
            mark_matches: params.emit_unmatched(left_is_build),
            emit_unmatched: params.emit_unmatched(!left_is_build),
            limit: get_ideal_morsel_size(),
            no_build_rows: DataFrame::empty_with_schema(build_schema),
            null_idxs: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.build.len()
    }

    pub fn is_empty(&self) -> bool {
        self.build.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.len() >= self.limit
    }

    /// Probes `table` with the rows `rows` of `frame` from `offset` on, until all are probed or
    /// the output is full, and moves `offset` past the probed rows. The columns of `frame` must be
    /// single chunks.
    pub async fn probe(
        &mut self,
        table: &ProbeTable,
        frame: &FrameWithKeys,
        rows: &[IdxSize],
        offset: &mut usize,
        params: &EquiJoinParams,
        state: &ExecutionState,
    ) -> PolarsResult<()> {
        let limit = self.limit.saturating_sub(self.len()).max(1) as IdxSize;
        let probe_start = self.probe_match.len();
        let probed_start = *offset;
        self.table_match.clear();
        // With a fused predicate, a pair only matches once the predicate holds, so the unmatched
        // rows are found after evaluating it.
        let fused = params.fused_predicate.is_some();
        *offset += unsafe {
            table.hash_table.probe_subset(
                &frame.keys,
                &rows[*offset..],
                &mut self.table_match,
                &mut self.probe_match,
                self.mark_matches && !fused,
                self.emit_unmatched && !fused,
                limit,
            )
        } as usize;

        if let Some(fused_predicate) = &params.fused_predicate {
            if !self.table_match.is_empty() {
                let kept = fused_predicate
                    .retain_matches(
                        self.left_is_build,
                        &table.payload,
                        &mut self.table_match,
                        &frame.frame,
                        &mut self.probe_match[probe_start..],
                        state,
                    )
                    .await?;
                self.table_match.truncate(kept);
                self.probe_match.truncate(probe_start + kept);
            }
            if self.mark_matches {
                table.hash_table.mark_rows(&self.table_match);
            }
            if self.emit_unmatched {
                // The kept pairs follow the order of the probed rows.
                let mut kept = self.probe_match[probe_start..].iter().peekable();
                self.unmatched_probe.clear();
                for row in &rows[probed_start..*offset] {
                    if kept.peek() == Some(&row) {
                        while kept.next_if_eq(&row).is_some() {}
                    } else {
                        self.unmatched_probe.push(*row);
                    }
                }
                self.table_match
                    .extend(std::iter::repeat_n(IdxSize::MAX, self.unmatched_probe.len()));
                self.probe_match.extend_from_slice(&self.unmatched_probe);
            }
        }

        if self.emit_unmatched {
            self.build
                .opt_gather_extend(&table.payload, &self.table_match, ShareStrategy::Always);
        } else {
            unsafe {
                self.build
                    .gather_extend(&table.payload, &self.table_match, ShareStrategy::Always)
            };
        }
        Ok(())
    }

    /// Adds the probe rows `rows` of the current probe frame without a match.
    pub fn push_unmatched(&mut self, rows: &[IdxSize]) {
        if self.null_idxs.len() < rows.len() {
            self.null_idxs.resize(rows.len(), IdxSize::MAX);
        }
        self.build.opt_gather_extend(
            &self.no_build_rows,
            &self.null_idxs[..rows.len()],
            ShareStrategy::Always,
        );
        self.probe_match.extend_from_slice(rows);
    }

    /// Gathers the probe rows of the matches from `payload`, the current probe frame, whose
    /// columns must be single chunks.
    pub fn gather_probe(&mut self, payload: &DataFrame) {
        unsafe {
            self.probe
                .gather_extend(payload, &self.probe_match, ShareStrategy::Always)
        };
        self.probe_match.clear();
    }

    /// Takes the output, once the probe rows are gathered.
    pub fn take(&mut self, params: &EquiJoinParams) -> DataFrame {
        debug_assert!(self.probe_match.is_empty());
        let build = self.build.freeze_reset();
        let probe = self.probe.freeze_reset();
        let (mut left, right) = if self.left_is_build {
            (build, probe)
        } else {
            (probe, build)
        };
        unsafe { left.hstack_mut_unchecked(right.columns()) };
        postprocess_join(left, params)
    }
}

/// The partitions while the probe input is received, shared by the probe tasks.
pub struct ProbePartitions {
    pub partitions: Vec<JoinPartition>,
    pub partitioner: HashPartitioner,
    // For unordered joins we relabel output morsels to speed up the linearizer.
    pub unordered_morsel_seq: AtomicU64,
}

pub struct ProbeState {
    pub partitions: ProbePartitions,
    /// One per pipeline, writing the probe rows of the stored partitions. Empty if no partition is
    /// stored.
    pub writers: Vec<PartitionedWriter>,
    /// The build rows with a null key, if they are emitted.
    pub null_build_rows: Option<StoredPartition>,
    pub stats: BuildStats,
    pub grant: Option<MemoryGrant>,
    pub max_seq_sent: MorselSeq,
    pub sampled_probe_morsels: BufferedStream,
}

impl ProbePartitions {
    /// Probes the tables in memory and writes the rows of the stored partitions with `writer`.
    /// Returns the max morsel sequence sent.
    pub async fn partition_and_probe(
        &self,
        mut recv: PortReceiver,
        mut send: PortSender,
        mut writer: Option<&mut PartitionedWriter>,
        params: &EquiJoinParams,
        state: &StreamingExecutionState,
    ) -> PolarsResult<MorselSeq> {
        // TODO: shuffle after partitioning and keep probe tables thread-local.
        let partitions = &self.partitions;
        let partitioner = &self.partitioner;
        let mut probe_partitions = Vec::new();
        let mut materialized_idxsize_range = Vec::new();
        let mut all_rows = Vec::new();
        let mut max_seq = MorselSeq::default();

        let left_is_build = params.left_is_build.unwrap();
        let mut out = ProbeOutput::new(params, left_is_build);
        let probe_limit = out.limit;
        let emit_unmatched = out.emit_unmatched;
        assert!(params.fused_predicate.is_none() || !params.preserve_order_probe);

        let (key_selectors, payload_selector) = if left_is_build {
            (&params.right_key_selectors, &params.right_payload_select)
        } else {
            (&params.left_key_selectors, &params.left_payload_select)
        };

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
            let mut frame = FrameWithKeys {
                frame: select_payload(df, payload_selector),
                keys: hash_keys,
                mask: None,
            };
            let mut payload_rechunked = false;
            let mut total_matches = 0;

            // Use selectivity estimate to reserve for morsel builders.
            let max_match_per_key_est = (selectivity_estimate * 1.2) as usize + 16;
            let out_est_size =
                ((selectivity_estimate * 1.2 * df_height as f64) as usize).min(probe_limit);
            out.build.reserve(out_est_size + max_match_per_key_est);

            let mut new_morsel = |out: &mut ProbeOutput| {
                let out_df = out.take(params);
                let out_seq = if params.preserve_order_probe {
                    in_seq
                } else {
                    MorselSeq::new(self.unordered_morsel_seq.fetch_add(1, Ordering::Relaxed))
                };
                max_seq = out_seq;
                Morsel::new_unregistered(out_df, out_seq, src_token.clone())
            };

            if params.preserve_order_probe {
                // To preserve the order we can't do bulk probes per partition and must follow
                // the order of the probe morsel. We can still group probes that are
                // consecutively on the same partition.
                probe_partitions.clear();
                frame
                    .keys
                    .gen_partitions(partitioner, &mut probe_partitions, emit_unmatched);

                let mut probe_group_start = 0;
                while probe_group_start < probe_partitions.len() {
                    let p_idx = probe_partitions[probe_group_start];
                    let mut probe_group_end = probe_group_start + 1;
                    while probe_partitions.get(probe_group_end) == Some(&p_idx) {
                        probe_group_end += 1;
                    }
                    let Some(p) = partitions
                        .get(p_idx as usize)
                        .and_then(|p| p.table.as_ref())
                    else {
                        probe_group_start = probe_group_end;
                        continue;
                    };

                    materialized_idxsize_range.extend(
                        materialized_idxsize_range.len() as IdxSize..probe_group_end as IdxSize,
                    );

                    while probe_group_start < probe_group_end {
                        out.probe(
                            p,
                            &frame,
                            &materialized_idxsize_range[..probe_group_end],
                            &mut probe_group_start,
                            params,
                            &state.in_memory_exec_state,
                        )
                        .await?;

                        if out.is_full() || probe_group_start == probe_partitions.len() {
                            rechunk_once(&mut frame.frame, &mut payload_rechunked);
                            out.gather_probe(&frame.frame);
                            let out_len = out.len();
                            let out_morsel = new_morsel(&mut out);
                            if send.send(out_morsel).await.is_err() {
                                return Ok(max_seq);
                            }
                            if probe_group_end != probe_partitions.len() {
                                // We had enough matches to need a mid-partition flush, let's assume there are a lot of
                                // matches and just do a large reserve.
                                let old_est = probe_limit + max_match_per_key_est;
                                out.build.reserve(old_est.max(out_len + 16));
                            }
                        }
                    }
                }

                if !out.probe_match.is_empty() {
                    rechunk_once(&mut frame.frame, &mut payload_rechunked);
                    out.gather_probe(&frame.frame);
                    if send.send(new_morsel(&mut out)).await.is_err() {
                        return Ok(max_seq);
                    }
                }
            } else {
                frame.frame.rechunk_mut();
                let split = split(&frame.keys, None, partitioner, NullRows::Separate);
                if emit_unmatched && let Some(split) = &split {
                    out.push_unmatched(split.partition(partitioner.num_partitions()));
                }

                for p in partitions {
                    let Some(table) = &p.table else {
                        continue;
                    };
                    let rows = split
                        .as_ref()
                        .map(|split| split.partition(p.stored.index()));
                    let rows = rows_or_all(rows, &mut all_rows, df_height);
                    let mut offset = 0;
                    while offset < rows.len() {
                        if out.is_full() {
                            out.gather_probe(&frame.frame);
                            let out_len = out.len();
                            if send.send(new_morsel(&mut out)).await.is_err() {
                                return Ok(max_seq);
                            }
                            // We had enough matches to need a mid-partition flush, let's assume there are a lot of
                            // matches and just do a large reserve.
                            let old_est = probe_limit + max_match_per_key_est;
                            out.build.reserve(old_est.max(out_len + 16));
                        }
                        let len_before = out.len();
                        out.probe(
                            table,
                            &frame,
                            rows,
                            &mut offset,
                            params,
                            &state.in_memory_exec_state,
                        )
                        .await?;
                        total_matches += out.len() - len_before;
                    }
                }

                if !out.is_empty() {
                    out.gather_probe(&frame.frame);
                    if send.send(new_morsel(&mut out)).await.is_err() {
                        return Ok(max_seq);
                    }
                }

                if let Some(writer) = writer.as_deref_mut() {
                    let stored = partitions.iter().filter(|p| p.table.is_none());
                    let stored = stored.map(|p| p.stored.index());
                    unsafe { writer.write(&frame, split.as_ref(), stored).await };
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
}

impl ProbeState {
    pub fn ordered_unmatched(&mut self, params: &EquiJoinParams) -> DataFrame {
        // TODO: parallelize this operator.

        let left_is_build = params.left_is_build.unwrap();
        let mut unmarked_idxs = Vec::new();
        let mut row_positions = Vec::new();

        unsafe {
            let mut build_out = DataFrameBuilder::new(params.payload_schema(left_is_build).clone());
            for p in self
                .partitions
                .partitions
                .iter()
                .filter_map(|p| p.table.as_ref())
            {
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

            let build_df = build_out.freeze().take_slice_unchecked(&perm);
            with_nulls_for_other_side(build_df, left_is_build, params)
        }
    }
}

impl Drop for ProbeState {
    fn drop(&mut self) {
        RAYON.install(|| {
            // Parallel drop as the state might be quite big.
            self.partitions.partitions.par_drain(..).for_each(drop);
        })
    }
}
