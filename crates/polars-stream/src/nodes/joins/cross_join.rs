use std::sync::Arc;

use polars_arrow::array::builder::ShareStrategy;
use polars_async::executor::TaskMetricAggregator;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::schema::Schema;
use polars_defs::join::{JoinArgs, JoinBuildSide, MaintainOrderJoin};
use polars_error::{polars_ensure, polars_warn};
use polars_mem_engine::column_to_mask;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::slice_enum::Slice;
use polars_utils::{IdxSize, format_pl_smallstr};

use crate::expression::StreamExpr;
use crate::morsel::get_ideal_morsel_size;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::in_memory_sink::InMemorySinkNode;

pub struct CrossJoinNode {
    left_is_build: bool,
    left_input_schema: Arc<Schema>,
    right_input_schema: Arc<Schema>,
    right_rename: Vec<Option<PlSmallStr>>,
    fused_predicate: Option<CrossJoinPredicate>,
    state: CrossJoinState,
}

pub(crate) struct CrossJoinColumn {
    pub is_left: bool,
    pub index: usize,
    pub name: PlSmallStr,
}

pub(crate) struct CrossJoinPredicate {
    pub expr: StreamExpr,
    pub operands: Vec<CrossJoinColumn>,
    pub output: Vec<CrossJoinColumn>,
}

struct CrossJoinChunk<'a> {
    build: &'a DataFrame,
    probe: &'a DataFrame,
    build_start: usize,
    build_len: usize,
    probe_start: usize,
    probe_len: usize,
    left_is_build: bool,
}

#[derive(Default)]
struct CrossJoinIndices {
    build: Vec<IdxSize>,
    probe: Vec<IdxSize>,
}

impl CrossJoinChunk<'_> {
    fn gather_columns(
        &self,
        columns: &[CrossJoinColumn],
        indices: &CrossJoinIndices,
        len: usize,
        candidates: bool,
    ) -> DataFrame {
        let columns = columns
            .iter()
            .map(|col| {
                let mut output = if col.is_left == self.left_is_build {
                    let build =
                        self.build.columns()[col.index].slice((self.build_start, self.build_len));
                    if candidates && self.probe_len == 1 {
                        build
                    } else {
                        // SAFETY: The indices are relative to this build slice.
                        unsafe { build.take_slice_unchecked(&indices.build) }
                    }
                } else {
                    let probe = &self.probe.columns()[col.index];
                    if self.probe_len == 1 {
                        probe.new_from_index(self.probe_start, len)
                    } else {
                        let probe = probe.slice((self.probe_start, self.probe_len));
                        // SAFETY: The indices are relative to this probe slice.
                        unsafe { probe.take_slice_unchecked(&indices.probe) }
                    }
                };
                output.rename(col.name.clone());
                output
            })
            .collect();
        // SAFETY: Cross-join output names are unique and all columns have the same length.
        unsafe { DataFrame::new_unchecked(len, columns) }
    }

    async fn filter(
        &self,
        predicate: &CrossJoinPredicate,
        indices: &mut CrossJoinIndices,
        state: &ExecutionState,
    ) -> PolarsResult<Option<DataFrame>> {
        let len = self.build_len * self.probe_len;
        indices.build.clear();
        indices.probe.clear();
        if self.probe_len > 1 {
            for probe_index in 0..self.probe_len as IdxSize {
                indices.build.extend(0..self.build_len as IdxSize);
                indices
                    .probe
                    .extend(std::iter::repeat_n(probe_index, self.build_len));
            }
        }
        let candidates = self.gather_columns(&predicate.operands, indices, len, true);
        let mask = predicate.expr.evaluate(&candidates, state).await?;
        let mask = column_to_mask(&mask, len)?;
        if mask.len() == 1 {
            return Ok((mask.get(0) == Some(true))
                .then(|| self.gather_columns(&predicate.output, indices, len, true)));
        }
        polars_ensure!(
            mask.len() == len,
            ShapeMismatch: "filter's length: {} differs from that of the series: {}",
            mask.len(), len
        );
        let mask = mask.rechunk();
        let mask = mask.downcast_as_array();
        let keep = match mask.validity() {
            Some(validity) => mask.values() & validity,
            None => mask.values().clone(),
        };
        let kept = keep.set_bits();
        if kept == 0 {
            return Ok(None);
        }
        if kept == len {
            return Ok(Some(self.gather_columns(
                &predicate.output,
                indices,
                len,
                true,
            )));
        }
        indices.build.clear();
        indices.probe.clear();
        for index in keep.true_idx_iter() {
            indices.build.push((index % self.build_len) as IdxSize);
            indices.probe.push((index / self.build_len) as IdxSize);
        }
        Ok(Some(self.gather_columns(
            &predicate.output,
            indices,
            kept,
            false,
        )))
    }
}

impl CrossJoinNode {
    pub(crate) fn new(
        left_input_schema: Arc<Schema>,
        right_input_schema: Arc<Schema>,
        args: &JoinArgs,
        fused_predicate: Option<CrossJoinPredicate>,
        task_metrics: Option<Arc<TaskMetricAggregator>>,
    ) -> Self {
        let left_is_build = match args.maintain_order {
            MaintainOrderJoin::None => match args.build_side {
                // TODO: size estimation.
                None | Some(JoinBuildSide::PreferLeft) | Some(JoinBuildSide::ForceLeft) => true,
                Some(JoinBuildSide::PreferRight) | Some(JoinBuildSide::ForceRight) => false,
            },
            MaintainOrderJoin::Left | MaintainOrderJoin::LeftRight => {
                if args.build_side == Some(JoinBuildSide::ForceLeft) {
                    polars_warn!("can't force left build-side with left-maintaining cross-join");
                }
                false
            },
            MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft => {
                if args.build_side == Some(JoinBuildSide::ForceRight) {
                    polars_warn!("can't force right build-side with right-maintaining cross-join");
                }
                true
            },
        };
        let build_input_schema = if left_is_build {
            &left_input_schema
        } else {
            &right_input_schema
        };
        let sink_node = InMemorySinkNode::new(build_input_schema.clone(), task_metrics);
        let right_rename = if fused_predicate.is_some() {
            Vec::new()
        } else {
            right_input_schema
                .iter_names()
                .map(|rname| {
                    if left_input_schema.contains(rname) {
                        Some(format_pl_smallstr!("{}{}", rname, args.suffix()))
                    } else {
                        None
                    }
                })
                .collect()
        };

        Self {
            left_is_build,
            left_input_schema,
            right_input_schema,
            right_rename,
            fused_predicate,
            state: CrossJoinState::Build(sink_node),
        }
    }
}

enum CrossJoinState {
    Build(InMemorySinkNode),
    Probe(DataFrame),
    Done,
}

impl ComputeNode for CrossJoinNode {
    fn name(&self) -> &str {
        "cross-join"
    }

    fn memory_usage(&self) -> NodeMemoryUsage {
        match self.state {
            CrossJoinState::Build(_) => NodeMemoryUsage::Accumulating,
            CrossJoinState::Probe(_) => NodeMemoryUsage::HoldingUntilDone,
            CrossJoinState::Done => NodeMemoryUsage::Bounded,
        }
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        let build_idx = if self.left_is_build { 0 } else { 1 };
        let probe_idx = 1 - build_idx;

        // Are we done?
        if send[0] == PortState::Done || recv[probe_idx] == PortState::Done {
            self.state = CrossJoinState::Done;
        }

        // Transition to build?
        if recv[build_idx] == PortState::Done {
            if let CrossJoinState::Build(sink_node) = &mut self.state {
                let df = sink_node.get_output()?.unwrap();
                if df.height() > 0 {
                    self.state = CrossJoinState::Probe(df);
                } else {
                    self.state = CrossJoinState::Done;
                }
            }
        }

        match &self.state {
            CrossJoinState::Build(_) => {
                recv[build_idx] = PortState::Ready;
                recv[probe_idx] = PortState::Blocked;
                send[0] = PortState::Blocked;
            },
            CrossJoinState::Probe(_) => {
                recv[build_idx] = PortState::Done;
                core::mem::swap(&mut recv[probe_idx], &mut send[0]);
            },
            CrossJoinState::Done => {
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 2 && send_ports.len() == 1);
        let build_idx = if self.left_is_build { 0 } else { 1 };
        let probe_idx = 1 - build_idx;
        match &mut self.state {
            CrossJoinState::Build(sink_node) => {
                assert!(send_ports[0].is_none());
                assert!(recv_ports[probe_idx].is_none());
                sink_node.spawn(
                    scope,
                    &mut recv_ports[build_idx..build_idx + 1],
                    &mut [],
                    state,
                    join_handles,
                );
            },
            CrossJoinState::Probe(build_df) => {
                assert!(recv_ports[build_idx].is_none());
                let receivers = recv_ports[probe_idx].take().unwrap().parallel();
                let senders = send_ports[0].take().unwrap().parallel();
                let ideal_morsel_size = get_ideal_morsel_size();

                for (mut recv, mut send) in receivers.into_iter().zip(senders) {
                    let left_is_build = self.left_is_build;
                    let left_input_schema = &self.left_input_schema;
                    let right_input_schema = &self.right_input_schema;
                    let right_rename = &self.right_rename;
                    let fused_predicate = self.fused_predicate.as_ref();
                    let build_df = &*build_df;
                    join_handles.push(
                        scope.spawn_task(TaskPriority::High, async move {
                            if let Some(predicate) = fused_predicate {
                                let mut indices = CrossJoinIndices::default();
                                let ideal_morsel_size =
                                    ideal_morsel_size.min(IdxSize::MAX as usize);
                                let build_step = build_df.height().min(ideal_morsel_size);
                                let probe_step = ideal_morsel_size / build_step;
                                while let Ok(morsel) = recv.recv().await {
                                    let probe_df = morsel.df().await;
                                    for probe_start in (0..probe_df.height()).step_by(probe_step) {
                                        let probe_len =
                                            (probe_df.height() - probe_start).min(probe_step);
                                        for build_start in
                                            (0..build_df.height()).step_by(build_step)
                                        {
                                            let chunk = CrossJoinChunk {
                                                build: build_df,
                                                probe: &probe_df,
                                                build_start,
                                                build_len: (build_df.height() - build_start)
                                                    .min(build_step),
                                                probe_start,
                                                probe_len,
                                                left_is_build,
                                            };
                                            if let Some(df) = chunk
                                                .filter(
                                                    predicate,
                                                    &mut indices,
                                                    &state.in_memory_exec_state,
                                                )
                                                .await?
                                            {
                                                let output = Morsel::new_unregistered(
                                                    df,
                                                    morsel.seq(),
                                                    morsel.source_token().clone(),
                                                );
                                                if send.send(output).await.is_err() {
                                                    return Ok(());
                                                }
                                            }
                                        }
                                    }
                                }
                                return Ok(());
                            }

                            let mut build_repeater =
                                DataFrameBuilder::new(left_input_schema.clone());
                            let mut probe_repeater =
                                DataFrameBuilder::new(right_input_schema.clone());
                            if !left_is_build {
                                core::mem::swap(&mut build_repeater, &mut probe_repeater);
                            }
                            let mut cached_build_df_repeated = DataFrame::empty();

                            while let Ok(morsel) = recv.recv().await {
                                let combine =
                                    |build_join_df: DataFrame, probe_join_df: DataFrame| unsafe {
                                        let (mut left_join_df, mut right_join_df);
                                        left_join_df = build_join_df;
                                        right_join_df = probe_join_df;
                                        if !left_is_build {
                                            core::mem::swap(&mut left_join_df, &mut right_join_df);
                                        }

                                        for (col, opt_rename) in
                                            right_join_df.columns_mut().iter_mut().zip(right_rename)
                                        {
                                            if let Some(rename) = opt_rename {
                                                col.rename(rename.clone());
                                            }
                                        }

                                        left_join_df.hstack_mut_unchecked(right_join_df.columns());
                                        Morsel::new_unregistered(
                                            left_join_df,
                                            morsel.seq(),
                                            morsel.source_token().clone(),
                                        )
                                    };

                                let probe_df = morsel.df().await;
                                if build_df.height() >= ideal_morsel_size {
                                    for probe_offset in 0..probe_df.height() {
                                        let mut build_offset = 0;
                                        while build_offset < build_df.height() {
                                            let height = (build_df.height() - build_offset)
                                                .min(ideal_morsel_size);
                                            let build_join_df =
                                                build_df.slice((build_offset, height));
                                            let probe_join_df =
                                                probe_df.new_from_index(probe_offset, height);
                                            let combined = combine(build_join_df, probe_join_df);
                                            if send.send(combined).await.is_err() {
                                                return Ok(());
                                            }
                                            build_offset += height;
                                        }
                                    }
                                } else {
                                    let max_build_repeats = ideal_morsel_size / build_df.height();
                                    let mut probe_offset = 0;
                                    while probe_offset < probe_df.height() {
                                        let build_repeats = (probe_df.height() - probe_offset)
                                            .min(max_build_repeats);
                                        let build_height = build_repeats * build_df.height();
                                        if build_height > cached_build_df_repeated.height() {
                                            build_repeater.subslice_extend_repeated(
                                                build_df,
                                                0,
                                                build_df.height(),
                                                build_repeats,
                                                ShareStrategy::Never,
                                            );
                                            cached_build_df_repeated =
                                                build_repeater.freeze_reset();
                                        }
                                        let build_join_df = cached_build_df_repeated
                                            .slice(Slice::head(build_height));

                                        probe_repeater.subslice_extend_each_repeated(
                                            &probe_df,
                                            probe_offset,
                                            build_repeats,
                                            build_df.height(),
                                            ShareStrategy::Always,
                                        );
                                        let probe_join_df = probe_repeater.freeze_reset();

                                        let combined = combine(build_join_df, probe_join_df);
                                        if send.send(combined).await.is_err() {
                                            return Ok(());
                                        }

                                        probe_offset += build_repeats;
                                    }
                                }
                            }
                            Ok(())
                        }),
                    );
                }
            },
            CrossJoinState::Done => unreachable!(),
        }
    }
}
