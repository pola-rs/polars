use std::mem;

use arrow::array::builder::ShareStrategy;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_ops::frame::{_finish_join, IEJoinOptions, InequalityOperator, JoinArgs};
use polars_ops::series::{SearchSortedSide, search_sorted};

use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken, get_ideal_morsel_size};
use crate::nodes::ComputeNode;
use crate::nodes::in_memory_sink::InMemorySinkNode;
use crate::pipe::{PortReceiver, PortSender, RecvPort, SendPort};

// TODO: [amber]
//   * Move sort into lowering
//   * Support build-side configuration

#[derive(Debug)]
enum RangeJoinState {
    Build(InMemorySinkNode),
    Probe(ProbeState),
    Done,
}

#[derive(Debug)]
struct ProbeState {
    point_df: DataFrame,
}

#[derive(Debug)]
pub struct RangeJoinNode {
    state: RangeJoinState,
    params: RangeJoinParams,
}

#[derive(Debug)]
struct RangeJoinParams {
    point_on: PlSmallStr,
    point_tmp_key_col: Option<PlSmallStr>,
    lower_on: Option<PlSmallStr>,
    lower_tmp_key_col: Option<PlSmallStr>,
    lower_op: Option<InequalityOperator>,
    upper_on: Option<PlSmallStr>,
    upper_op: Option<InequalityOperator>,
    upper_tmp_key_col: Option<PlSmallStr>,
    point_schema: SchemaRef,
    interval_schema: SchemaRef,
    output_schema: SchemaRef,
    args: JoinArgs,
    left_is_point: bool,
}

impl RangeJoinParams {
    fn point_key_col(&self) -> &PlSmallStr {
        self.point_tmp_key_col.as_ref().unwrap_or(&self.point_on)
    }
    fn lower_key_col(&self) -> Option<&PlSmallStr> {
        self.lower_tmp_key_col.as_ref().or(self.lower_on.as_ref())
    }
    fn upper_key_col(&self) -> Option<&PlSmallStr> {
        self.upper_tmp_key_col.as_ref().or(self.upper_on.as_ref())
    }
}

impl RangeJoinNode {
    pub fn new(
        left_schema: SchemaRef,
        right_schema: SchemaRef,
        left_on: Vec<PlSmallStr>,
        right_on: Vec<PlSmallStr>,
        tmp_left_key_cols: [Option<PlSmallStr>; 2],
        tmp_right_key_cols: [Option<PlSmallStr>; 2],
        output_schema: SchemaRef,
        args: JoinArgs,
        options: IEJoinOptions,
    ) -> Self {
        let left_is_point = left_on.len() == 1;
        let ops_n = if options.operator2.is_some() { 2 } else { 1 };
        let op1_is_lower_bound = match (left_is_point, options.operator1) {
            (true, InequalityOperator::Gt | InequalityOperator::GtEq) => true,
            (true, InequalityOperator::Lt | InequalityOperator::LtEq) => false,
            (false, InequalityOperator::Gt | InequalityOperator::GtEq) => false,
            (false, InequalityOperator::Lt | InequalityOperator::LtEq) => true,
        };
        let mut point_on;
        let mut point_tmp_key_cols;
        let point_schema;
        let mut interval_on_vec;
        let mut interval_tmp_key_cols;
        let interval_schema;
        if left_is_point {
            assert!(left_on.len() == 1 && right_on.len() == ops_n);
            point_on = left_on;
            point_tmp_key_cols = tmp_left_key_cols;
            point_schema = left_schema;
            interval_on_vec = right_on;
            interval_tmp_key_cols = tmp_right_key_cols;
            interval_schema = right_schema;
        } else {
            assert!(right_on.len() == 1 && left_on.len() == ops_n);
            point_on = right_on;
            point_tmp_key_cols = tmp_right_key_cols;
            point_schema = right_schema;
            interval_on_vec = left_on;
            interval_tmp_key_cols = tmp_left_key_cols;
            interval_schema = left_schema;
        };
        let point_on = mem::take(&mut point_on[0]);
        let point_tmp_key_col = mem::take(&mut point_tmp_key_cols[0]);
        let (lower_on, lower_tmp_key_col, lower_op, upper_on, upper_tmp_key_col, upper_op) =
            match (ops_n, op1_is_lower_bound) {
                (2, _) => (
                    interval_on_vec.get_mut(0).map(mem::take),
                    mem::take(&mut interval_tmp_key_cols[0]),
                    Some(options.operator1),
                    interval_on_vec.get_mut(1).map(mem::take),
                    mem::take(&mut interval_tmp_key_cols[1]),
                    Some(options.operator2.unwrap()),
                ),
                (1, true) => (
                    interval_on_vec.get_mut(0).map(mem::take),
                    mem::take(&mut interval_tmp_key_cols[0]),
                    Some(options.operator1),
                    None,
                    None,
                    None,
                ),
                (1, false) => (
                    None,
                    None,
                    None,
                    interval_on_vec.get_mut(0).map(mem::take),
                    mem::take(&mut interval_tmp_key_cols[0]),
                    Some(options.operator1),
                ),
                _ => unreachable!(),
            };
        let params = RangeJoinParams {
            point_on,
            point_tmp_key_col,
            lower_on,
            lower_tmp_key_col,
            lower_op,
            upper_on,
            upper_tmp_key_col,
            upper_op,
            point_schema: point_schema.clone(),
            interval_schema,
            output_schema,
            args,
            left_is_point,
        };
        RangeJoinNode {
            state: RangeJoinState::Build(InMemorySinkNode::new(point_schema)),
            params,
        }
    }
}

impl ComputeNode for RangeJoinNode {
    fn name(&self) -> &str {
        "range-join"
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        !matches!(self.state, RangeJoinState::Done)
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        let (point, interval) = recv.split_at_mut(1);

        if send[0] == PortState::Done {
            self.state = RangeJoinState::Done;
        }

        if let RangeJoinState::Build(sink_node) = &mut self.state
            && point[0] == PortState::Done
        {
            self.state = RangeJoinState::Probe(transition_to_probe(sink_node, &self.params)?);
        }

        if let RangeJoinState::Probe(_) = &self.state
            && interval[0] == PortState::Done
        {
            self.state = RangeJoinState::Done;
        }

        match self.state {
            RangeJoinState::Build(ref mut sink_node) => {
                sink_node.update_state(point, &mut [], state)?;
                interval[0] = PortState::Blocked;
                send[0] = PortState::Blocked;
            },
            RangeJoinState::Probe(_) => {
                point[0] = PortState::Done;
                mem::swap(&mut interval[0], &mut send[0]);
            },
            RangeJoinState::Done => {
                point[0] = PortState::Done;
                interval[0] = PortState::Done;
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

        let params = &self.params;
        let point_idx = if params.left_is_point { 0 } else { 1 };
        let interval_idx = 1 - point_idx;

        match &mut self.state {
            RangeJoinState::Build(sink_node) => {
                assert!(recv_ports[interval_idx].is_none());
                let build = recv_ports[point_idx].take().unwrap();
                sink_node.spawn(scope, &mut [Some(build)], &mut [], state, join_handles)
            },
            RangeJoinState::Probe(probe_state) => {
                assert!(recv_ports[point_idx].is_none());
                let probe_state = &*probe_state;
                let recv = recv_ports[interval_idx].take().unwrap().parallel();
                let send = send_ports[0].take().unwrap().parallel();
                join_handles.extend(Iterator::zip(recv.into_iter(), send.into_iter()).map(
                    |(recv, send)| {
                        scope.spawn_task(TaskPriority::High, async move {
                            compute_and_emit_task(recv, send, probe_state, params).await
                        })
                    },
                ));
            },
            RangeJoinState::Done => unreachable!(),
        }
    }
}

fn transition_to_probe(
    sink_node: &mut InMemorySinkNode,
    params: &RangeJoinParams,
) -> PolarsResult<ProbeState> {
    let sort_options = SortMultipleOptions::default().with_multithreaded(true);
    let mut point_df = sink_node.get_output()?.unwrap();
    point_df.sort_in_place([params.point_key_col()], sort_options)?;
    Ok(ProbeState { point_df })
}

async fn compute_and_emit_task(
    mut recv: PortReceiver,
    mut send: PortSender,
    probe_state: &ProbeState,
    params: &RangeJoinParams,
) -> PolarsResult<()> {
    let sss_lower = match params.lower_op {
        Some(InequalityOperator::GtEq) => Some(SearchSortedSide::Left),
        Some(InequalityOperator::Gt) => Some(SearchSortedSide::Right),
        Some(_) => unreachable!("lower_op is not a lower-bound operator"),
        _ => None,
    };
    let sss_upper = match params.upper_op {
        Some(InequalityOperator::LtEq) => Some(SearchSortedSide::Right),
        Some(InequalityOperator::Lt) => Some(SearchSortedSide::Left),
        Some(_) => unreachable!("upper_op is not an upper-bound operator"),
        _ => None,
    };

    let ProbeState { point_df } = probe_state;
    let point_key = point_df
        .column(params.point_key_col())?
        .as_materialized_series();

    let mut seq = MorselSeq::default();
    let mut st = SourceToken::default();
    let wait_group = WaitGroup::default();
    let mut builder_point = DataFrameBuilder::new(params.point_schema.clone());
    let mut builder_interval = DataFrameBuilder::new(params.interval_schema.clone());

    loop {
        let interval_df;
        if let Ok(morsel) = recv.recv().await {
            (interval_df, seq, st, _) = morsel.into_inner();
        } else {
            if !builder_point.is_empty() {
                freeze_builders_and_emit(
                    &mut send,
                    &mut builder_point,
                    &mut builder_interval,
                    params,
                    seq,
                    st.clone(),
                    None,
                )
                .await?;
            }
            return Ok(());
        };

        let starts = params
            .lower_key_col()
            .map(|c| {
                let search_values = interval_df.column(c)?.as_materialized_series();
                search_sorted(point_key, search_values, sss_lower.unwrap(), false)
            })
            .transpose()?;
        let ends = params
            .upper_key_col()
            .map(|c| {
                let search_values = interval_df.column(c)?.as_materialized_series();
                search_sorted(point_key, search_values, sss_upper.unwrap(), false)
            })
            .transpose()?;

        let starts = match starts {
            Some(v) => v,
            None => IdxCa::new_vec(
                PlSmallStr::EMPTY,
                vec![0 as IdxSize; ends.as_ref().unwrap().len()],
            ),
        };
        let ends = match ends {
            Some(v) => v,
            None => IdxCa::new_vec(
                PlSmallStr::EMPTY,
                vec![point_df.height() as IdxSize; starts.len()],
            ),
        };

        for (row_idx, (start, end)) in
            Iterator::zip(starts.into_no_null_iter(), ends.into_no_null_iter()).enumerate()
        {
            if !(start <= end) {
                continue;
            }

            let match_len = (end - start) as usize;
            builder_point.subslice_extend(
                point_df,
                start as usize,
                match_len,
                ShareStrategy::Never,
            );
            builder_interval.subslice_extend_repeated(
                &interval_df,
                row_idx,
                1,
                match_len,
                ShareStrategy::Never,
            );
            debug_assert!(builder_point.len() == builder_interval.len());

            if builder_point.len() >= get_ideal_morsel_size() {
                freeze_builders_and_emit(
                    &mut send,
                    &mut builder_point,
                    &mut builder_interval,
                    params,
                    seq,
                    st.clone(),
                    Some(wait_group.token()),
                )
                .await?;
                wait_group.wait().await;
            }
        }
    }
}

async fn freeze_builders_and_emit(
    send: &mut PortSender,
    builder_point: &mut DataFrameBuilder,
    builder_interval: &mut DataFrameBuilder,
    params: &RangeJoinParams,
    seq: MorselSeq,
    st: SourceToken,
    wt: Option<WaitToken>,
) -> PolarsResult<()> {
    let results_point = builder_point.freeze_reset();
    let results_interval = builder_interval.freeze_reset();

    let mut output = if params.left_is_point {
        _finish_join(results_point, results_interval, params.args.suffix.clone())?
    } else {
        _finish_join(results_interval, results_point, params.args.suffix.clone())?
    };

    drop_key_columns(&mut output, params);
    debug_assert!(*output.schema() == params.output_schema);
    let mut morsel = Morsel::new(output, seq, st);
    if let Some(wt) = wt {
        morsel.set_consume_token(wt);
    }
    if send.send(morsel).await.is_err() {
        return Ok(());
    }
    Ok(())
}

fn drop_key_columns(df: &mut DataFrame, params: &RangeJoinParams) {
    for col in [
        &params.point_tmp_key_col,
        &params.lower_tmp_key_col,
        &params.upper_tmp_key_col,
    ] {
        if let Some(col) = col
            && df.schema().contains(col)
        {
            df.drop_in_place(col).unwrap();
        }
    }
}
