use std::mem;
use std::ops::Range;

use polars_core::prelude::search_sorted::binary_search_ca;
use polars_core::prelude::*;
use polars_core::utils::{
    _split_offsets, accumulate_dataframes_vertical_unchecked, concat_df_unchecked,
};
use polars_ops::frame::{_finish_join, IEJoinOptions, JoinArgs, iejoin_par_partition};
use polars_ops::series::SearchSortedSide;

use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{Morsel, get_ideal_morsel_size};
use crate::nodes::ComputeNode;
use crate::nodes::in_memory_sink::InMemorySinkNode;
use crate::pipe::{PortReceiver, PortSender, RecvPort, SendPort};

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
    point: RangeJoinSideParams,
    interval: RangeJoinSideParams,
    args: JoinArgs,
    options: IEJoinOptions,
    left_is_point: bool,
}

#[derive(Debug)]
struct RangeJoinSideParams {
    schema: SchemaRef,
    on: Vec<PlSmallStr>,
    tmp_key_cols: [Option<PlSmallStr>; 2],
}

impl RangeJoinSideParams {
    fn key_col<const IDX: usize>(&self) -> &PlSmallStr {
        debug_assert!((0..=1).contains(&IDX));
        let idx = IDX.clamp(0, self.on.len() - 1);
        self.tmp_key_cols[idx].as_ref().unwrap_or(&self.on[idx])
    }
}

impl RangeJoinNode {
    pub fn new(
        left_schema: SchemaRef,
        right_schema: SchemaRef,
        left_on: PlSmallStr,
        right_on: Vec<PlSmallStr>,
        tmp_left_key_cols: Option<PlSmallStr>,
        tmp_right_key_cols: [Option<PlSmallStr>; 2],
        args: JoinArgs,
        options: IEJoinOptions,
    ) -> Self {
        if options.operator2.is_some() {
            assert!(left_on.len() > 1 || right_on.len() > 1);
        }
        let point = RangeJoinSideParams {
            schema: left_schema.clone(),
            on: vec![left_on],
            tmp_key_cols: [tmp_left_key_cols, None],
        };
        let interval = RangeJoinSideParams {
            schema: right_schema.clone(),
            on: right_on,
            tmp_key_cols: tmp_right_key_cols,
        };
        let params = RangeJoinParams {
            point,
            interval,
            args,
            options,
            left_is_point: true,
        };
        RangeJoinNode {
            state: RangeJoinState::Build(InMemorySinkNode::new(left_schema)),
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

        // TODO: [amber] Which side is point?
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

        // TODO: [amber] Which side is point?
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
    point_df.sort_in_place([params.point.key_col::<0>()], sort_options)?;
    Ok(ProbeState { point_df })
}

async fn compute_and_emit_task(
    mut recv: PortReceiver,
    mut send: PortSender,
    probe_state: &ProbeState,
    params: &RangeJoinParams,
) -> PolarsResult<()> {
    let ProbeState { point_df } = probe_state;
    let point_key = point_df
        .column(params.point.key_col::<0>())?
        .as_materialized_series();

    let point_key_i64: &Int64Chunked = point_key.as_ref().as_ref();

    // TODO: LEFT HERE
    //
    // Very crude implementation of the double-bounded range join.
    // Make it less buggy and less slow.

    loop {
        let Ok(morsel) = recv.recv().await else {
            return Ok(());
        };
        let (df, seq, st, wt) = morsel.into_inner();

        let lower_bound_key: &Int64Chunked = df
            .column(params.interval.key_col::<0>())?
            .as_materialized_series()
            .as_ref()
            .as_ref();
        let upper_bound_key: &Int64Chunked = df
            .column(params.interval.key_col::<1>())?
            .as_materialized_series()
            .as_ref()
            .as_ref();

        let starts = binary_search_ca(
            point_key_i64,
            lower_bound_key.iter(),
            SearchSortedSide::Left,
            false,
        );
        let ends = binary_search_ca(
            point_key_i64,
            upper_bound_key.iter(),
            SearchSortedSide::Right,
            false,
        );
        for (row_idx, (start, end)) in
            Iterator::zip(starts.into_iter(), ends.into_iter()).enumerate()
        {
            let sliced = point_df.slice(start as i64, (end - start) as usize);
            let repeat = std::iter::repeat_n(df.slice(row_idx as i64, 1), (end - start) as usize);
            let interval_gather = accumulate_dataframes_vertical_unchecked(repeat);
            debug_assert_eq!(sliced.height(), interval_gather.height());

            let mut output = if params.left_is_point {
                _finish_join(sliced, interval_gather, params.args.suffix.clone())?
            } else {
                _finish_join(interval_gather, sliced, params.args.suffix.clone())?
            };
            drop_key_columns(&mut output, params);
            if send
                .send(Morsel::new(output, seq, st.clone()))
                .await
                .is_err()
            {
                return Ok(());
            }
        }
    }
}

fn drop_key_columns(df: &mut DataFrame, params: &RangeJoinParams) {
    for col in Iterator::chain(
        params.point.tmp_key_cols.iter(),
        params.interval.tmp_key_cols.iter(),
    ) {
        if let Some(col) = col
            && df.schema().contains(col)
        {
            df.drop_in_place(col).unwrap();
        }
    }
}
