use std::mem;
use std::ops::Range;

use polars_core::prelude::*;
use polars_core::utils::Container;
use polars_ops::frame::{
    _finish_join, IEJoinOptions, InequalityOperator, JoinArgs, iejoin_par_partition,
};

use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::Morsel;
use crate::nodes::ComputeNode;
use crate::nodes::in_memory_sink::InMemorySinkNode;
use crate::pipe::{PortReceiver, PortSender, RecvPort, SendPort};

#[derive(Debug)]
enum IEJoinState {
    Build(InMemorySinkNode),
    Probe(ProbeState),
    Done,
}

#[derive(Debug)]
struct ProbeState {
    // TODO: [amber] Add doc comments
    build_df: DataFrame,
    build_l1_s: IdxCa,
}

#[derive(Debug)]
pub struct IEJoinNode {
    state: IEJoinState,
    params: IEJoinParams,
}

#[derive(Debug)]
struct IEJoinParams {
    build: RangeJoinSideParams,
    probe: RangeJoinSideParams,
    args: JoinArgs,
    options: IEJoinOptions,
}

impl IEJoinParams {
    fn left_is_build(&self) -> bool {
        true
    }
}

#[derive(Debug)]
struct RangeJoinSideParams {
    schema: SchemaRef,
    on: (PlSmallStr, Option<PlSmallStr>),
    tmp_key_cols: [Option<PlSmallStr>; 2],
}

impl RangeJoinSideParams {
    fn l1_key_col(&self) -> &PlSmallStr {
        self.tmp_key_cols[0].as_ref().unwrap_or(&self.on.0)
    }
    fn l2_key_col(&self) -> Option<&PlSmallStr> {
        self.tmp_key_cols[1].as_ref().or(self.on.1.as_ref())
    }
}

impl IEJoinNode {
    pub fn new(
        left_schema: SchemaRef,
        right_schema: SchemaRef,
        left_on: (PlSmallStr, Option<PlSmallStr>),
        right_on: (PlSmallStr, Option<PlSmallStr>),
        tmp_left_key_cols: [Option<PlSmallStr>; 2],
        tmp_right_key_cols: [Option<PlSmallStr>; 2],
        args: JoinArgs,
        options: IEJoinOptions,
    ) -> Self {
        let have_second_predicate = options.operator2.is_some();
        assert!(have_second_predicate == left_on.1.is_some());
        assert!(have_second_predicate == right_on.1.is_some());
        let build = RangeJoinSideParams {
            schema: left_schema.clone(),
            on: left_on,
            tmp_key_cols: tmp_left_key_cols,
        };
        let probe = RangeJoinSideParams {
            schema: right_schema.clone(),
            on: right_on,
            tmp_key_cols: tmp_right_key_cols,
        };
        let params = IEJoinParams {
            build,
            probe,
            args,
            options,
        };
        IEJoinNode {
            state: IEJoinState::Build(InMemorySinkNode::new(left_schema)),
            params,
        }
    }
}

impl ComputeNode for IEJoinNode {
    fn name(&self) -> &str {
        "ie-join"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        // TODO: [amber] Which side is build?
        let (build, probe) = recv.split_at_mut(1);

        if send[0] == PortState::Done {
            self.state = IEJoinState::Done;
        }

        if let IEJoinState::Build(sink_node) = &mut self.state
            && build[0] == PortState::Done
        {
            self.state = IEJoinState::Probe(transition_to_probe(sink_node, &self.params)?);
        }

        if let IEJoinState::Probe(_) = &self.state
            && probe[0] == PortState::Done
        {
            self.state = IEJoinState::Done;
        }

        match self.state {
            IEJoinState::Build(ref mut sink_node) => {
                sink_node.update_state(build, &mut [], state)?;
                probe[0] = PortState::Blocked;
                send[0] = PortState::Blocked;
            },
            IEJoinState::Probe(_) => {
                build[0] = PortState::Done;
                mem::swap(&mut probe[0], &mut send[0]);
            },
            IEJoinState::Done => {
                build[0] = PortState::Done;
                probe[0] = PortState::Done;
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

        // TODO: [amber] Which side is build?
        let build_idx = 0;
        let probe_idx = 1 - build_idx;
        let params = &self.params;

        match &mut self.state {
            IEJoinState::Build(sink_node) => {
                assert!(recv_ports[probe_idx].is_none());
                let build = recv_ports[build_idx].take().unwrap();
                sink_node.spawn(scope, &mut [Some(build)], &mut [], state, join_handles)
            },
            IEJoinState::Probe(probe_state) => {
                let probe_state = &*probe_state;
                assert!(recv_ports[build_idx].is_none());
                let probe = recv_ports[probe_idx].take().unwrap().parallel();
                let send = send_ports[0].take().unwrap().parallel();
                let (distr_send, distr_recv) =
                    async_channel::bounded(*DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

                join_handles.extend(probe.into_iter().map(|recv| {
                    let send = distr_send.clone();
                    scope.spawn_task(TaskPriority::High, async move {
                        distribute_work_task(recv, probe_state, params, send).await
                    })
                }));
                join_handles.extend(send.into_iter().map(|send| {
                    let recv = distr_recv.clone();
                    scope.spawn_task(TaskPriority::High, async move {
                        compute_and_emit_task(recv, send, probe_state, params).await
                    })
                }));
            },
            IEJoinState::Done => unreachable!(),
        }
    }
}

fn transition_to_probe(
    sink_node: &mut InMemorySinkNode,
    params: &IEJoinParams,
) -> PolarsResult<ProbeState> {
    let build_df = sink_node.get_output()?.expect("sink_node is empty");
    let build_l1 = build_df
        .column(params.build.l1_key_col())?
        .as_materialized_series()
        .to_owned();
    let l1_sort_options = SortOptions::default()
        .with_maintain_order(true)
        .with_multithreaded(true)
        .with_nulls_last(false)
        .with_order_descending(params.options.l1_descending());

    let build_l1_s = build_l1.arg_sort(l1_sort_options);

    // After this point we do not need the temporary columns anymore, as they
    // are moved to build_l1_s and build_l2.
    // for tmp_key_col in &params.build.tmp_key_cols {
    //     if let Some(name) = tmp_key_col {
    //         build_df.drop_in_place(name)?;
    //     }
    // }

    Ok(ProbeState {
        build_df,
        build_l1_s,
    })
}

async fn distribute_work_task(
    mut recv: PortReceiver,
    probe_state: &ProbeState,
    params: &IEJoinParams,
    distributor: async_channel::Sender<(Morsel, IdxCa, Range<usize>, Range<usize>)>,
) -> PolarsResult<()> {
    let l1_sort_options = SortOptions::default()
        .with_maintain_order(true)
        .with_nulls_last(false)
        .with_order_descending(params.options.l1_descending());

    loop {
        let Ok(morsel) = recv.recv().await else {
            return Ok(());
        };
        let l1_select_probe = morsel
            .df()
            .column(params.probe.l1_key_col())?
            .as_materialized_series()
            .to_owned();
        let probe_l1_s = l1_select_probe.arg_sort(l1_sort_options);

        // TODO: [amber] Partition this better
        let range_build = 0..probe_state.build_l1_s.len();
        let range_probe = 0..probe_l1_s.len();
        distributor
            .send((morsel, probe_l1_s, range_build, range_probe))
            .await
            .expect("send error");
    }
}

async fn compute_and_emit_task(
    dist_recv: async_channel::Receiver<(Morsel, IdxCa, Range<usize>, Range<usize>)>,
    mut send: PortSender,
    probe_state: &ProbeState,
    params: &IEJoinParams,
) -> PolarsResult<()> {
    loop {
        let Ok((morsel, probe_l1, range_build, range_probe)) = dist_recv.recv().await else {
            return Ok(());
        };
        let build_l1 = &probe_state.build_l1_s;
        let (probe_df, seq, st, wt) = morsel.into_inner();

        let build_key_columns = match params.build.l2_key_col() {
            None => &[params.build.l1_key_col()][..],
            Some(l2_key_col) => &[params.build.l1_key_col(), l2_key_col][..],
        };
        let probe_key_columns = match params.probe.l2_key_col() {
            None => &[params.probe.l1_key_col()][..],
            Some(l2_key_col) => &[params.probe.l1_key_col(), l2_key_col][..],
        };

        let selected_build = probe_state
            .build_df
            .select_to_vec(build_key_columns)?
            .into_iter()
            .map(|c| {
                c.as_materialized_series()
                    .slice(range_build.start as i64, range_build.len())
                    .to_owned()
            })
            .collect::<Vec<_>>();

        let selected_probe = probe_df
            .select_to_vec(probe_key_columns)?
            .into_iter()
            .map(|c| {
                c.as_materialized_series()
                    .slice(range_probe.start as i64, range_probe.len())
                    .to_owned()
            })
            .collect::<Vec<_>>();

        let Some((build_rows, probe_rows)) = iejoin_par_partition(
            build_l1,
            &probe_l1,
            &selected_build,
            &selected_probe,
            &params.options,
        )?
        else {
            continue;
        };

        let build_gather = unsafe { probe_state.build_df.take_unchecked(&build_rows) };
        let probe_gather = unsafe { probe_df.take_unchecked(&probe_rows) };
        let output = if params.left_is_build() {
            _finish_join(build_gather, probe_gather, params.args.suffix.clone())?
        } else {
            _finish_join(probe_gather, build_gather, params.args.suffix.clone())?
        };
        // TODO: [amber] We still need to drop the key columns
        let mut morsel = Morsel::new(output, seq, st);
        if let Some(wt) = wt {
            morsel.set_consume_token(wt);
        }
        dbg!(morsel.df().height());
        if send.send(morsel).await.is_err() {
            return Ok(());
        }
    }
}
