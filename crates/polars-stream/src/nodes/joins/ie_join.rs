use polars_core::prelude::*;
use polars_ops::frame::{IEJoinOptions, JoinArgs};

use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::distributor_channel as dc;
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{MorselSeq, SourceToken};
use crate::nodes::ComputeNode;
use crate::nodes::in_memory_sink::InMemorySinkNode;
use crate::nodes::joins::utils::DataFrameSearchBuffer as DFSB;
use crate::pipe::{PortReceiver, PortSender, RecvPort, SendPort};

#[derive(Clone, Copy, Debug)]
enum NeedMore {
    Left,
    Right,
}

#[derive(Debug)]
enum IEJoinState {
    Build(InMemorySinkNode),
    Probe(ProbeState),
    Done,
}

#[derive(Debug)]
struct ProbeState {}

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

#[derive(Debug)]
struct RangeJoinSideParams {
    schema: SchemaRef,
    on: (PlSmallStr, Option<PlSmallStr>),
    tmp_key_cols: (Option<PlSmallStr>, Option<PlSmallStr>),
}

impl RangeJoinSideParams {
    fn fst_key_col(&self) -> &PlSmallStr {
        self.tmp_key_cols.0.as_ref().unwrap_or(&self.on.0)
    }
    fn snd_key_col(&self) -> Option<&PlSmallStr> {
        self.tmp_key_cols.1.as_ref().or(self.on.1.as_ref())
    }
}

impl IEJoinNode {
    pub fn new(
        left_schema: SchemaRef,
        right_schema: SchemaRef,
        left_on: (PlSmallStr, Option<PlSmallStr>),
        right_on: (PlSmallStr, Option<PlSmallStr>),
        tmp_left_key_cols: (Option<PlSmallStr>, Option<PlSmallStr>),
        tmp_right_key_cols: (Option<PlSmallStr>, Option<PlSmallStr>),
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

        let input_channels_done = recv.iter().all(|r| *r == PortState::Done);
        // let input_buffers_empty = self.left_dfsb.is_empty() && self.right_dfsb.is_empty();

        if send[0] == PortState::Done {
            self.state = IEJoinState::Done;
        }

        if let IEJoinState::Build(build) = &self.state
            && input_channels_done
        {
            self.state = IEJoinState::Probe(transition_to_probe(build)?);
        }

        todo!()
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

        match &mut self.state {
            IEJoinState::Build(sink_node) => {
                // assert!(send_ports[0].is_none());
                // assert!(recv_ports[probe_idx].is_none());
                // sink_node.spawn(
                //     scope,
                //     &mut recv_ports[build_idx..build_idx + 1],
                //     &mut [],
                //     state,
                //     join_handles,
                // );
                todo!()
            },
            IEJoinState::Probe(probe) => todo!(),
            IEJoinState::Done => unreachable!(),
        }
    }
}

async fn distribute_work_task(
    mut recv_left: Option<PortReceiver>,
    mut recv_right: Option<PortReceiver>,
    mut distributor: dc::Sender<(DFSB, DFSB, MorselSeq, SourceToken)>,
    right_dfsb: &mut DFSB,
    params: &IEJoinParams,
) -> PolarsResult<()> {
    todo!()
}

fn transition_to_probe(build: &InMemorySinkNode) -> PolarsResult<ProbeState> {
    todo!()
}

async fn compute_and_emit_task(
    mut dist_recv: dc::Receiver<(DFSB, DFSB, MorselSeq, SourceToken)>,
    mut send: PortSender,
    params: &IEJoinParams,
) -> PolarsResult<()> {
    todo!()
}
