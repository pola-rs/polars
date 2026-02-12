use polars_core::prelude::*;
use polars_ops::frame::{IEJoinOptions, JoinArgs};

use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::distributor_channel as dc;
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{MorselSeq, SourceToken};
use crate::nodes::ComputeNode;
use crate::nodes::joins::utils::DataFrameSearchBuffer as DFSB;
use crate::pipe::{PortReceiver, PortSender, RecvPort, SendPort};

#[derive(Clone, Copy, Debug)]
enum NeedMore {
    Left,
    Right,
}

#[derive(Debug, PartialEq)]
enum RangeJoinState {
    Running,
    FlushInputBuffers,
    Done,
}

#[derive(Debug)]
pub struct RangeJoinNode {
    state: RangeJoinState,
    params: RangeJoinParams,
    left_dfsb: DFSB,
    right_dfsb: DFSB,
}

#[derive(Debug)]
struct RangeJoinParams {
    left: RangeJoinSideParams,
    right: RangeJoinSideParams,
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

impl RangeJoinNode {
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
        let left = RangeJoinSideParams {
            schema: left_schema.clone(),
            on: left_on,
            tmp_key_cols: tmp_left_key_cols,
        };
        let right = RangeJoinSideParams {
            schema: right_schema.clone(),
            on: right_on,
            tmp_key_cols: tmp_right_key_cols,
        };
        let params = RangeJoinParams {
            left,
            right,
            args,
            options,
        };
        RangeJoinNode {
            state: RangeJoinState::Running,
            params,
            left_dfsb: DFSB::empty_with_schema(left_schema),
            right_dfsb: DFSB::empty_with_schema(right_schema),
        }
    }
}

impl ComputeNode for RangeJoinNode {
    fn name(&self) -> &str {
        "range-join"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        let input_channels_done = recv.iter().all(|r| *r == PortState::Done);
        let input_buffers_empty = self.left_dfsb.is_empty() && self.right_dfsb.is_empty();

        if send[0] == PortState::Done {
            self.state = RangeJoinState::Done;
        }

        if self.state == RangeJoinState::Running && input_channels_done {
            self.state = RangeJoinState::FlushInputBuffers;
        }

        if self.state == RangeJoinState::FlushInputBuffers && input_buffers_empty {
            self.state = RangeJoinState::Done;
        }

        let recv0_blocked = recv[0] == PortState::Blocked;
        let recv1_blocked = recv[1] == PortState::Blocked;
        let send_blocked = send[0] == PortState::Blocked;
        match self.state {
            RangeJoinState::Running => {
                recv.fill(PortState::Ready);
                send[0] = PortState::Ready;
                if recv0_blocked {
                    recv[1] = PortState::Blocked;
                    send[0] = PortState::Blocked;
                }
                if recv1_blocked {
                    recv[0] = PortState::Blocked;
                    send[0] = PortState::Blocked;
                }
                if send_blocked {
                    recv[0] = PortState::Blocked;
                    recv[1] = PortState::Blocked;
                }
            },
            RangeJoinState::FlushInputBuffers => {
                recv.fill(PortState::Done);
                send[0] = PortState::Ready;
            },
            RangeJoinState::Done => {
                recv.fill(PortState::Done);
                send[0] = PortState::Done;
            },
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

        match self.state {
            RangeJoinState::Running | RangeJoinState::FlushInputBuffers => {
                let flush = self.state == RangeJoinState::FlushInputBuffers;
                let recv_left = recv_ports[0].take().map(RecvPort::serial);
                let recv_right = recv_ports[1].take().map(RecvPort::serial);
                let send = send_ports[0].take().unwrap().parallel();
                let (distributor, dist_recv) =
                    dc::distributor_channel(send.len(), *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);
                let left_dfsb = &mut self.left_dfsb;
                let right_dfsb = &mut self.right_dfsb;
                let params = &self.params;
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    distribute_work_task(
                        recv_left,
                        recv_right,
                        distributor,
                        left_dfsb,
                        right_dfsb,
                        params,
                    )
                    .await
                }));
                join_handles.extend(dist_recv.into_iter().zip(send).map(|(mut recv, mut send)| {
                    scope.spawn_task(TaskPriority::High, async move {
                        compute_and_emit_task(recv, send, params).await
                    })
                }));
            },
            RangeJoinState::Done => unreachable!(),
        }
    }
}

async fn distribute_work_task(
    mut recv_left: Option<PortReceiver>,
    mut recv_right: Option<PortReceiver>,
    mut distributor: dc::Sender<(DFSB, DFSB, MorselSeq, SourceToken)>,
    left_dfsb: &mut DFSB,
    right_dfsb: &mut DFSB,
    params: &RangeJoinParams,
) -> PolarsResult<()> {
    todo!()
}

async fn compute_and_emit_task(
    mut dist_recv: dc::Receiver<(DFSB, DFSB, MorselSeq, SourceToken)>,
    mut send: PortSender,
    params: &RangeJoinParams,
) -> PolarsResult<()> {
    todo!()
}
