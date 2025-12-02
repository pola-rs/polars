use polars_core::prelude::*;
use polars_ops::frame::JoinArgs;

use crate::async_executor::{JoinHandle, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::expression::StreamExpr;
use crate::graph::PortState;
use crate::nodes::ComputeNode;
use crate::pipe::{RecvPort, SendPort};

#[derive(Default)]
pub struct MergeJoinNode {
    left_input_schema: Arc<Schema>,
    right_input_schema: Arc<Schema>,
    left_key_schema: Arc<Schema>,
    right_key_schema: Arc<Schema>,
    unique_key_schema: Arc<Schema>,
    left_key_selectors: Vec<StreamExpr>,
    right_key_selectors: Vec<StreamExpr>,
    args: JoinArgs,
    state: MergeJoinState,
}

#[derive(Default, PartialEq, Eq)]
enum MergeJoinState {
    #[default]
    Running,
    Flushing,
    Done,
}

impl MergeJoinNode {
    pub fn new(
        left_input_schema: Arc<Schema>,
        right_input_schema: Arc<Schema>,
        left_key_schema: Arc<Schema>,
        right_key_schema: Arc<Schema>,
        unique_key_schema: Arc<Schema>,
        left_key_selectors: Vec<StreamExpr>,
        right_key_selectors: Vec<StreamExpr>,
        args: JoinArgs,
    ) -> PolarsResult<Self> {
        Ok(Self {
            left_input_schema,
            right_input_schema,
            left_key_schema,
            right_key_schema,
            unique_key_schema,
            left_key_selectors,
            right_key_selectors,
            args,
            ..Default::default()
        })
    }

    fn buffers_are_empty(&self) -> bool {
        true
    }
}

impl ComputeNode for MergeJoinNode {
    fn name(&self) -> &str {
        "merge-join"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        if send[0] == PortState::Done {
            self.state = MergeJoinState::Done;
        }

        // Transition to Flushing state to flush any remaining output.
        if self.state == MergeJoinState::Running
            && recv[0] == PortState::Done
            && recv[1] == PortState::Done
        {
            self.state = MergeJoinState::Flushing;
        }

        // Are we done flushing?
        if self.state == MergeJoinState::Flushing && self.buffers_are_empty() {
            self.state = MergeJoinState::Done;
        }

        match self.state {
            MergeJoinState::Running => {
                recv[0] = PortState::Ready;
                recv[1] = PortState::Ready;
                send[0] = PortState::Ready;
            },
            MergeJoinState::Flushing => {
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
                send[0] = PortState::Ready;
            },
            MergeJoinState::Done => {
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

        todo!()
    }
}
