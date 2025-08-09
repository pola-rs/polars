use polars_core::frame::DataFrame;
use polars_error::PolarsResult;
use polars_plan::prelude::PlanCallback;

use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::nodes::ComputeNode;
use crate::pipe::{RecvPort, SendPort};

pub struct CallbackSinkNode {
    function: PlanCallback<DataFrame, bool>,
    maintain_order: bool,

    is_done: bool,
}

impl CallbackSinkNode {
    pub fn new(function: PlanCallback<DataFrame, bool>, maintain_order: bool) -> Self {
        Self {
            function,
            maintain_order,

            is_done: false,
        }
    }
}

impl ComputeNode for CallbackSinkNode {
    fn name(&self) -> &str {
        "sink_batches"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.is_empty());

        if self.is_done || recv[0] == PortState::Done {
            recv[0] = PortState::Done;
        } else {
            recv[0] = PortState::Ready;
        }

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1 && send_ports.is_empty());
        let mut recv = recv_ports[0]
            .take()
            .unwrap()
            .serial_with_maintain_order(self.maintain_order);

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(m) = recv.recv().await {
                if !self.function.call(m.into_df())? {
                    self.is_done = true;
                    break;
                }
            }

            Ok(())
        }));
    }
}
