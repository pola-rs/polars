use polars_error::polars_err;

use super::compute_node_prelude::*;
use crate::expression::StreamExpr;

pub struct ShiftNode {
    column: StreamExpr,
    offset: StreamExpr,
    fill_value: Option<StreamExpr>,
}

impl ShiftNode {
    pub fn new(column: StreamExpr, offset: StreamExpr, fill_value: Option<StreamExpr>) -> Self {
        Self {
            column,
            offset,
            fill_value,
        }
    }
}

impl ComputeNode for ShiftNode {
    fn name(&self) -> &str {
        "shift"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);
        recv.swap_with_slice(send);
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
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);
        let mut receiver = recv_ports[0].take().unwrap().serial();
        let mut sender = send_ports[0].take().unwrap().serial();

        let t = scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = receiver.recv().await {
                dbg!(morsel.df().shift(1));

                if sender.send(morsel).await.is_err() {
                    break;
                }
            }

            Ok(())
        });
        join_handles.push(t);
    }
}
