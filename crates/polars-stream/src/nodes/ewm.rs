use arrow::legacy::kernels::ewm::EwmStateUpdate;
use polars_core::prelude::IntoColumn;
use polars_core::series::Series;
use polars_error::PolarsResult;

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::pipe::{RecvPort, SendPort};

pub struct EwmNode {
    name: &'static str,
    state: Box<dyn EwmStateUpdate + Send>,
}

impl EwmNode {
    pub fn new(name: &'static str, state: Box<dyn EwmStateUpdate + Send>) -> Self {
        Self { name, state }
    }
}

impl ComputeNode for EwmNode {
    fn name(&self) -> &str {
        self.name
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
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert_eq!(recv_ports.len(), 1);
        assert_eq!(send_ports.len(), 1);

        let mut recv = recv_ports[0].take().unwrap().serial();
        let mut send = send_ports[0].take().unwrap().serial();

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(mut morsel) = recv.recv().await {
                let df = morsel.df_mut();

                debug_assert_eq!(df.width(), 1);

                unsafe {
                    let c = df.get_columns_mut().get_mut(0).unwrap();

                    *c = Series::from_chunks_and_dtype_unchecked(
                        c.name().clone(),
                        vec![self.state.ewm_state_update(
                            c.as_materialized_series().rechunk().chunks()[0].as_ref(),
                        )],
                        c.dtype(),
                    )
                    .into_column()
                }

                if send.send(morsel).await.is_err() {
                    break;
                }
            }

            Ok(())
        }));
    }
}
