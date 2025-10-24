use arrow::legacy::kernels::ewm::{DynEwmMeanState, EwmMeanState};
use polars_core::prelude::{DataType, IntoColumn};
use polars_core::series::Series;
use polars_core::with_match_physical_float_type;
use polars_error::PolarsResult;
use polars_ops::series::EWMOptions;

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::pipe::{RecvPort, SendPort};

pub struct EwmMeanNode {
    state: DynEwmMeanState,
}

impl EwmMeanNode {
    pub fn new(dtype: DataType, options: &EWMOptions) -> Self {
        let state: DynEwmMeanState = with_match_physical_float_type!(dtype, |$T| {
            let state: EwmMeanState<$T> = EwmMeanState::new(
                options.alpha as $T,
                options.adjust,
                options.min_periods,
                options.ignore_nulls,
            );

            state.into()
        });

        Self { state }
    }
}

impl ComputeNode for EwmMeanNode {
    fn name(&self) -> &str {
        "ewm-mean"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

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
                        vec![
                            self.state
                                .update(c.as_materialized_series().rechunk().chunks()[0].as_ref()),
                        ],
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
