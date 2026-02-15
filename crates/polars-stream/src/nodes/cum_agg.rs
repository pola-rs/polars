use polars_core::prelude::{AnyValue, DataType, IntoColumn};
use polars_core::utils::last_non_null;
use polars_error::{PolarsResult, polars_bail};
#[cfg(feature = "dtype-decimal")]
use polars_ops::series::cum_mean_decimal_with_init;
use polars_ops::series::{
    cum_count_with_init, cum_max_with_init, cum_mean_with_init, cum_min_with_init,
    cum_prod_with_init, cum_sum_with_init,
};

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::pipe::{RecvPort, SendPort};

pub struct CumAggNode {
    state: CumAggState,
    kind: CumAggKind,
}

#[derive(Debug, Clone)]
pub enum CumAggState {
    Single(AnyValue<'static>),
    Mean(CumMeanState),
}

#[derive(Debug, Clone)]
pub enum CumMeanState {
    Float {
        sum: f64,
        count: u64,
        err: f64,
    },
    #[cfg(feature = "dtype-decimal")]
    Decimal {
        sum: i128,
        count: u64,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum CumAggKind {
    Min,
    Max,
    Sum,
    Count,
    Prod,
    Mean,
}

impl CumAggNode {
    #[allow(unused_variables)]
    pub fn new(kind: CumAggKind, dtype: &DataType) -> Self {
        let state = match kind {
            #[cfg(feature = "dtype-decimal")]
            CumAggKind::Mean if matches!(dtype, DataType::Decimal(_, _)) => {
                CumAggState::Mean(CumMeanState::Decimal { sum: 0, count: 0 })
            },
            CumAggKind::Mean => CumAggState::Mean(CumMeanState::Float {
                sum: 0.0,
                count: 0,
                err: 0.0,
            }),
            _ => CumAggState::Single(AnyValue::Null),
        };
        Self { state, kind }
    }
}

impl ComputeNode for CumAggNode {
    fn name(&self) -> &str {
        match self.kind {
            CumAggKind::Min => "cum_min",
            CumAggKind::Max => "cum_max",
            CumAggKind::Sum => "cum_sum",
            CumAggKind::Count => "cum_count",
            CumAggKind::Prod => "cum_prod",
            CumAggKind::Mean => "cum_mean",
        }
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
            while let Ok(mut m) = recv.recv().await {
                assert_eq!(m.df().width(), 1);
                if m.df().height() == 0 {
                    continue;
                }
                let s = m.df()[0].as_materialized_series();
                let out = match (&mut self.state, self.kind) {
                    (CumAggState::Mean(mean_state), CumAggKind::Mean) => match mean_state {
                        #[cfg(feature = "dtype-decimal")]
                        CumMeanState::Decimal { sum, count } => {
                            let (out, new_sum, new_count) =
                                cum_mean_decimal_with_init(s, false, Some(*sum), Some(*count))?;
                            *sum = new_sum;
                            *count = new_count;
                            out
                        },
                        #[cfg(feature = "dtype-decimal")]
                        CumMeanState::Float { .. } if matches!(s.dtype(), DataType::Decimal(_, _)) =>
                        {
                            polars_bail!(InvalidOperation: "incorrect state type; expected MeanState::Decimal")
                        },
                        CumMeanState::Float { sum, count, err } => {
                            let (out, new_sum, new_count, new_err) =
                                cum_mean_with_init(s, false, Some(*sum), Some(*count), Some(*err))?;
                            *sum = new_sum;
                            *count = new_count;
                            *err = new_err;
                            out
                        },
                    },
                    (CumAggState::Single(init), kind) => {
                        let out = match kind {
                            CumAggKind::Count => {
                                cum_count_with_init(s, false, init.extract().unwrap_or_default())
                            },
                            CumAggKind::Max => cum_max_with_init(s, false, &*init),
                            CumAggKind::Min => cum_min_with_init(s, false, &*init),
                            CumAggKind::Prod => cum_prod_with_init(s, false, &*init),
                            CumAggKind::Sum => cum_sum_with_init(s, false, &*init),
                            CumAggKind::Mean => polars_bail!(
                                InvalidOperation: "should be used with CumAggState::Mean"
                            ),
                        }?;
                        // Update state with the last non-null value.
                        let last_non_null_idx = if out.has_nulls() {
                            last_non_null(out.chunks().iter().map(|arr| arr.as_ref()), out.len())
                        } else {
                            Some(out.len() - 1)
                        };
                        if let Some(idx) = last_non_null_idx {
                            *init = out.get(idx).unwrap().into_static();
                        }
                        out
                    },
                    (state, kind) => polars_bail!(
                        InvalidOperation: "unexpected state {:?} for kind {:?} in CumAggNode", state, kind
                    ),
                };
                *m.df_mut() = out.into_column().into_frame();

                if send.send(m).await.is_err() {
                    break;
                }
            }

            Ok(())
        }));
    }
}
