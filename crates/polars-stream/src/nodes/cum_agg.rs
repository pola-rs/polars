use polars_core::prelude::{AnyValue, IntoColumn};
use polars_core::utils::last_non_null;
use polars_error::PolarsResult;
use polars_ops::series::{
    cum_count_with_init, cum_max_with_init, cum_min_with_init, cum_prod_with_init,
    cum_sum_with_init,
};

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::pipe::{RecvPort, SendPort};

pub struct CumAggNode {
    state: AnyValue<'static>,
    kind: CumAggKind,
}

#[derive(Debug, Clone, Copy)]
pub enum CumAggKind {
    Min,
    Max,
    Sum,
    Count,
    Prod,
}

impl CumAggNode {
    pub fn new(kind: CumAggKind) -> Self {
        Self {
            state: AnyValue::Null,
            kind,
        }
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
                let out = match self.kind {
                    CumAggKind::Min => cum_min_with_init(s, false, &self.state),
                    CumAggKind::Max => cum_max_with_init(s, false, &self.state),
                    CumAggKind::Sum => cum_sum_with_init(s, false, &self.state),
                    CumAggKind::Count => {
                        cum_count_with_init(s, false, self.state.extract().unwrap_or_default())
                    },
                    CumAggKind::Prod => cum_prod_with_init(s, false, &self.state),
                }?;

                // Find the last non-null value and set that as the state.
                let last_non_null_idx = if out.has_nulls() {
                    last_non_null(out.chunks().iter().map(|c| c.validity()), out.len())
                } else {
                    Some(out.len() - 1)
                };
                if let Some(idx) = last_non_null_idx {
                    self.state = out.get(idx).unwrap().into_static();
                }
                *m.df_mut() = out.into_column().into_frame();

                if send.send(m).await.is_err() {
                    break;
                }
            }

            Ok(())
        }));
    }
}
