use polars_error::polars_err;

use super::compute_node_prelude::*;
use crate::expression::StreamExpr;

pub struct FilterNode {
    predicate: StreamExpr,
}

impl FilterNode {
    pub fn new(predicate: StreamExpr) -> Self {
        Self { predicate }
    }
}

impl ComputeNode for FilterNode {
    fn name(&self) -> &str {
        "filter"
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
        let receivers = recv_ports[0].take().unwrap().parallel();
        let senders = send_ports[0].take().unwrap().parallel();

        for (mut recv, mut send) in receivers.into_iter().zip(senders) {
            let slf = &*self;
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                while let Ok(morsel) = recv.recv().await {

                    let morsel = morsel.async_try_map(|df| async move {
                        let mask = slf.predicate.evaluate(&df, &state.in_memory_exec_state).await?;
                        let mask = mask.bool().map_err(|_| {
                            polars_err!(
                                ComputeError: "filter predicate must be of type `Boolean`, got `{}`", mask.dtype()
                            )
                        })?;

                        // We already parallelize, call the sequential filter.
                        df._filter_seq(mask)
                    }).await?;

                    if morsel.df().height() == 0 {
                        continue;
                    }

                    if send.send(morsel).await.is_err() {
                        break;
                    }
                }

                Ok(())
            }));
        }
    }
}
