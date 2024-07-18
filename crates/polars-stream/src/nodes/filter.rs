use std::sync::Arc;

use polars_error::polars_err;
use polars_expr::prelude::PhysicalExpr;

use super::compute_node_prelude::*;

pub struct FilterNode {
    predicate: Arc<dyn PhysicalExpr>,
}

impl FilterNode {
    pub fn new(predicate: Arc<dyn PhysicalExpr>) -> Self {
        Self { predicate }
    }
}

impl ComputeNode for FilterNode {
    fn name(&self) -> &str {
        "filter"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) {
        assert!(recv.len() == 1 && send.len() == 1);
        recv.swap_with_slice(send);
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv: &mut [Option<RecvPort<'_>>],
        send: &mut [Option<SendPort<'_>>],
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv.len() == 1 && send.len() == 1);
        let receivers = recv[0].take().unwrap().parallel();
        let senders = send[0].take().unwrap().parallel();

        for (mut recv, mut send) in receivers.into_iter().zip(senders) {
            let slf = &*self;
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                while let Ok(morsel) = recv.recv().await {
                    let morsel = morsel.try_map(|df| {
                        let mask = slf.predicate.evaluate(&df, state)?;
                        let mask = mask.bool().map_err(|_| {
                            polars_err!(
                                ComputeError: "filter predicate must be of type `Boolean`, got `{}`", mask.dtype()
                            )
                        })?;

                        // We already parallelize, call the sequential filter.
                        df._filter_seq(mask)
                    })?;

                    if morsel.df().is_empty() {
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
