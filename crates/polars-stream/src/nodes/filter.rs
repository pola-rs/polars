use std::sync::Arc;

use polars_error::{polars_err, PolarsResult};
use polars_expr::prelude::PhysicalExpr;
use polars_expr::state::ExecutionState;

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskScope};
use crate::async_primitives::pipe::{Receiver, Sender};
use crate::morsel::Morsel;

pub struct FilterNode {
    predicate: Arc<dyn PhysicalExpr>,
}

impl FilterNode {
    pub fn new(predicate: Arc<dyn PhysicalExpr>) -> Self {
        Self { predicate }
    }
}

impl ComputeNode for FilterNode {
    fn spawn<'env, 's>(
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        _pipeline: usize,
        recv: Vec<Receiver<Morsel>>,
        send: Vec<Sender<Morsel>>,
        state: &'s ExecutionState,
    ) -> JoinHandle<PolarsResult<()>> {
        let [mut recv] = <[_; 1]>::try_from(recv).ok().unwrap();
        let [mut send] = <[_; 1]>::try_from(send).ok().unwrap();

        scope.spawn_task(true, async move {
            while let Ok(morsel) = recv.recv().await {
                let morsel = morsel.try_map(|df| {
                    let mask = self.predicate.evaluate(&df, state)?;
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
        })
    }
}
