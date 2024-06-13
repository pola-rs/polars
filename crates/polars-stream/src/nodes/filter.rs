use std::sync::Arc;

use polars_error::PolarsResult;
use polars_expr::prelude::PhysicalExpr;
use polars_expr::state::ExecutionState;

use crate::async_primitives::pipe::{Receiver, Sender};
use crate::morsel::Morsel;
use super::ComputeNode;

pub struct FilterNode {
    predicate: Arc<dyn PhysicalExpr>,
}

impl ComputeNode for FilterNode {
    async fn process(
        &self,
        recv: Vec<Receiver<Morsel>>,
        send: Vec<Sender<Morsel>>,
        state: &ExecutionState,
    ) -> PolarsResult<()> {
        let [mut recv] = <[_; 1]>::try_from(recv).ok().unwrap();
        let [mut send] = <[_; 1]>::try_from(send).ok().unwrap();
        while let Ok(morsel) = recv.recv().await {
            let morsel = morsel.try_map(|df| {
                let mask = self.predicate.evaluate(&df, state)?;
                df.filter(mask.bool().unwrap())
            })?;

            if morsel.df().is_empty() {
                continue;
            }

            if let Err(_) = send.send(morsel).await {
                break;
            }
        }

        Ok(())
    }
}
