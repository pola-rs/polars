use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::schema::Schema;
use polars_core::series::Series;
use polars_error::PolarsResult;
use polars_expr::prelude::PhysicalExpr;
use polars_expr::state::ExecutionState;

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskScope};
use crate::async_primitives::pipe::{Receiver, Sender};
use crate::morsel::Morsel;

pub struct SelectNode {
    selectors: Vec<Arc<dyn PhysicalExpr>>,
    schema: Arc<Schema>,
    extend_original: bool,
}

impl SelectNode {
    pub fn new(
        selectors: Vec<Arc<dyn PhysicalExpr>>,
        schema: Arc<Schema>,
        extend_original: bool,
    ) -> Self {
        Self {
            selectors,
            schema,
            extend_original,
        }
    }
}

impl ComputeNode for SelectNode {
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
                    // Select columns.
                    let mut selected: Vec<Series> = self
                        .selectors
                        .iter()
                        .map(|s| s.evaluate(&df, state))
                        .collect::<PolarsResult<_>>()?;

                    // Extend or create new dataframe.
                    let ret = if self.extend_original {
                        let mut out = df.clone();
                        out._add_columns(selected, &self.schema)?;
                        out
                    } else {
                        // Broadcast scalars.
                        let max_non_unit_length = selected
                            .iter()
                            .map(|s| s.len())
                            .filter(|l| *l != 1)
                            .max()
                            .unwrap_or(1);
                        for s in &mut selected {
                            if s.len() != max_non_unit_length {
                                assert!(s.len() == 1, "got series of incompatible lengths");
                                *s = s.new_from_index(0, max_non_unit_length);
                            }
                        }
                        unsafe { DataFrame::new_no_checks(selected) }
                    };

                    PolarsResult::Ok(ret)
                })?;

                if send.send(morsel).await.is_err() {
                    break;
                }
            }

            Ok(())
        })
    }
}
