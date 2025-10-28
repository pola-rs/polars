use std::sync::Arc;

use polars_core::POOL;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, GroupPositions};
use polars_error::PolarsResult;
use polars_expr::prelude::{AggregationContext, ExecutionState, PhysicalExpr};

#[derive(Clone)]
pub struct StreamExpr {
    inner: Arc<dyn PhysicalExpr>,
    // Whether the expression can be re-entering the engine (e.g. a function use the lazy api
    // within that function)
    reentrant: bool,
}

impl StreamExpr {
    pub fn new(phys_expr: Arc<dyn PhysicalExpr>, reentrant: bool) -> Self {
        Self {
            inner: phys_expr,
            reentrant,
        }
    }

    pub async fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        if self.reentrant {
            let state = state.clone();
            let phys_expr = self.inner.clone();
            let df = df.clone();
            polars_io::pl_async::get_runtime()
                .spawn_blocking(move || POOL.without_threading(|| phys_expr.evaluate(&df, &state)))
                .await
                .unwrap()
        } else {
            POOL.without_threading(|| self.inner.evaluate(df, state))
        }
    }

    pub fn evaluate_blocking(
        &self,
        df: &DataFrame,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        POOL.without_threading(|| self.inner.evaluate(df, state))
    }

    pub async fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        if self.reentrant {
            let state = state.split();
            let groups = <GroupPositions as Clone>::clone(groups);
            let phys_expr = self.inner.clone();
            let df = df.clone();
            polars_io::pl_async::get_runtime()
                .spawn_blocking(move || {
                    POOL.without_threading(|| {
                        Ok(phys_expr
                            .evaluate_on_groups(&df, &groups, &state)?
                            .into_static())
                    })
                })
                .await
                .unwrap()
        } else {
            POOL.without_threading(|| self.inner.evaluate_on_groups(df, groups, state))
        }
    }
}
