use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, GroupPositions};
use polars_error::{PolarsResult, polars_bail};
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
                .spawn_blocking(move || phys_expr.evaluate(&df, &state))
                .await
                .unwrap()
        } else {
            self.inner.evaluate(df, state)
        }
    }

    /// Broadcasts unit-length results to df height. Errors if length does not match df height otherwise.
    pub async fn evaluate_preserve_len_broadcast(
        &self,
        df: &DataFrame,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        let mut c = self.evaluate(df, state).await?;

        if c.len() != df.height() {
            if c.len() != 1 {
                polars_bail!(
                    ShapeMismatch:
                    "expression result length {} does not match df height {}",
                    c.len(), df.height(),
                )
            }

            c = c.new_from_index(0, df.height());
        }

        Ok(c)
    }

    pub fn evaluate_blocking(
        &self,
        df: &DataFrame,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        self.inner.evaluate(df, state)
    }

    pub async fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        if self.reentrant {
            let state = state.split();
            // @NOTE: Clones only the Arc, relatively cheap.
            let groups = <GroupPositions as Clone>::clone(groups);
            let phys_expr = self.inner.clone();
            let df = df.clone();
            polars_io::pl_async::get_runtime()
                .spawn_blocking(move || {
                    Ok(phys_expr
                        .evaluate_on_groups(&df, &groups, &state)?
                        .into_static())
                })
                .await
                .unwrap()
        } else {
            self.inner.evaluate_on_groups(df, groups, state)
        }
    }
}
