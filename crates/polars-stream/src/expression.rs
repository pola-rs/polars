use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::Series;
use polars_error::PolarsResult;
use polars_expr::prelude::{ExecutionState, PhysicalExpr};

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

    pub async fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
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
}
