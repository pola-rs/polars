use std::sync::Arc;

use futures::{StreamExt as _, TryStreamExt as _};
use polars_core::frame::DataFrame;
use polars_core::prelude::sort::arg_sort;
use polars_core::prelude::{Column, IdxArr, SortMultipleOptions};
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;

use crate::async_executor::TaskPriority;
use crate::async_primitives::opt_spawned_future::parallelize_first_to_local;
use crate::expression::StreamExpr;

#[derive(Clone)]
pub struct ArgSortBy {
    pub by: Arc<[StreamExpr]>,
    pub sort_options: SortMultipleOptions,
    pub in_memory_exec_state: Arc<ExecutionState>,
}

impl ArgSortBy {
    pub async fn arg_sort_by_par(self, df: &Arc<DataFrame>) -> PolarsResult<IdxArr> {
        let ArgSortBy {
            by,
            sort_options,
            in_memory_exec_state,
        } = self;

        let sort_by_cols: Vec<Column> = futures::stream::iter(parallelize_first_to_local(
            TaskPriority::Low,
            (0..by.len()).map(|i| {
                let df = Arc::clone(df);
                let by = Arc::clone(&by);
                let in_memory_exec_state = Arc::clone(&in_memory_exec_state);

                async move {
                    by[i]
                        .evaluate(&df, in_memory_exec_state.as_ref())
                        .await
                        .map(|c| c.rechunk())
                }
            }),
        ))
        .then(|x| x)
        .try_collect()
        .await?;

        Ok(arg_sort(&sort_by_cols, sort_options)?
            .downcast_as_array()
            .clone())
    }
}
