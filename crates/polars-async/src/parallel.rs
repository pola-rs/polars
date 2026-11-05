use std::future::Future;

use polars_error::PolarsResult;

use crate::ASYNC;
use crate::executor::{self, TaskMetricAggregator, TaskPriority};

/// Evaluates `f` on every item, each in its own task. Blocks until all items are done.
pub fn for_each_parallel<T, F: Future<Output = PolarsResult<()>> + Send>(
    items: impl IntoIterator<Item = T>,
    task_metrics: Option<&TaskMetricAggregator>,
    f: &(impl Fn(T) -> F + Sync),
) -> PolarsResult<()> {
    executor::task_scope(task_metrics, |scope| {
        let handles: Vec<_> = items
            .into_iter()
            .map(|item| scope.spawn_task(TaskPriority::High, f(item)))
            .collect();
        ASYNC.block_in_place_on(async {
            for handle in handles {
                handle.await?;
            }
            Ok(())
        })
    })
}
