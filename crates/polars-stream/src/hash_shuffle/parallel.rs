use crossbeam_queue::ArrayQueue;
use polars_async::executor::{self, TaskPriority};
use polars_core::runtime::ASYNC;
use polars_error::PolarsResult;
use polars_ooc::memory_manager;

use crate::execute::StreamingExecutionState;

/// Calls `f` on every item, with up to `num_pipelines` tasks that take the items in order.
pub(crate) fn for_each_parallel<T: Send>(
    items: impl ExactSizeIterator<Item = T>,
    f: &(dyn Fn(T) -> PolarsResult<()> + Sync),
    state: &StreamingExecutionState,
) -> PolarsResult<()> {
    if items.len() == 0 {
        return Ok(());
    }
    let num_tasks = state.num_pipelines.min(items.len());
    let queue = queue_of(items);
    executor::task_scope(state.task_metrics(), |scope| {
        let handles: Vec<_> = (0..num_tasks)
            .map(|_| scope.spawn_task(TaskPriority::High, take_items(&queue, f)))
            .collect();
        ASYNC.block_in_place_on(async {
            for handle in handles {
                handle.await?;
            }
            PolarsResult::Ok(())
        })
    })
}

/// Calls `f` on items taken from `queue` until it is empty, letting the memory manager spill
/// after each item.
pub(crate) async fn take_items<T>(
    queue: &ArrayQueue<T>,
    f: impl Fn(T) -> PolarsResult<()>,
) -> PolarsResult<()> {
    while let Some(item) = queue.pop() {
        f(item)?;
        memory_manager().spill().await;
    }
    Ok(())
}

pub(crate) fn queue_of<T>(items: impl ExactSizeIterator<Item = T>) -> ArrayQueue<T> {
    let queue = ArrayQueue::new(items.len().max(1));
    for item in items {
        queue.push(item).ok().unwrap();
    }
    queue
}
