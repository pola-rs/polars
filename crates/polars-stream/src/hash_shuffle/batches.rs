use futures::future::BoxFuture;
use polars_async::executor::{self, TaskPriority};
use polars_core::runtime::ASYNC;
use polars_error::PolarsResult;
use polars_ooc::memory_manager;
use polars_utils::IdxSize;
use polars_utils::hashing::HashPartitioner;

use super::frame_with_keys::{FrameWithKeys, SpillFrameWithKeys};
use super::split::{SplitRows, split};
use crate::execute::StreamingExecutionState;

/// Orders frames (those still in memory first, then by input) and cuts them into batches of
/// about `batch_bytes`, at least one frame each.
pub fn make_batches(
    frames: Vec<(usize, SpillFrameWithKeys)>,
    batch_bytes: usize,
) -> Vec<Vec<(usize, SpillFrameWithKeys)>> {
    let mut frames: Vec<_> = frames
        .into_iter()
        .map(|(input, frame)| (!frame.is_in_memory(), input, frame))
        .collect();
    frames.sort_by_key(|(spilled, input, _)| (*spilled, *input));

    let mut batches = Vec::new();
    let mut batch = Vec::new();
    let mut batch_size = 0;
    for (_, input, frame) in frames {
        if !batch.is_empty() && batch_size + frame.estimated_size() > batch_bytes {
            batches.push(std::mem::take(&mut batch));
            batch_size = 0;
        }
        batch_size += frame.estimated_size();
        batch.push((input, frame));
    }
    if !batch.is_empty() {
        batches.push(batch);
    }
    batches
}

/// A frame of the current batch: loaded, prepared and split.
pub struct LoadedFrame {
    pub input: usize,
    pub frame: FrameWithKeys,
    /// `None`: all rows, in partition 0 (one partition, no mask).
    pub split: Option<SplitRows>,
}

impl LoadedFrame {
    /// The rows of partition `p`; `None` for all rows.
    pub fn partition_rows(&self, p: usize) -> Option<&[IdxSize]> {
        self.split.as_ref().map(|split| split.partition(p))
    }
}

/// Runs once on each loaded frame of the given input, before it is split.
pub type PrepareFrame<'p> =
    dyn Fn(usize, FrameWithKeys) -> BoxFuture<'p, PolarsResult<FrameWithKeys>> + Sync + 'p;

/// Calls `build` on every partition once per batch, with the frames of that batch loaded. While
/// low-priority tasks build the partitions, high-priority tasks load, prepare and split the next
/// batch (each frame once) and free the previous one.
pub fn build_from_batches<S: Send>(
    batches: Vec<Vec<(usize, SpillFrameWithKeys)>>,
    partitioner: &HashPartitioner,
    partitions: &mut [S],
    prepare: Option<&PrepareFrame<'_>>,
    build: &(dyn Fn(&mut S, &[LoadedFrame]) -> PolarsResult<()> + Sync),
    state: &StreamingExecutionState,
) -> PolarsResult<()> {
    let mut current = Vec::new();
    let mut to_free = Vec::new();
    for to_load in batches {
        let loaded = build_and_load(
            to_load,
            std::mem::take(&mut to_free),
            &current,
            partitions,
            partitioner,
            prepare,
            build,
            state,
        )?;
        to_free = std::mem::replace(&mut current, loaded);
    }
    // Build from the last batch while freeing the one before it, then free the last batch.
    build_and_load(
        Vec::new(),
        to_free,
        &current,
        partitions,
        partitioner,
        prepare,
        build,
        state,
    )?;
    build_and_load(
        Vec::new(),
        current,
        &[],
        partitions,
        partitioner,
        prepare,
        build,
        state,
    )?;
    Ok(())
}

/// One task scope with a task per item: high-priority tasks drop the frames `to_free` and load,
/// prepare and split the frames `to_load`, while low-priority tasks build every partition from
/// `batch`, unless it is empty. Returns the loaded frames.
#[allow(clippy::too_many_arguments)]
fn build_and_load<S: Send>(
    to_load: Vec<(usize, SpillFrameWithKeys)>,
    to_free: Vec<LoadedFrame>,
    batch: &[LoadedFrame],
    partitions: &mut [S],
    partitioner: &HashPartitioner,
    prepare: Option<&PrepareFrame<'_>>,
    build: &(dyn Fn(&mut S, &[LoadedFrame]) -> PolarsResult<()> + Sync),
    state: &StreamingExecutionState,
) -> PolarsResult<Vec<LoadedFrame>> {
    executor::task_scope(state.task_metrics(), |scope| {
        let free_handles: Vec<_> = to_free
            .into_iter()
            .map(|frame| scope.spawn_task(TaskPriority::High, async move { drop(frame) }))
            .collect();
        let load_handles: Vec<_> = to_load
            .into_iter()
            .map(|(input, frame)| {
                scope.spawn_task(TaskPriority::High, async move {
                    let loaded = load_frame(input, frame, partitioner, prepare).await?;
                    memory_manager().spill().await;
                    PolarsResult::Ok(loaded)
                })
            })
            .collect();
        let build_handles: Vec<_> = if batch.is_empty() {
            Vec::new()
        } else {
            partitions
                .iter_mut()
                .map(|partition| {
                    scope.spawn_task(TaskPriority::Low, async move {
                        build(partition, batch)?;
                        memory_manager().spill().await;
                        PolarsResult::Ok(())
                    })
                })
                .collect()
        };

        ASYNC.block_in_place_on(async {
            for handle in free_handles {
                handle.await;
            }
            let mut loaded = Vec::with_capacity(load_handles.len());
            for handle in load_handles {
                loaded.push(handle.await?);
            }
            for handle in build_handles {
                handle.await?;
            }
            PolarsResult::Ok(loaded)
        })
    })
}

async fn load_frame(
    input: usize,
    frame: SpillFrameWithKeys,
    partitioner: &HashPartitioner,
    prepare: Option<&PrepareFrame<'_>>,
) -> PolarsResult<LoadedFrame> {
    let mut frame = frame.load().await;
    if let Some(prepare) = prepare {
        frame = prepare(input, frame).await?;
    }
    let split = split(&frame.keys, frame.mask.as_ref(), partitioner);
    Ok(LoadedFrame {
        input,
        frame,
        split,
    })
}
