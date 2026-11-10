use futures::future::BoxFuture;
use polars_async::executor::{self, TaskPriority};
use polars_core::runtime::ASYNC;
use polars_error::PolarsResult;
use polars_ooc::memory_manager;
use polars_utils::IdxSize;
use polars_utils::hashing::HashPartitioner;

use super::frame_with_keys::{FrameWithKeys, SpillFrameWithKeys};
use super::split::{NullRows, SplitRows, split};
use crate::execute::StreamingExecutionState;

/// A frame of a batch, with its input and its index in the frames the batches were made from.
pub struct BatchFrame {
    index: usize,
    input: usize,
    frame: SpillFrameWithKeys,
}

/// Orders frames (those still in memory first, then by input) and cuts them into batches of
/// about `batch_bytes`, at least one frame each.
pub fn make_batches(
    frames: Vec<(usize, SpillFrameWithKeys)>,
    batch_bytes: usize,
) -> Vec<Vec<BatchFrame>> {
    let mut frames: Vec<_> = frames
        .into_iter()
        .enumerate()
        .map(|(index, (input, frame))| {
            let spilled = !frame.is_in_memory();
            (
                spilled,
                BatchFrame {
                    index,
                    input,
                    frame,
                },
            )
        })
        .collect();
    frames.sort_by_key(|(spilled, f)| (*spilled, f.input));
    cut_batches(frames.into_iter().map(|(_, f)| f), batch_bytes)
}

/// Cuts frames into batches of about `batch_bytes`, at least one frame each, keeping their order.
pub fn make_ordered_batches(
    frames: Vec<(usize, SpillFrameWithKeys)>,
    batch_bytes: usize,
) -> Vec<Vec<BatchFrame>> {
    let frames = frames
        .into_iter()
        .enumerate()
        .map(|(index, (input, frame))| BatchFrame {
            index,
            input,
            frame,
        });
    cut_batches(frames, batch_bytes)
}

fn cut_batches(
    frames: impl Iterator<Item = BatchFrame>,
    batch_bytes: usize,
) -> Vec<Vec<BatchFrame>> {
    let mut batches = Vec::new();
    let mut batch = Vec::new();
    let mut batch_size = 0;
    for f in frames {
        let size = f.frame.estimated_size();
        if !batch.is_empty() && batch_size + size > batch_bytes {
            batches.push(std::mem::take(&mut batch));
            batch_size = 0;
        }
        batch_size += size;
        batch.push(f);
    }
    if !batch.is_empty() {
        batches.push(batch);
    }
    batches
}

/// A frame of the current batch: loaded and split.
pub struct LoadedFrame {
    /// The index of the frame in the frames the batches were made from.
    pub index: usize,
    pub input: usize,
    pub frame: FrameWithKeys,
    /// `None`: all rows, in partition 0 (one partition, no mask).
    pub split: Option<SplitRows>,
}

impl LoadedFrame {
    /// The rows of partition `p`; `None` for all rows, which are then in partition 0.
    pub fn partition_rows(&self, p: usize) -> Option<&[IdxSize]> {
        match &self.split {
            Some(split) => Some(split.partition(p)),
            None => (p != 0).then_some(&[]),
        }
    }
}

/// Adds the rows of a batch to one partition.
pub type BuildPartition<'b, S> = dyn for<'a> Fn(
        &'a mut S,
        &'a [LoadedFrame],
        &'a StreamingExecutionState,
    ) -> BoxFuture<'a, PolarsResult<()>>
    + Sync
    + 'b;

/// Calls `build` on every partition once per batch, with the frames of that batch loaded. While
/// low-priority tasks build the partitions, high-priority tasks load and split the next batch
/// (each frame once) and free the previous one. Rows without a hash go where `nulls` says.
pub fn build_from_batches<S: Send>(
    batches: Vec<Vec<BatchFrame>>,
    partitioner: &HashPartitioner,
    partitions: &mut [S],
    nulls: NullRows,
    build: &BuildPartition<'_, S>,
    state: &StreamingExecutionState,
) -> PolarsResult<()> {
    // One task scope with a task per item: high-priority tasks drop the frames `to_free` and load
    // and split the frames `to_load`, while low-priority tasks build every partition from
    // `batch`, unless it is empty. Returns the loaded frames.
    let build_and_load = |to_load: Vec<BatchFrame>,
                          to_free: Vec<LoadedFrame>,
                          batch: &[LoadedFrame],
                          partitions: &mut [S]|
     -> PolarsResult<Vec<LoadedFrame>> {
        executor::task_scope(state.task_metrics(), |scope| {
            let free_handles: Vec<_> = to_free
                .into_iter()
                .map(|frame| scope.spawn_task(TaskPriority::High, async move { drop(frame) }))
                .collect();
            let load_handles: Vec<_> = to_load
                .into_iter()
                .map(|frame| {
                    scope.spawn_task(TaskPriority::High, async move {
                        let loaded = load_frame(frame, partitioner, nulls).await?;
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
                            build(partition, batch, state).await?;
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
    };

    let mut current = Vec::new();
    let mut to_free = Vec::new();
    for to_load in batches {
        let loaded = build_and_load(to_load, std::mem::take(&mut to_free), &current, partitions)?;
        to_free = std::mem::replace(&mut current, loaded);
    }
    // Build from the last batch while freeing the one before it, then free the last batch.
    build_and_load(Vec::new(), to_free, &current, partitions)?;
    build_and_load(Vec::new(), current, &[], partitions)?;
    Ok(())
}

async fn load_frame(
    frame: BatchFrame,
    partitioner: &HashPartitioner,
    nulls: NullRows,
) -> PolarsResult<LoadedFrame> {
    let BatchFrame {
        index,
        input,
        frame,
    } = frame;
    let frame = frame.load().await;
    let split = split(&frame.keys, frame.mask.as_ref(), partitioner, nulls);
    Ok(LoadedFrame {
        index,
        input,
        frame,
        split,
    })
}
