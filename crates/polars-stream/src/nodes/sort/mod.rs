mod flush;
mod sample;
mod split_tree;
mod tuning;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use polars_arrow::array::builder::ShareStrategy;
use polars_async::executor::{self, JoinHandle, TaskMetricAggregator, TaskPriority, TaskScope};
use polars_core::chunked_array::ops::sort::options::SortMultipleOptions;
use polars_core::datatypes::DataType;
use polars_core::frame::DataFrame;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::runtime::{ASYNC, RAYON};
use polars_core::schema::Schema;
use polars_core::series::{IntoSeries, Series};
use polars_core::utils::{accumulate_dataframes_vertical_unchecked, slice_offsets};
use polars_error::PolarsResult;
use polars_ooc::{
    LeastRecentSpillContext, MostRecentSpillContext, ParameterFreeSpillContext, SpillFrame,
};
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use self::flush::{Flush, PendingBucket};
use self::sample::{KeySample, split_keys};
use self::split_tree::BucketClassifier;
use self::tuning::SortTuning;
use super::ComputeNode;
use super::in_memory_source::InMemorySourceNode;
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, get_ideal_morsel_size};
use crate::pipe::{RecvPort, SendPort};
use crate::utils::in_memory_linearize::linearize;

struct BufferAndSample {
    /// The buffered morsels, one list per pipe per phase.
    morsels: Mutex<Vec<Vec<Morsel>>>,
    /// The key sample of each pipe, kept across phases.
    samples: Vec<KeySample>,
    bytes: AtomicU64,
    spill_ctx: LeastRecentSpillContext,
}

enum SortState {
    BufferAndSample(BufferAndSample),
    Source(InMemorySourceNode),
    Flush(Flush),
    Done,
}

/// Sorts its input on a single directly sortable key column.
pub struct SortNode {
    state: SortState,
    key: PlSmallStr,
    key_dtype: DataType,
    input_schema: Arc<Schema>,
    slice: Option<(i64, usize)>,
    limit: Option<IdxSize>,
    sort_options: SortMultipleOptions,
    tuning: SortTuning,
}

impl SortNode {
    pub fn new(
        key: PlSmallStr,
        input_schema: Arc<Schema>,
        slice: Option<(i64, usize)>,
        mut sort_options: SortMultipleOptions,
        task_metrics: Option<Arc<TaskMetricAggregator>>,
    ) -> Self {
        assert_eq!(sort_options.descending.len(), 1);
        assert_eq!(sort_options.nulls_last.len(), 1);
        let limit = sort_options.limit.take();
        let key_dtype = match input_schema.get(&key).unwrap().to_physical() {
            DataType::String => DataType::Binary,
            dt => dt,
        };

        Self {
            state: SortState::BufferAndSample(BufferAndSample {
                morsels: Mutex::default(),
                samples: Vec::new(),
                bytes: AtomicU64::new(0),
                spill_ctx: LeastRecentSpillContext::new("sort".into(), task_metrics),
            }),
            key,
            key_dtype,
            input_schema,
            slice,
            limit,
            sort_options,
            tuning: SortTuning::from_env(),
        }
    }

    /// The `(offset, len)` rows of the sorted output to emit. The limit applies
    /// before the slice.
    fn output_range(&self, height: usize) -> (usize, usize) {
        let height = self
            .limit
            .map_or(height, |limit| height.min(limit as usize));
        match self.slice {
            Some((offset, len)) => slice_offsets(offset, len, height),
            None => (0, height),
        }
    }

    fn sort_in_memory(&self, df: DataFrame) -> PolarsResult<DataFrame> {
        let range = self.output_range(df.height());
        sort_range(df, &self.key, self.sort_options.clone(), range)
    }

    /// Turns the buffered input into the state that emits the sorted output.
    fn finish_buffer(
        &self,
        buffer: BufferAndSample,
        state: &StreamingExecutionState,
    ) -> PolarsResult<SortState> {
        let BufferAndSample {
            morsels,
            samples,
            bytes,
            ..
        } = buffer;
        let frames: Vec<SpillFrame> = linearize(morsels.into_inner())
            .into_iter()
            .map(Morsel::into_sf)
            .collect();

        let total_bytes = bytes.load(Ordering::Relaxed);
        let threshold = self.tuning.partition_threshold();
        if polars_core::config::verbose() {
            eprintln!("sort: buffered {total_bytes} bytes, partition threshold {threshold} bytes");
        }

        if total_bytes > threshold && !frames.is_empty() {
            let start = std::time::Instant::now();
            let flush = self.partition(frames, samples, total_bytes, state)?;
            if polars_core::config::verbose() {
                eprintln!("sort: partitioned in {:.3}s", start.elapsed().as_secs_f64());
            }
            return Ok(SortState::Flush(flush));
        }

        let df = if frames.is_empty() {
            DataFrame::empty_with_schema(&self.input_schema)
        } else {
            accumulate_dataframes_vertical_unchecked(
                frames.into_iter().map(SpillFrame::into_df_blocking),
            )
        };
        Ok(SortState::Source(InMemorySourceNode::new(
            Arc::new(self.sort_in_memory(df)?),
            MorselSeq::new(0),
        )))
    }

    /// Splits the buffered input into key-range buckets, restricted to the
    /// requested output range, and returns the state that emits them in order.
    fn partition(
        &self,
        frames: Vec<SpillFrame>,
        samples: Vec<KeySample>,
        total_bytes: u64,
        state: &StreamingExecutionState,
    ) -> PolarsResult<Flush> {
        let total_rows: usize = frames.iter().map(SpillFrame::height).sum();
        let descending = self.sort_options.descending[0];
        let maintain_order = self.sort_options.maintain_order;

        let stride = samples.iter().map(KeySample::stride).max().unwrap_or(1);
        let sample_parts: Vec<_> = samples
            .into_iter()
            .flat_map(|s| s.into_parts_with_stride(stride))
            .collect();
        let sample_len: usize = sample_parts.iter().map(|s| s.len() - s.null_count()).sum();
        let num_tasks = state.num_pipelines.max(1);
        let b = self.tuning.bucket_count(total_bytes, sample_len, num_tasks);
        let split = split_keys(sample_parts, &self.key_dtype, descending, b)?;
        let classifier = BucketClassifier::new(&split, descending);
        let num_buckets = classifier.num_buckets();
        let flush_rows =
            self.tuning
                .flush_rows(total_bytes, total_rows, num_tasks * (num_buckets + 1));
        let spill_ctx =
            MostRecentSpillContext::new("sort-partition".into(), state.task_metrics.clone());

        if polars_core::config::verbose() {
            eprintln!(
                "sort: partitioning {total_rows} rows into {num_buckets} buckets, flushing builders every {flush_rows} rows"
            );
        }

        // Contiguous frame ranges per task keep the input order within each
        // bucket; without maintain_order an interleaved split prefetches better.
        let num_frames = frames.len();
        let mut frames_per_task: Vec<Vec<SpillFrame>> =
            (0..num_tasks).map(|_| Vec::new()).collect();
        if maintain_order {
            let mut frames = frames.into_iter();
            for (t, task_frames) in frames_per_task.iter_mut().enumerate() {
                let len = (t + 1) * num_frames / num_tasks - t * num_frames / num_tasks;
                task_frames.extend(frames.by_ref().take(len));
            }
        } else {
            for (i, frame) in frames.into_iter().enumerate() {
                frames_per_task[i % num_tasks].push(frame);
            }
        }

        let out_per_task = executor::task_scope(state.task_metrics(), |scope| {
            let mut join_handles = Vec::new();
            for task_frames in frames_per_task {
                let classifier = &classifier;
                let input_schema = &self.input_schema;
                let key = &self.key;
                let spill_ctx = &spill_ctx;
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    partition_frames(
                        task_frames,
                        classifier,
                        input_schema,
                        key,
                        flush_rows,
                        maintain_order,
                        spill_ctx,
                    )
                    .await
                }));
            }

            ASYNC.block_in_place_on(async move {
                let mut out_per_task = Vec::with_capacity(join_handles.len());
                for handle in join_handles {
                    out_per_task.push(handle.await?);
                }
                PolarsResult::Ok(out_per_task)
            })
        })?;

        let mut buckets: Vec<Vec<SpillFrame>> = (0..num_buckets + 1).map(|_| Vec::new()).collect();
        for task_out in out_per_task {
            for (bucket, frames) in buckets.iter_mut().zip(task_out) {
                bucket.extend(frames);
            }
        }

        let (range_start, range_len) = self.output_range(total_rows);
        let range_end = range_start + range_len;

        // The null bucket is emitted on the side the nulls sort to.
        let null_bucket = num_buckets;
        let order: Vec<usize> = if self.sort_options.nulls_last[0] {
            (0..num_buckets).chain([null_bucket]).collect()
        } else {
            [null_bucket].into_iter().chain(0..num_buckets).collect()
        };

        let needs_sort = classifier.needs_sort();
        let mut pending = Vec::new();
        let mut dropped = Vec::new();
        let mut row = 0;
        for b in order {
            let frames = core::mem::take(&mut buckets[b]);
            // A bucket of a single key value can be arbitrarily large, so it
            // is emitted one frame at a time.
            let units: Vec<Vec<SpillFrame>> = if needs_sort[b] {
                vec![frames]
            } else {
                frames.into_iter().map(|frame| vec![frame]).collect()
            };
            for frames in units {
                let height: usize = frames.iter().map(SpillFrame::height).sum();
                let lo = range_start.max(row);
                let hi = range_end.min(row + height);
                if lo < hi {
                    pending.push(PendingBucket {
                        frames,
                        local: (lo - row, hi - lo),
                        needs_sort: needs_sort[b],
                    });
                } else {
                    dropped.extend(frames);
                }
                row += height;
            }
        }
        drop_parallel(dropped);

        let bucket_bytes = total_bytes / (num_buckets as u64 + 1);
        Ok(Flush::new(
            pending,
            get_ideal_morsel_size().max(1),
            self.tuning.flush_ahead(bucket_bytes, num_tasks),
            spill_ctx,
        ))
    }
}

impl ComputeNode for SortNode {
    fn name(&self) -> &str {
        "sort"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        // State transitions.
        match &mut self.state {
            // If the output doesn't want any more data, transition to being done.
            _ if send[0] == PortState::Done => {
                if let SortState::BufferAndSample(buffer) =
                    core::mem::replace(&mut self.state, SortState::Done)
                {
                    drop_parallel(buffer.morsels.into_inner().into_iter().flatten());
                }
            },
            // The input is done, sort what we buffered.
            SortState::BufferAndSample(_) if recv[0] == PortState::Done => {
                let SortState::BufferAndSample(buffer) =
                    core::mem::replace(&mut self.state, SortState::Done)
                else {
                    unreachable!()
                };
                self.state = self.finish_buffer(buffer, state)?;
            },
            // Defer to source node implementation.
            SortState::Source(src) => {
                src.update_state(&mut [], send, state)?;
                if send[0] == PortState::Done {
                    self.state = SortState::Done;
                }
            },
            SortState::Flush(flush) if flush.is_finished() => self.state = SortState::Done,
            // Nothing to change.
            SortState::Done | SortState::BufferAndSample(_) | SortState::Flush(_) => {},
        }

        // Communicate our state.
        match &mut self.state {
            SortState::BufferAndSample(_) => {
                recv[0] = PortState::Ready;
                send[0] = PortState::Blocked;
            },
            SortState::Source(_) | SortState::Flush(_) => {
                recv[0] = PortState::Done;
                send[0] = PortState::Ready;
            },
            SortState::Done => {
                recv[0] = PortState::Done;
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        matches!(self.state, SortState::BufferAndSample(_))
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);
        let Self {
            state: sort_state,
            key,
            sort_options,
            tuning,
            ..
        } = self;

        match sort_state {
            SortState::BufferAndSample(buffer) => {
                assert!(send_ports[0].is_none());
                let receivers = recv_ports[0].take().unwrap().parallel();
                let BufferAndSample {
                    morsels,
                    samples,
                    bytes,
                    spill_ctx,
                } = buffer;
                let (morsels, bytes, spill_ctx) = (&*morsels, &*bytes, &*spill_ctx);
                let num_samples = samples.len().max(receivers.len());
                samples.resize_with(num_samples, || KeySample::new(tuning.sample_rows));

                for (mut recv, sample) in receivers.into_iter().zip(samples.iter_mut()) {
                    let key = &*key;
                    join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                        let mut pipe_morsels = Vec::new();
                        while let Ok(mut morsel) = recv.recv().await {
                            morsel.take_consume_token();
                            {
                                let df = morsel.sf().get().await;
                                bytes.fetch_add(df.estimated_size(false) as u64, Ordering::Relaxed);
                                let keys = key_series(&df, key)?;
                                sample.add(&keys);
                            }
                            spill_ctx.register(morsel.sf()).await;
                            pipe_morsels.push(morsel);
                        }

                        morsels.lock().push(pipe_morsels);
                        Ok(())
                    }));
                }
            },
            SortState::Source(source) => {
                assert!(recv_ports[0].is_none());
                source.spawn(scope, &mut [], send_ports, state, join_handles);
            },
            SortState::Flush(flush) => {
                assert!(recv_ports[0].is_none());
                let senders = send_ports[0].take().unwrap().parallel();
                flush.spawn(scope, senders, key, sort_options, join_handles);
            },
            SortState::Done => unreachable!(),
        }
    }
}

/// The key column in the representation that is sampled and classified: the
/// physical representation, with `String` as `Binary`.
fn key_series(df: &DataFrame, key: &str) -> PolarsResult<Series> {
    let keys = df
        .column(key)?
        .as_materialized_series()
        .to_physical_repr()
        .into_owned();
    Ok(match keys.dtype() {
        DataType::String => keys.str().unwrap().as_binary().into_series(),
        _ => keys,
    })
}

/// Sorts `df` and restricts it to the `(offset, len)` row range of the result.
fn sort_range(
    df: DataFrame,
    key: &str,
    sort_options: SortMultipleOptions,
    (offset, len): (usize, usize),
) -> PolarsResult<DataFrame> {
    let by_column = vec![df.column(key)?.clone()];
    if offset == 0 {
        // A `(0, len)` slice would send `sort_impl` down the bottom-k path,
        // which row-encodes the key.
        return Ok(df.sort_impl(by_column, sort_options, None)?.slice(0, len));
    }
    df.sort_impl(by_column, sort_options, Some((offset as i64, len)))
}

/// Drops the items on the thread pool, as there may be many large frames.
fn drop_parallel<T: Send>(items: impl IntoIterator<Item = T>) {
    let items: Vec<T> = items.into_iter().collect();
    RAYON.install(|| items.into_par_iter().for_each(drop));
}

/// Partitions one task's share of the buffered frames into per-bucket frames.
async fn partition_frames(
    frames: Vec<SpillFrame>,
    classifier: &BucketClassifier,
    input_schema: &Arc<Schema>,
    key: &str,
    flush_rows: usize,
    maintain_order: bool,
    spill_ctx: &MostRecentSpillContext,
) -> PolarsResult<Vec<Vec<SpillFrame>>> {
    let num_outputs = classifier.num_buckets() + 1;
    let mut builders: Vec<DataFrameBuilder> = (0..num_outputs)
        .map(|_| DataFrameBuilder::new(input_schema.clone()))
        .collect();
    let mut idxs_per_bucket: Vec<Vec<IdxSize>> = (0..num_outputs).map(|_| Vec::new()).collect();
    let mut out: Vec<Vec<SpillFrame>> = (0..num_outputs).map(|_| Vec::new()).collect();

    for frame in frames {
        if frame.height() == 0 {
            continue;
        }
        let mut df = frame.into_df().await;
        df.rechunk_mut();

        for idxs in idxs_per_bucket.iter_mut() {
            idxs.clear();
        }
        {
            let keys = key_series(&df, key)?;
            classifier.gen_idxs_per_bucket(&keys, &mut idxs_per_bucket);
        }

        // If all rows land in one bucket the frame is stored as is.
        if let Some(b) = idxs_per_bucket
            .iter()
            .position(|idxs| idxs.len() == df.height())
        {
            if maintain_order && !builders[b].is_empty() {
                out[b].push(SpillFrame::new(builders[b].freeze_reset(), spill_ctx).await);
            }
            out[b].push(SpillFrame::new(df, spill_ctx).await);
            continue;
        }

        for b in 0..num_outputs {
            if idxs_per_bucket[b].is_empty() {
                continue;
            }
            // Reserving up front avoids repeated reallocation while growing.
            if builders[b].is_empty() {
                builders[b].reserve(flush_rows + idxs_per_bucket[b].len());
            }
            // SAFETY: the indices are row offsets within the rechunked frame.
            unsafe {
                builders[b].gather_extend(&df, &idxs_per_bucket[b], ShareStrategy::Never);
            }
            if builders[b].len() >= flush_rows {
                out[b].push(SpillFrame::new(builders[b].freeze_reset(), spill_ctx).await);
            }
        }
    }

    for (b, builder) in builders.iter_mut().enumerate() {
        if !builder.is_empty() {
            out[b].push(SpillFrame::new(builder.freeze_reset(), spill_ctx).await);
        }
    }

    Ok(out)
}
