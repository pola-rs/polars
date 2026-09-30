use std::collections::{BTreeMap, VecDeque};

use futures::future::join_all;
use parking_lot::Mutex;
use polars_async::executor::{JoinHandle, TaskPriority, TaskScope};
use polars_async::primitives::wait_group::WaitGroup;
use polars_core::chunked_array::ops::sort::options::SortMultipleOptions;
use polars_core::frame::DataFrame;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_error::PolarsResult;
use polars_ooc::{MostRecentSpillContext, SpillFrame};
use polars_utils::pl_str::PlSmallStr;
use tokio::sync::Notify;

use super::{drop_parallel, sort_range};
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::pipe::PortSender;

/// A bucket of the partitioned input, in emission order.
pub struct PendingBucket {
    pub frames: Vec<SpillFrame>,
    /// The rows of this bucket to emit, as an offset into the concatenation
    /// of `frames` (sorted when `needs_sort`) and a length.
    pub local: (usize, usize),
    pub needs_sort: bool,
}

struct FlushInner {
    /// Buckets not yet sorted, in output order, with the sequence number of
    /// their first morsel.
    unsorted: VecDeque<(PendingBucket, u64)>,
    /// Sorted rows waiting to be emitted, by the sequence number of their
    /// first morsel.
    sorted: BTreeMap<u64, DataFrame>,
    /// The sequence number of the next morsel to emit.
    next_seq: u64,
    num_sorting: usize,
    /// Set when a sort failed, so that the other tasks stop instead of waiting
    /// for it.
    failed: bool,
}

impl FlushInner {
    fn is_drained(&self) -> bool {
        self.unsorted.is_empty() && self.sorted.is_empty() && self.num_sorting == 0
    }
}

/// Emits the partitioned buckets in order. The sender tasks share one
/// emission cursor; a task with nothing to emit sorts a bucket ahead, up to
/// `max_ahead` buckets sorted or being sorted.
pub struct Flush {
    inner: Mutex<FlushInner>,
    notify: Notify,
    morsel_size: usize,
    max_ahead: usize,
    /// Keeps the bucket frames spillable while they wait to be sorted.
    _spill_ctx: MostRecentSpillContext,
}

enum Step<'a> {
    Emit(DataFrame, u64),
    Sort(PendingBucket, u64),
    Wait(tokio::sync::futures::Notified<'a>),
    Stop,
}

impl Flush {
    pub fn new(
        buckets: Vec<PendingBucket>,
        morsel_size: usize,
        max_ahead: usize,
        spill_ctx: MostRecentSpillContext,
    ) -> Self {
        let mut seq = 0;
        let mut unsorted = VecDeque::with_capacity(buckets.len());
        for bucket in buckets {
            let num_morsels = bucket.local.1.div_ceil(morsel_size) as u64;
            unsorted.push_back((bucket, seq));
            seq += num_morsels;
        }
        Self {
            inner: Mutex::new(FlushInner {
                unsorted,
                sorted: BTreeMap::new(),
                next_seq: 0,
                num_sorting: 0,
                failed: false,
            }),
            notify: Notify::new(),
            morsel_size,
            max_ahead,
            _spill_ctx: spill_ctx,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.inner.lock().is_drained()
    }

    fn next_step(&self, source_token: &SourceToken) -> Step<'_> {
        let mut inner = self.inner.lock();
        if source_token.stop_requested() || inner.failed {
            return Step::Stop;
        }
        let seq = inner.next_seq;
        if let Some(entry) = inner.sorted.first_entry()
            && *entry.key() == seq
        {
            let df = entry.remove();
            inner.next_seq += 1;
            if df.height() > self.morsel_size {
                let rest = df.slice(self.morsel_size as i64, usize::MAX);
                inner.sorted.insert(seq + 1, rest);
            }
            return Step::Emit(df.slice(0, self.morsel_size), seq);
        }

        if inner.sorted.len() + inner.num_sorting < self.max_ahead {
            if let Some((bucket, seq)) = inner.unsorted.pop_front() {
                inner.num_sorting += 1;
                return Step::Sort(bucket, seq);
            }
        }
        if inner.is_drained() {
            return Step::Stop;
        }
        // The next bucket to emit is being sorted, which notifies when done.
        Step::Wait(self.notify.notified())
    }

    fn finish_sort(&self, seq: u64, sorted: PolarsResult<DataFrame>) -> PolarsResult<()> {
        let result = {
            let mut inner = self.inner.lock();
            inner.num_sorting -= 1;
            match sorted {
                Ok(df) => {
                    inner.sorted.insert(seq, df);
                    Ok(())
                },
                Err(e) => {
                    inner.failed = true;
                    Err(e)
                },
            }
        };
        self.notify.notify_waiters();
        result
    }

    pub fn spawn<'env, 's>(
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        senders: Vec<PortSender>,
        key: &'env PlSmallStr,
        sort_options: &SortMultipleOptions,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let source_token = SourceToken::new();
        let mut sort_options = sort_options.clone();
        sort_options.multithreaded = false;

        for mut send in senders {
            let sort_options = sort_options.clone();
            let source_token = source_token.clone();
            join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                let wait_group = WaitGroup::default();
                loop {
                    match self.next_step(&source_token) {
                        Step::Emit(df, seq) => {
                            let mut morsel = Morsel::new_unregistered(
                                df,
                                MorselSeq::new(seq),
                                source_token.clone(),
                            );
                            morsel.set_consume_token(wait_group.token());
                            if send.send(morsel).await.is_err() {
                                return Ok(());
                            }
                            wait_group.wait().await;
                        },
                        Step::Sort(bucket, seq) => {
                            let sorted = sort_bucket(bucket, key, &sort_options).await;
                            self.finish_sort(seq, sorted)?;
                        },
                        Step::Wait(notified) => notified.await,
                        Step::Stop => return Ok(()),
                    }
                }
            }));
        }
    }
}

impl Drop for Flush {
    fn drop(&mut self) {
        let inner = self.inner.get_mut();
        drop_parallel(
            inner
                .unsorted
                .drain(..)
                .flat_map(|(bucket, _)| bucket.frames),
        );
    }
}

/// Loads a bucket and sorts it.
async fn sort_bucket(
    bucket: PendingBucket,
    key: &PlSmallStr,
    sort_options: &SortMultipleOptions,
) -> PolarsResult<DataFrame> {
    let dfs = join_all(bucket.frames.into_iter().map(SpillFrame::into_df)).await;
    let df = accumulate_dataframes_vertical_unchecked(dfs);
    let (offset, len) = bucket.local;
    if bucket.needs_sort {
        sort_range(df, key, sort_options.clone(), bucket.local)
    } else {
        Ok(df.slice(offset as i64, len))
    }
}
