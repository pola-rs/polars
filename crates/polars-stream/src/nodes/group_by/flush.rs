use std::cmp::Reverse;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crossbeam_queue::ArrayQueue;
use polars_async::executor::{JoinHandle, TaskPriority, TaskScope};
use polars_async::primitives::wait_group::WaitGroup;
use polars_core::frame::DataFrame;
use polars_error::PolarsResult;
use polars_ooc::memory_manager;

use super::pre_agg::SplitPreAggs;
use super::table::{GroupByParams, PartitionTable};
use crate::execute::StreamingExecutionState;
use crate::hash_shuffle::store::StoredPartition;
use crate::metrics::{MetricReporter, kind};
use crate::morsel::{Morsel, MorselSeq, SourceToken, get_ideal_morsel_size};
use crate::pipe::PortSender;

/// The work of one output task, kept across execution phases.
enum TaskState {
    Idle,
    /// A stored partition being built from its frames. Its groups of the pre-aggregates are
    /// already added.
    Building {
        table: PartitionTable,
        stored: StoredPartition,
    },
    /// A result being sent, from row `offset` on.
    Sending {
        frame: DataFrame,
        offset: usize,
    },
}

/// What the output tasks share.
struct Shared {
    /// Results ready to be sent.
    outputs: ArrayQueue<DataFrame>,
    /// Partitions to build, largest first.
    partitions: ArrayQueue<StoredPartition>,
    /// The pre-aggregates the stored partitions are built from.
    pre_aggs: SplitPreAggs,
    params: Arc<GroupByParams>,
    /// The number of groups to reserve in each partition table.
    reserve_groups: usize,
    next_seq: AtomicU64,
    /// Set when a task failed, so that the others stop.
    failed: AtomicBool,
    actual_groups: MetricReporter<kind::Sum>,
}

/// Sends the results as morsels across execution phases. Each output task builds or sends at
/// most one partition at a time, keeps it in its state when a phase ends, and resumes it first.
pub struct Flush {
    shared: Shared,
    /// Task i alone touches `tasks[i]`.
    tasks: Vec<TaskState>,
}

impl Flush {
    pub fn new(
        outputs: Vec<DataFrame>,
        mut partitions: Vec<StoredPartition>,
        pre_aggs: SplitPreAggs,
        num_tasks: usize,
        params: Arc<GroupByParams>,
        reserve_groups: usize,
        actual_groups: MetricReporter<kind::Sum>,
    ) -> Self {
        partitions.sort_by_key(|p| Reverse(p.estimated_size()));
        let shared = Shared {
            outputs: ArrayQueue::new(outputs.len().max(1)),
            partitions: ArrayQueue::new(partitions.len().max(1)),
            pre_aggs,
            params,
            reserve_groups,
            next_seq: AtomicU64::new(0),
            failed: AtomicBool::new(false),
            actual_groups,
        };
        for frame in outputs {
            shared.outputs.push(frame).ok().unwrap();
        }
        for partition in partitions {
            shared.partitions.push(partition).ok().unwrap();
        }
        Self {
            shared,
            tasks: (0..num_tasks).map(|_| TaskState::Idle).collect(),
        }
    }

    pub fn is_finished(&self) -> bool {
        self.shared.outputs.is_empty()
            && self.shared.partitions.is_empty()
            && self
                .tasks
                .iter()
                .all(|task| matches!(task, TaskState::Idle))
    }

    /// One output task per sender.
    pub fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        senders: Vec<PortSender>,
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert_eq!(senders.len(), self.tasks.len());
        let num_resumed = self
            .tasks
            .iter()
            .filter(|task| !matches!(task, TaskState::Idle))
            .count();
        if num_resumed > 0 && polars_config::config().verbose() {
            eprintln!("[group-by]: resuming {num_resumed} partly built or sent partitions");
        }

        let Self { shared, tasks } = self;
        let shared = &*shared;
        let source_token = SourceToken::new();
        for (send, task) in senders.into_iter().zip(tasks.iter_mut()) {
            let source_token = source_token.clone();
            join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                let result = shared.run(task, send, &source_token, state).await;
                if result.is_err() {
                    shared.failed.store(true, Ordering::Relaxed);
                }
                result
            }));
        }
    }
}

impl Shared {
    /// Works on `task` until no work is left, a stop is requested or another task failed.
    async fn run(
        &self,
        task: &mut TaskState,
        mut send: PortSender,
        source_token: &SourceToken,
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        let wait_group = WaitGroup::default();
        let morsel_size = get_ideal_morsel_size().max(1);
        loop {
            if source_token.stop_requested() || self.failed.load(Ordering::Relaxed) {
                return Ok(());
            }

            match task {
                TaskState::Idle => {
                    if let Some(frame) = self.outputs.pop() {
                        *task = TaskState::Sending { frame, offset: 0 };
                    } else if let Some(stored) = self.partitions.pop() {
                        let mut table =
                            PartitionTable::new(self.params.clone(), self.reserve_groups);
                        for (pre_agg, groups) in self.pre_aggs.in_partition(stored.index()) {
                            table.add_pre_agg(pre_agg, groups)?;
                        }
                        *task = TaskState::Building { table, stored };
                    } else {
                        return Ok(());
                    }
                },
                TaskState::Building { table, stored } => {
                    if let Some((input, frame)) = stored.pop_frame() {
                        let frame = frame.load().await;
                        let frame = self.params.prepare_frame(input, frame, state).await?;
                        table.add_rows(input, &frame, None)?;
                    } else {
                        let frame = table.finish()?;
                        self.actual_groups.add(frame.height() as i64);
                        *task = if frame.height() > 0 {
                            TaskState::Sending { frame, offset: 0 }
                        } else {
                            TaskState::Idle
                        };
                    }
                    memory_manager().spill().await;
                },
                TaskState::Sending { frame, offset } => {
                    // An empty result is sent as one empty morsel.
                    let df = frame.slice(*offset as i64, morsel_size);
                    *offset += df.height();
                    if *offset >= frame.height() {
                        *task = TaskState::Idle;
                    }
                    let seq = MorselSeq::new(self.next_seq.fetch_add(1, Ordering::Relaxed));
                    let mut morsel = Morsel::new_unregistered(df, seq, source_token.clone());
                    morsel.set_consume_token(wait_group.token());
                    if send.send(morsel).await.is_err() {
                        return Ok(());
                    }
                    wait_group.wait().await;
                },
            }
        }
    }
}
