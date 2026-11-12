use std::cmp::Reverse;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use polars_async::parallel::for_each_parallel;
use polars_async::primitives::wait_group::WaitGroup;
use polars_core::config;
use polars_expr::idx_table::IdxTable;
use polars_ooc::{MemoryGrant, MemoryReservation, memory_manager};
use polars_utils::IdxSize;

use super::build::PartitionBuilder;
use super::probe::{
    ProbeOutput, ProbeState, ProbeTable, output_columns, with_nulls_for_other_side,
    with_nulls_for_other_side_projected,
};
use super::{EquiJoinParams, JoinPartition};
use crate::hash_shuffle::frame_with_keys::FrameWithKeys;
use crate::hash_shuffle::split::rows_or_all;
use crate::hash_shuffle::store::StoredPartition;
use crate::morsel::{SourceToken, get_ideal_morsel_size};
use crate::nodes::compute_node_prelude::*;
use crate::nodes::joins::tuning::BuildStats;

/// The work of one flush task, kept across execution phases.
enum TaskState {
    Idle,
    /// The table of a stored partition being built from its build frames. Its `bytes` are
    /// admitted.
    Building {
        stored: StoredPartition,
        builder: PartitionBuilder,
        bytes: u64,
        _reservation: MemoryReservation,
    },
    /// The table of a stored partition being probed with its probe frames, from row `offset` of
    /// `frame` on.
    Probing {
        stored: StoredPartition,
        table: ProbeTable,
        frame: Option<Box<FrameWithKeys>>,
        offset: usize,
        bytes: u64,
    },
    /// The build rows of a table without a match being sent, from row `offset` on.
    EmitUnmatched {
        table: ProbeTable,
        offset: usize,
        bytes: u64,
    },
    /// The frames of one side of a partition being sent with nulls for the other side, from row
    /// `offset` of `frame` on.
    EmitFrames {
        stored: StoredPartition,
        is_left: bool,
        frame: Option<DataFrame>,
        offset: usize,
    },
}

/// One flush task: its work and the output it gathers.
struct Task {
    state: TaskState,
    out: ProbeOutput,
    all_rows: Vec<IdxSize>,
    unmarked: Vec<IdxSize>,
}

/// Work taken from the queue.
enum Work {
    Ready(TaskState),
    /// A stored partition, admitted.
    Stored(JoinPartition),
}

/// The work left and the memory of the admitted tables, shared by the flush tasks.
struct Queue {
    /// Work that needs no admission: tables already in memory first, then frames to send.
    ready: VecDeque<TaskState>,
    /// Stored partitions, largest table first.
    stored: VecDeque<JoinPartition>,
    /// The bytes of the tables in memory or being built.
    admitted: u64,
}

impl Queue {
    /// Takes ready work, or else the largest stored partition if its table fits in `grant_bytes`
    /// next to the admitted tables, or if no table is admitted.
    fn take(&mut self, grant_bytes: u64) -> Option<Work> {
        if let Some(state) = self.ready.pop_front() {
            return Some(Work::Ready(state));
        }
        let bytes = self.stored.front()?.table_bytes;
        if self.admitted > 0 && self.admitted + bytes > grant_bytes {
            return None;
        }
        self.admitted += bytes;
        Some(Work::Stored(self.stored.pop_front().unwrap()))
    }
}

/// What the flush tasks share.
struct Shared {
    queue: Mutex<Queue>,
    /// Shrunk to the remaining bytes.
    grant: MemoryGrant,
    /// The bytes of the tables not yet freed, including those not yet built.
    remaining: AtomicU64,
    /// An empty table of the kind the partitions use.
    table: Box<dyn IdxTable>,
    stats: BuildStats,
    next_seq: AtomicU64,
    /// Set when a task failed, so that the others stop.
    failed: AtomicBool,
}

/// Emits what is left once the probe input ends, across execution phases: the build rows without
/// a match of the tables in memory and of the null keys, and the join of each stored partition,
/// whose tables are built as the memory grant allows. Each task works on one item at a time,
/// keeps it in its state when a phase ends, and resumes it first. A task that finds no work it can
/// start ends its phase: waiting would keep its output port open, which holds back a downstream
/// node that receives the morsels in order.
pub struct FlushState {
    shared: Shared,
    /// Task i alone touches `tasks[i]`.
    tasks: Vec<Task>,
}

impl FlushState {
    pub fn new(
        probe: &mut ProbeState,
        params: &EquiJoinParams,
        table: &dyn IdxTable,
        state: &StreamingExecutionState,
    ) -> PolarsResult<Self> {
        let left_is_build = params.left_is_build.unwrap();
        let build_outer = params.emit_unmatched_build();
        let probe_outer = params.emit_unmatched_probe();

        let mut partitions = core::mem::take(&mut probe.partitions.partitions);
        for writer in core::mem::take(&mut probe.writers) {
            for (p, frames) in partitions.iter_mut().zip(writer.finish()) {
                p.stored.append(1, frames);
            }
        }

        let mut tables = VecDeque::new();
        let mut frames = VecDeque::new();
        let mut stored = Vec::new();
        let mut to_free = Vec::new();
        let mut table_bytes = 0;
        let (mut probe_rows, mut probe_bytes) = (0, 0);
        for mut partition in partitions {
            if let Some(in_memory) = partition.table.take() {
                if build_outer {
                    table_bytes += partition.table_bytes;
                    tables.push_back(TaskState::EmitUnmatched {
                        table: in_memory,
                        offset: 0,
                        bytes: partition.table_bytes,
                    });
                } else {
                    to_free.push(in_memory);
                }
                continue;
            }

            probe_rows += partition.stored.num_rows(1);
            probe_bytes += partition.stored.input_size(1);
            let has_build_rows = partition.stored.num_rows(0) > 0;
            let has_probe_rows = partition.stored.num_rows(1) > 0;
            if has_build_rows && has_probe_rows {
                stored.push(partition);
            } else if (has_build_rows && build_outer) || (has_probe_rows && probe_outer) {
                // The rows of the side that has rows have no match.
                frames.push_back(TaskState::EmitFrames {
                    stored: partition.stored,
                    is_left: has_build_rows == left_is_build,
                    frame: None,
                    offset: 0,
                });
            }
        }
        if let Some(partition) = probe.null_build_rows.take()
            && partition.num_rows(0) > 0
        {
            frames.push_back(TaskState::EmitFrames {
                stored: partition,
                is_left: left_is_build,
                frame: None,
                offset: 0,
            });
        }
        if !stored.is_empty() && config::verbose() {
            eprintln!(
                "[equi-join]: flushing {} stored partitions with {probe_rows} probe rows ({probe_bytes} bytes)",
                stored.len()
            );
        }

        for_each_parallel(to_free, state.task_metrics(), &|table| async move {
            drop(table);
            Ok(())
        })?;

        stored.sort_by_key(|p| Reverse(p.table_bytes));
        let remaining = table_bytes + stored.iter().map(|p| p.table_bytes).sum::<u64>();
        let grant = probe.grant.take().unwrap();
        grant.shrink_to(remaining);
        tables.append(&mut frames);
        let queue = Queue {
            ready: tables,
            stored: stored.into(),
            admitted: table_bytes,
        };

        let shared = Shared {
            queue: Mutex::new(queue),
            grant,
            remaining: AtomicU64::new(remaining),
            table: table.new_empty(),
            stats: core::mem::take(&mut probe.stats),
            next_seq: AtomicU64::new(probe.max_seq_sent.successor().to_u64()),
            failed: AtomicBool::new(false),
        };
        let tasks = (0..state.num_pipelines)
            .map(|_| Task {
                state: TaskState::Idle,
                out: ProbeOutput::new(params, left_is_build),
                all_rows: Vec::new(),
                unmarked: Vec::new(),
            })
            .collect();
        Ok(Self { shared, tasks })
    }

    pub fn is_finished(&self) -> bool {
        let queue = self.shared.queue.lock().unwrap();
        queue.ready.is_empty()
            && queue.stored.is_empty()
            && self
                .tasks
                .iter()
                .all(|task| matches!(task.state, TaskState::Idle))
    }

    /// One task per sender.
    pub fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        senders: Vec<PortSender>,
        params: &'env EquiJoinParams,
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert_eq!(senders.len(), self.tasks.len());
        let num_resumed = self
            .tasks
            .iter()
            .filter(|task| !matches!(task.state, TaskState::Idle))
            .count();
        if num_resumed > 0 && config::verbose() {
            eprintln!("[equi-join]: resuming {num_resumed} partly flushed partitions");
        }
        let Self { shared, tasks } = self;
        let shared = &*shared;
        let source_token = SourceToken::new();
        for (send, task) in senders.into_iter().zip(tasks.iter_mut()) {
            let source_token = source_token.clone();
            join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                let result = shared.run(task, send, &source_token, params, state).await;
                if result.is_err() {
                    shared.failed.store(true, Ordering::Relaxed);
                }
                result
            }));
        }
    }
}

impl Shared {
    /// Releases the bytes of a freed table.
    fn release(&self, bytes: u64) {
        self.queue.lock().unwrap().admitted -= bytes;
        let remaining = self.remaining.fetch_sub(bytes, Ordering::Relaxed) - bytes;
        self.grant.shrink_to(remaining);
    }

    /// Takes the next work that can start into `task`. Returns `false` if there is none.
    async fn take_work(&self, task: &mut Task, params: &EquiJoinParams) -> bool {
        let work = self.queue.lock().unwrap().take(self.grant.bytes());
        match work {
            Some(Work::Ready(state)) => task.state = state,
            Some(Work::Stored(partition)) => {
                let reservation = memory_manager().reserve(partition.net_table_bytes()).await;
                task.state = TaskState::Building {
                    builder: partition.table_builder(&*self.table, params, &self.stats),
                    stored: partition.stored,
                    bytes: partition.table_bytes,
                    _reservation: reservation,
                };
            },
            None => return false,
        }
        true
    }

    /// Works on `task` until no work it can start is left, a stop is requested or another task
    /// failed.
    async fn run(
        &self,
        task: &mut Task,
        mut send: PortSender,
        source_token: &SourceToken,
        params: &EquiJoinParams,
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        let wait_group = WaitGroup::default();
        let morsel_size = get_ideal_morsel_size().max(1);
        let left_is_build = params.left_is_build.unwrap();
        let track_unmatchable = params.emit_unmatched_build();
        loop {
            if source_token.stop_requested() || self.failed.load(Ordering::Relaxed) {
                return Ok(());
            }

            let mut output = None;
            match &mut task.state {
                TaskState::Idle => {
                    if !self.take_work(task, params).await {
                        return Ok(());
                    }
                    continue;
                },
                TaskState::Building {
                    stored, builder, ..
                } => match stored.pop_frame() {
                    Some((0, frame)) => {
                        let frame = frame.load().await;
                        builder.add_rows(&frame, None, track_unmatchable, None);
                    },
                    next => {
                        let frame = match next {
                            Some((_, frame)) => Some(Box::new(frame.load().await)),
                            None => None,
                        };
                        let TaskState::Building {
                            stored,
                            builder,
                            bytes,
                            ..
                        } = core::mem::replace(&mut task.state, TaskState::Idle)
                        else {
                            unreachable!()
                        };
                        task.state = TaskState::Probing {
                            stored,
                            table: builder.finish(),
                            frame,
                            offset: 0,
                            bytes,
                        };
                    },
                },
                TaskState::Probing {
                    stored,
                    table,
                    frame,
                    offset,
                    bytes,
                } => {
                    if task.out.is_full() || (frame.is_none() && !task.out.is_empty()) {
                        if let Some(frame) = frame {
                            task.out.gather_probe(&frame.frame);
                        }
                        output = Some(task.out.take(params));
                    } else if let Some(f) = frame {
                        let rows = rows_or_all(None, &mut task.all_rows, f.keys.len());
                        if *offset < rows.len() {
                            let exec_state = &state.in_memory_exec_state;
                            task.out
                                .probe(table, f, rows, offset, params, exec_state)
                                .await?;
                        } else {
                            task.out.gather_probe(&f.frame);
                            *frame = None;
                            *offset = 0;
                            if let Some((_, next)) = stored.pop_frame() {
                                *frame = Some(Box::new(next.load().await));
                            }
                        }
                    } else if track_unmatchable {
                        let TaskState::Probing { table, bytes, .. } =
                            core::mem::replace(&mut task.state, TaskState::Idle)
                        else {
                            unreachable!()
                        };
                        task.state = TaskState::EmitUnmatched {
                            table,
                            offset: 0,
                            bytes,
                        };
                    } else {
                        let bytes = *bytes;
                        task.state = TaskState::Idle;
                        self.release(bytes);
                    }
                },
                TaskState::EmitUnmatched {
                    table,
                    offset,
                    bytes,
                } => {
                    *offset += table.hash_table.unmarked_keys(
                        &mut task.unmarked,
                        *offset as IdxSize,
                        morsel_size as IdxSize,
                    ) as usize;
                    if task.unmarked.is_empty() {
                        let bytes = *bytes;
                        task.state = TaskState::Idle;
                        self.release(bytes);
                    } else {
                        // Gather only the output columns, not those only a fused predicate reads.
                        let payload =
                            output_columns(&table.payload, params.output_idx(left_is_build));
                        let df =
                            unsafe { payload.take_slice_unchecked_impl(&task.unmarked, false) };
                        output = Some(with_nulls_for_other_side_projected(
                            df,
                            left_is_build,
                            params,
                        ));
                    }
                },
                TaskState::EmitFrames {
                    stored,
                    is_left,
                    frame,
                    offset,
                } => {
                    if let Some(df) = frame
                        && *offset < df.height()
                    {
                        let df = df.slice(*offset as i64, morsel_size);
                        *offset += df.height();
                        output = Some(with_nulls_for_other_side(df, *is_left, params));
                    } else if let Some((_, next)) = stored.pop_frame() {
                        *frame = Some(next.load().await.frame);
                        *offset = 0;
                    } else {
                        task.state = TaskState::Idle;
                    }
                },
            }

            if let Some(df) = output {
                let seq = MorselSeq::new(self.next_seq.fetch_add(1, Ordering::Relaxed));
                let mut morsel = Morsel::new_unregistered(df, seq, source_token.clone());
                morsel.set_consume_token(wait_group.token());
                if send.send(morsel).await.is_err() {
                    return Ok(());
                }
                wait_group.wait().await;
            }
            memory_manager().spill().await;
        }
    }
}
