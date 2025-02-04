use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_utils::pl_str::PlSmallStr;

use super::{ComputeNode, JoinHandle, Morsel, PortState, RecvPort, SendPort, TaskPriority};
use crate::async_primitives::connector::{connector, Receiver, Sender};

#[cfg(feature = "csv")]
pub mod csv;
#[cfg(feature = "ipc")]
pub mod ipc;
pub mod multi_scan;
#[cfg(feature = "parquet")]
pub mod parquet;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhaseResult {
    Stopped,
    Finished,
}

/// The state needed to manage a spawned [`SourceNode`].
struct StartedSourceComputeNode {
    send_port_send: Sender<SourceOutput>,
    join_handles: Vec<JoinHandle<PolarsResult<()>>>,
    phase_result_rx: Receiver<PhaseResult>,
}

/// A [`ComputeNode`] to wrap a [`SourceNode`].
pub struct SourceComputeNode<T: SourceNode + Send + Sync> {
    source: T,
    num_pipelines: usize,
    started: Option<StartedSourceComputeNode>,
}

impl<T: SourceNode + Send + Sync> SourceComputeNode<T> {
    pub fn new(source: T) -> Self {
        Self {
            source,
            num_pipelines: 0,
            started: None,
        }
    }
}

impl<T: SourceNode> ComputeNode for SourceComputeNode<T> {
    fn name(&self) -> &str {
        self.source.name()
    }

    fn initialize(&mut self, num_pipelines: usize) {
        self.num_pipelines = num_pipelines;
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
    ) -> polars_error::PolarsResult<()> {
        assert!(recv.is_empty());
        assert_eq!(send.len(), 1);

        if self
            .started
            .as_ref()
            .is_some_and(|s| s.join_handles.is_empty())
        {
            send[0] = PortState::Done;
        }

        if send[0] != PortState::Done {
            send[0] = PortState::Ready;
        }

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s super::TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.is_empty());
        assert_eq!(send_ports.len(), 1);

        let started = self.started.get_or_insert_with(|| {
            let (tx, rx) = connector();
            let mut join_handles = Vec::new();
            let phase_result_rx =
                self.source
                    .spawn_source(self.num_pipelines, rx, state, &mut join_handles, None);

            StartedSourceComputeNode {
                send_port_send: tx,
                join_handles,
                phase_result_rx,
            }
        });

        let send = send_ports[0].take().unwrap();
        let source_output = if self.source.is_source_output_parallel() {
            SourceOutput::Parallel(send.parallel())
        } else {
            SourceOutput::Serial(send.serial())
        };
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            if started.send_port_send.send(source_output).await.is_err() {
                return Ok(());
            };

            // Wait for an indication of whether we were stopped or whether we are finished.
            let Ok(phase_result) = started.phase_result_rx.recv().await else {
                return Ok(());
            };
            if matches!(phase_result, PhaseResult::Finished) {
                for handle in std::mem::take(&mut started.join_handles) {
                    handle.await?;
                }
            }

            Ok(())
        }));
    }
}

/// The output of a [`SourceNode`].
///
/// This is essentially an owned [`SendPort`].
pub enum SourceOutput {
    Serial(Sender<Morsel>),
    Parallel(Vec<Sender<Morsel>>),
}

impl SourceOutput {
    pub fn serial(self) -> Sender<Morsel> {
        match self {
            Self::Serial(s) => s,
            _ => panic!(),
        }
    }

    pub fn parallel(self) -> Vec<Sender<Morsel>> {
        match self {
            Self::Parallel(s) => s,
            _ => panic!(),
        }
    }
}

/// A node in the streaming physical graph that only produces [`Morsel`]s.
///
/// These can be converting into [`ComputeNode`]s that will have non-scoped tasks.
pub trait SourceNode: Sized + Send + Sync {
    const EFFICIENT_PRED_PD: bool;
    const EFFICIENT_SLICE_PD: bool;

    fn name(&self) -> &str;

    fn is_source_output_parallel(&self) -> bool {
        false
    }

    /// Start all the tasks for the [`SourceNode`].
    ///
    /// This should repeatedly take a [`SourceOutput`] from `output_recv` and output its
    /// [`Morsel`] into that channel. When a stop is requested, the output should be dropped
    /// and the task should wait for a new output to be provided. When the source is finished,
    /// the output channel should also be dropped.
    ///
    /// This returns a channel that after each drop of the output gives the result of the phase. It
    /// return [`PhaseResult::Stopped`] if the source is not yet finished (and it was thus stopped
    /// due to a requested stop), or [`PhaseResult::Finished`] if the source is finished.
    ///
    /// It should produce at least one task that lives until the source is finished and all the
    /// join handles for the source tasks should be directly or indirectly awaited by await all
    /// `join_handles`.
    ///
    /// If the `unfiltered_row_count` is given as `Some(..)` a scalar column is appended at the end
    /// of the dataframe that contains the unrestricted row count for each `Morsel` (i.e. the row
    /// count before slicing and predicate filtering).
    fn spawn_source(
        &mut self,
        num_pipelines: usize,
        output_recv: Receiver<SourceOutput>,
        state: &ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
        unrestricted_row_count: Option<PlSmallStr>,
    ) -> Receiver<PhaseResult>;
}
