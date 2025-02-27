use futures::stream::FuturesUnordered;
use futures::StreamExt;
use polars_core::config;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;

use super::io_sources::PhaseOutcomeToken;
use super::{ComputeNode, JoinHandle, Morsel, PortState, RecvPort, SendPort, TaskScope};
use crate::async_executor::{spawn, AbortOnDropHandle};
use crate::async_primitives::connector::{connector, Receiver, Sender};
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::nodes::TaskPriority;

#[cfg(feature = "csv")]
pub mod csv;
#[cfg(feature = "ipc")]
pub mod ipc;
#[cfg(feature = "json")]
pub mod json;
#[cfg(feature = "parquet")]
pub mod parquet;
pub mod partition;

pub enum SinkInputPort {
    Serial(Receiver<Morsel>),
    Parallel(Vec<Receiver<Morsel>>),
}

pub struct SinkInput {
    pub outcome: PhaseOutcomeToken,
    pub port: SinkInputPort,

    #[allow(unused)]
    /// Dropping this indicates that the phase is done.
    wait_token: WaitToken,
}

pub struct SinkRecvPort {
    num_pipelines: usize,
    recv: Receiver<SinkInput>,
}

impl SinkInputPort {
    pub fn serial(self) -> Receiver<Morsel> {
        match self {
            Self::Serial(s) => s,
            _ => panic!(),
        }
    }

    pub fn parallel(self) -> Vec<Receiver<Morsel>> {
        match self {
            Self::Parallel(s) => s,
            _ => panic!(),
        }
    }
}

impl SinkRecvPort {
    #[allow(clippy::type_complexity)]
    pub fn parallel(
        mut self,
    ) -> (
        JoinHandle<PolarsResult<()>>,
        Vec<Receiver<(WaitToken, PhaseOutcomeToken, Receiver<Morsel>)>>,
    ) {
        let (mut rx_senders, rx_receivers) = (0..self.num_pipelines)
            .map(|_| connector())
            .collect::<(Vec<_>, Vec<_>)>();
        let handle = spawn(TaskPriority::High, async move {
            let wg = WaitGroup::default();

            while let Ok(input) = self.recv.recv().await {
                let inputs = input.port.parallel();

                let mut outcomes = Vec::with_capacity(inputs.len());
                for (input, rx_sender) in inputs.into_iter().zip(rx_senders.iter_mut()) {
                    let outcome = PhaseOutcomeToken::new();
                    if rx_sender
                        .send((wg.token(), outcome.clone(), input))
                        .await
                        .is_err()
                    {
                        return Ok(());
                    }
                    outcomes.push(outcome);
                }

                wg.wait().await;
                for outcome in outcomes {
                    if outcome.did_finish() {
                        return Ok(());
                    }
                }
                input.outcome.stop();
            }

            Ok(())
        });

        (handle, rx_receivers)
    }
    fn serial(self) -> Receiver<SinkInput> {
        self.recv
    }
}

impl SinkInput {
    pub fn from_port(port: SinkInputPort) -> (PhaseOutcomeToken, WaitGroup, Self) {
        let outcome = PhaseOutcomeToken::new();
        let wait_group = WaitGroup::default();

        let input = Self {
            outcome: outcome.clone(),
            wait_token: wait_group.token(),
            port,
        };
        (outcome, wait_group, input)
    }
}

pub trait SinkNode {
    fn name(&self) -> &str;
    fn is_sink_input_parallel(&self) -> bool;
    fn spawn_sink(
        &mut self,
        num_pipelines: usize,
        recv_ports_recv: SinkRecvPort,
        state: &ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    );
}

/// The state needed to manage a spawned [`SinkNode`].
struct StartedSinkComputeNode {
    input_send: Sender<SinkInput>,
    join_handles: FuturesUnordered<AbortOnDropHandle<PolarsResult<()>>>,
}

/// A [`ComputeNode`] to wrap a [`SinkNode`].
pub struct SinkComputeNode {
    sink: Box<dyn SinkNode + Send + Sync>,
    num_pipelines: usize,
    started: Option<StartedSinkComputeNode>,
}

impl SinkComputeNode {
    pub fn new(sink: Box<dyn SinkNode + Send + Sync>) -> Self {
        Self {
            sink,
            num_pipelines: 0,
            started: None,
        }
    }
}

impl<T: SinkNode + Send + Sync + 'static> From<T> for SinkComputeNode {
    fn from(value: T) -> Self {
        Self::new(Box::new(value))
    }
}

impl ComputeNode for SinkComputeNode {
    fn name(&self) -> &str {
        self.sink.name()
    }
    fn initialize(&mut self, num_pipelines: usize) {
        self.num_pipelines = num_pipelines;
    }
    fn update_state(
        &mut self,
        recv: &mut [PortState],
        _send: &mut [PortState],
    ) -> PolarsResult<()> {
        if recv[0] != PortState::Done {
            recv[0] = PortState::Ready;
        }

        if recv[0] == PortState::Done {
            if let Some(mut started) = self.started.take() {
                drop(started.input_send);
                polars_io::pl_async::get_runtime().block_on(async move {
                    // Either the task finished or some error occurred.
                    while let Some(ret) = started.join_handles.next().await {
                        ret?;
                    }
                    PolarsResult::Ok(())
                })?;
            }
        }

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert_eq!(recv_ports.len(), 1);
        assert!(send_ports.is_empty());

        let name = self.name().to_string();
        let started = self.started.get_or_insert_with(|| {
            let (tx, rx) = connector();
            let mut join_handles = Vec::new();

            self.sink.spawn_sink(
                self.num_pipelines,
                SinkRecvPort {
                    num_pipelines: self.num_pipelines,
                    recv: rx,
                },
                state,
                &mut join_handles,
            );
            // One of the tasks might throw an error. In which case, we need to cancel all
            // handles and find the error.
            let join_handles: FuturesUnordered<_> =
                join_handles.drain(..).map(AbortOnDropHandle::new).collect();

            StartedSinkComputeNode {
                input_send: tx,
                join_handles,
            }
        });

        let recv = recv_ports[0].take().unwrap();
        let sink_input = if self.sink.is_sink_input_parallel() {
            SinkInputPort::Parallel(recv.parallel())
        } else {
            SinkInputPort::Serial(recv.serial())
        };
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let (outcome, wait_group, sink_input) = SinkInput::from_port(sink_input);
            if started.input_send.send(sink_input).await.is_ok() {
                // Wait for the phase to finish.
                wait_group.wait().await;
                if !outcome.did_finish() {
                    return Ok(());
                }

                if config::verbose() {
                    eprintln!("[{name}]: Last data sent.");
                }
            }

            // Either the task finished or some error occurred.
            while let Some(ret) = started.join_handles.next().await {
                ret?;
            }

            Ok(())
        }));
    }
}
