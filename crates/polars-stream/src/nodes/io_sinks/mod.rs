use futures::stream::FuturesUnordered;
use futures::StreamExt;
use polars_core::config;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;

use super::{
    ComputeNode, JoinHandle, Morsel, PhaseOutcome, PortState, RecvPort, SendPort, TaskScope,
};
use crate::async_executor::{spawn, AbortOnDropHandle};
use crate::async_primitives::connector::{connector, Receiver, Sender};
use crate::async_primitives::distributor_channel;
use crate::async_primitives::linearizer::{Inserter, Linearizer};
use crate::async_primitives::wait_group::WaitGroup;
use crate::nodes::TaskPriority;
use crate::DEFAULT_LINEARIZER_BUFFER_SIZE;

#[cfg(feature = "csv")]
pub mod csv;
#[cfg(feature = "ipc")]
pub mod ipc;
#[cfg(feature = "json")]
pub mod json;
#[cfg(feature = "parquet")]
pub mod parquet;

// This needs to be low to increase the backpressure.
const DEFAULT_SINK_LINEARIZER_BUFFER_SIZE: usize = 1;
const DEFAULT_SINK_DISTRIBUTOR_BUFFER_SIZE: usize = 1;

pub enum SinkInputPort {
    Serial(Receiver<Morsel>),
    Parallel(Vec<Receiver<Morsel>>),
}

pub struct SinkRecvPort {
    num_pipelines: usize,
    recv: Receiver<(PhaseOutcome, SinkInputPort)>,
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
    /// Receive the [`RecvPort`] in parallel and create a [`Linearizer`] for each phase.
    ///
    /// This is useful for sinks that process incoming [`Morsel`]s row-wise as the processing can
    /// be done in parallel and then be linearized into the actual sink.
    #[allow(clippy::type_complexity)]
    pub fn parallel_into_linearize<T: Send + Sync + Ord + 'static>(
        mut self,
    ) -> (
        JoinHandle<PolarsResult<()>>,
        Vec<Receiver<(PhaseOutcome, Receiver<Morsel>, Inserter<T>)>>,
        Receiver<(PhaseOutcome, Linearizer<T>)>,
    ) {
        let (mut rx_senders, rx_receivers) = (0..self.num_pipelines)
            .map(|_| connector())
            .collect::<(Vec<_>, Vec<_>)>();
        let (mut tx_linearizer, rx_linearizer) = connector();
        let handle = spawn(TaskPriority::High, async move {
            let mut outcomes = Vec::with_capacity(self.num_pipelines + 1);
            let wg = WaitGroup::default();

            while let Ok((phase_outcome, port)) = self.recv.recv().await {
                let inputs = port.parallel();

                let (linearizer, senders) =
                    Linearizer::<T>::new(self.num_pipelines, DEFAULT_SINK_LINEARIZER_BUFFER_SIZE);

                for ((input, rx_sender), sender) in
                    inputs.into_iter().zip(rx_senders.iter_mut()).zip(senders)
                {
                    let (token, outcome) = PhaseOutcome::new_shared_wait(wg.token());
                    if rx_sender.send((outcome, input, sender)).await.is_err() {
                        return Ok(());
                    }
                    outcomes.push(token);
                }
                let (token, outcome) = PhaseOutcome::new_shared_wait(wg.token());
                if tx_linearizer.send((outcome, linearizer)).await.is_err() {
                    return Ok(());
                }
                outcomes.push(token);

                wg.wait().await;
                for outcome in &outcomes {
                    if outcome.did_finish() {
                        return Ok(());
                    }
                }

                phase_outcome.stopped();
                outcomes.clear();
            }

            Ok(())
        });

        (handle, rx_receivers, rx_linearizer)
    }

    /// Serialize the input and allow for long lived lasts to listen to a constant channel.
    pub fn serial(
        mut self,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) -> Receiver<Morsel> {
        let (mut tx, rx) = connector();
        join_handles.push(spawn(TaskPriority::High, async move {
            while let Ok((outcome, port_rx)) = self.recv.recv().await {
                let mut port_rx = port_rx.serial();
                while let Ok(morsel) = port_rx.recv().await {
                    if tx.send(morsel).await.is_err() {
                        return Ok(());
                    }
                }
                outcome.stopped();
            }
            Ok(())
        }));
        rx
    }

    /// Receive the [`RecvPort`] serially that distributes amongst workers then [`Linearize`] again
    /// to the end.
    ///
    /// This is useful for sinks that process incoming [`Morsel`]s column-wise as the processing
    /// of the columns can be done in parallel.
    #[allow(clippy::type_complexity)]
    pub fn serial_into_distribute<D, L>(
        mut self,
    ) -> (
        JoinHandle<PolarsResult<()>>,
        Receiver<(
            PhaseOutcome,
            Option<Receiver<Morsel>>,
            distributor_channel::Sender<D>,
        )>,
        Vec<Receiver<(PhaseOutcome, distributor_channel::Receiver<D>, Inserter<L>)>>,
        Receiver<(PhaseOutcome, Linearizer<L>)>,
    )
    where
        D: Send + Sync + 'static,
        L: Send + Sync + Ord + 'static,
    {
        let (mut tx_linearizer, rx_linearizer) = connector();
        let (mut rx_senders, rx_receivers) = (0..self.num_pipelines)
            .map(|_| connector())
            .collect::<(Vec<_>, Vec<_>)>();
        let (mut tx_end, rx_end) = connector();
        let handle = spawn(TaskPriority::High, async move {
            let mut outcomes = Vec::with_capacity(self.num_pipelines + 2);
            let wg = WaitGroup::default();

            let mut stop = false;
            while !stop {
                let input = self.recv.recv().await;
                stop |= input.is_err(); // We want to send one last message without receiver when
                                        // the channel is dropped. This allows us to flush buffers.
                let (phase_outcome, receiver) = match input {
                    Ok((outcome, port)) => (Some(outcome), Some(port.serial())),
                    Err(()) => (None, None),
                };

                let (dist_tx, dist_rxs) = distributor_channel::distributor_channel::<D>(
                    self.num_pipelines,
                    DEFAULT_SINK_DISTRIBUTOR_BUFFER_SIZE,
                );
                let (linearizer, senders) =
                    Linearizer::<L>::new(self.num_pipelines, DEFAULT_LINEARIZER_BUFFER_SIZE);

                let (token, outcome) = PhaseOutcome::new_shared_wait(wg.token());
                if tx_linearizer
                    .send((outcome, receiver, dist_tx))
                    .await
                    .is_err()
                {
                    return Ok(());
                }
                outcomes.push(token);
                for ((dist_rx, rx_sender), sender) in
                    dist_rxs.into_iter().zip(rx_senders.iter_mut()).zip(senders)
                {
                    let (token, outcome) = PhaseOutcome::new_shared_wait(wg.token());
                    if rx_sender.send((outcome, dist_rx, sender)).await.is_err() {
                        return Ok(());
                    }
                    outcomes.push(token);
                }
                let (token, outcome) = PhaseOutcome::new_shared_wait(wg.token());
                if tx_end.send((outcome, linearizer)).await.is_err() {
                    return Ok(());
                }
                outcomes.push(token);

                wg.wait().await;
                for outcome in &outcomes {
                    if outcome.did_finish() {
                        return Ok(());
                    }
                }

                if let Some(outcome) = phase_outcome {
                    outcome.stopped()
                }
                outcomes.clear();
            }

            Ok(())
        });

        (handle, rx_linearizer, rx_receivers, rx_end)
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
    input_send: Sender<(PhaseOutcome, SinkInputPort)>,
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

        let wait_group = WaitGroup::default();
        let recv = recv_ports[0].take().unwrap();
        let sink_input = if self.sink.is_sink_input_parallel() {
            SinkInputPort::Parallel(recv.parallel())
        } else {
            SinkInputPort::Serial(recv.serial())
        };
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let (token, outcome) = PhaseOutcome::new_shared_wait(wait_group.token());
            if started.input_send.send((outcome, sink_input)).await.is_ok() {
                // Wait for the phase to finish.
                wait_group.wait().await;
                if !token.did_finish() {
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
