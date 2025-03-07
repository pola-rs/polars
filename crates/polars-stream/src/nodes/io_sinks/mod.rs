use std::{fs, io};

use futures::stream::FuturesUnordered;
use futures::StreamExt;
use polars_core::config;
use polars_core::frame::DataFrame;
use polars_core::prelude::Column;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_plan::dsl::SyncOnCloseType;

use super::io_sources::PhaseOutcomeToken;
use super::{
    ComputeNode, JoinHandle, Morsel, PhaseOutcome, PortState, RecvPort, SendPort, TaskScope,
};
use crate::async_executor::{spawn, AbortOnDropHandle};
use crate::async_primitives::connector::{connector, Receiver, Sender};
use crate::async_primitives::distributor_channel;
use crate::async_primitives::wait_group::WaitGroup;
use crate::nodes::TaskPriority;
use crate::prelude::TracedAwait;

#[cfg(feature = "csv")]
pub mod csv;
#[cfg(feature = "ipc")]
pub mod ipc;
#[cfg(feature = "json")]
pub mod json;
#[cfg(feature = "parquet")]
pub mod parquet;
pub mod partition;

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
    pub fn parallel(
        mut self,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) -> Vec<Receiver<Morsel>> {
        let (txs, rxs) = (0..self.num_pipelines)
            .map(|_| connector())
            .collect::<(Vec<_>, Vec<_>)>();
        let (mut pass_txs, pass_rxs) = (0..self.num_pipelines)
            .map(|_| connector())
            .collect::<(Vec<_>, Vec<_>)>();
        let mut outcomes = Vec::<PhaseOutcomeToken>::with_capacity(self.num_pipelines);
        let wg = WaitGroup::default();

        join_handles.push(spawn(TaskPriority::High, async move {
            while let Ok((outcome, port_rxs)) = self.recv.recv().traced_await().await {
                let port_rxs = port_rxs.parallel();
                for (pass_tx, port_rx) in pass_txs.iter_mut().zip(port_rxs) {
                    let (token, outcome) = PhaseOutcome::new_shared_wait(wg.token());
                    if pass_tx
                        .send((outcome, port_rx))
                        .traced_await()
                        .await
                        .is_err()
                    {
                        return Ok(());
                    }
                    outcomes.push(token);
                }

                wg.wait().traced_await().await;
                for outcome_token in &outcomes {
                    if outcome_token.did_finish() {
                        return Ok(());
                    }
                }
                outcomes.clear();
                outcome.stopped();
            }

            Ok(())
        }));
        join_handles.extend(pass_rxs.into_iter().zip(txs).map(|(mut pass_rx, mut tx)| {
            spawn(TaskPriority::High, async move {
                while let Ok((outcome, mut rx)) = pass_rx.recv().traced_await().await {
                    while let Ok(morsel) = rx.recv().traced_await().await {
                        if tx.send(morsel).traced_await().await.is_err() {
                            return Ok(());
                        }
                    }
                    outcome.stopped();
                }
                Ok(())
            })
        }));

        rxs
    }

    /// Serialize the input and allow for long lived lasts to listen to a constant channel.
    pub fn serial(
        mut self,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) -> Receiver<Morsel> {
        let (mut tx, rx) = connector();
        join_handles.push(spawn(TaskPriority::High, async move {
            while let Ok((outcome, port_rx)) = self.recv.recv().traced_await().await {
                let mut port_rx = port_rx.serial();
                while let Ok(morsel) = port_rx.recv().traced_await().await {
                    if tx.send(morsel).traced_await().await.is_err() {
                        return Ok(());
                    }
                }
                outcome.stopped();
            }
            Ok(())
        }));
        rx
    }
}

/// Spawn a task that linearizes and buffers morsels until a given a maximum chunk size is reached
/// and then distributes the columns amongst worker tasks.
fn buffer_and_distribute_columns_task(
    mut rx: Receiver<Morsel>,
    mut dist_tx: distributor_channel::Sender<(usize, usize, Column)>,
    chunk_size: usize,
    schema: SchemaRef,
) -> JoinHandle<PolarsResult<()>> {
    spawn(TaskPriority::High, async move {
        let mut seq = 0usize;
        let mut buffer = DataFrame::empty_with_schema(schema.as_ref());

        while let Ok(morsel) = rx.recv().traced_await().await {
            let (df, _, _, consume_token) = morsel.into_inner();
            // @NOTE: This also performs schema validation.
            buffer.vstack_mut(&df)?;

            while buffer.height() >= chunk_size {
                let df;
                (df, buffer) = buffer.split_at(buffer.height().min(chunk_size) as i64);

                for (i, column) in df.take_columns().into_iter().enumerate() {
                    if dist_tx.send((seq, i, column)).traced_await().await.is_err() {
                        return Ok(());
                    }
                }
                seq += 1;
            }
            drop(consume_token); // Increase the backpressure. Only free up a pipeline when the
                                 // morsel has started encoding in its entirety. This still
                                 // allows for parallelism of Morsels, but prevents large
                                 // bunches of Morsels from stacking up here.
        }

        // Flush the remaining rows.
        assert!(buffer.height() <= chunk_size);
        for (i, column) in buffer.take_columns().into_iter().enumerate() {
            if dist_tx.send((seq, i, column)).traced_await().await.is_err() {
                return Ok(());
            }
        }

        PolarsResult::Ok(())
    })
}

pub trait SinkNode {
    fn name(&self) -> &str;
    fn is_sink_input_parallel(&self) -> bool;
    fn do_maintain_order(&self) -> bool;

    fn spawn_sink(
        &mut self,
        num_pipelines: usize,
        recv_ports_recv: SinkRecvPort,
        state: &ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    );
    fn spawn_sink_once(
        &mut self,
        num_pipelines: usize,
        recv_ports_recv: SinkInputPort,
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
                    while let Some(ret) = started.join_handles.next().traced_await().await {
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
            SinkInputPort::Serial(recv.serial_with_maintain_order(self.sink.do_maintain_order()))
        };
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let (token, outcome) = PhaseOutcome::new_shared_wait(wait_group.token());
            if started
                .input_send
                .send((outcome, sink_input))
                .traced_await()
                .await
                .is_ok()
            {
                // Wait for the phase to finish.
                wait_group.wait().traced_await().await;
                if !token.did_finish() {
                    return Ok(());
                }

                if config::verbose() {
                    eprintln!("[{name}]: Last data sent.");
                }
            }

            // Either the task finished or some error occurred.
            while let Some(ret) = started.join_handles.next().traced_await().await {
                ret?;
            }

            Ok(())
        }));
    }
}

pub fn sync_on_close(sync_on_close: SyncOnCloseType, file: &mut fs::File) -> io::Result<()> {
    match sync_on_close {
        SyncOnCloseType::None => Ok(()),
        SyncOnCloseType::Data => file.sync_data(),
        SyncOnCloseType::All => file.sync_all(),
    }
}

pub async fn tokio_sync_on_close(
    sync_on_close: SyncOnCloseType,
    file: &mut tokio::fs::File,
) -> io::Result<()> {
    match sync_on_close {
        SyncOnCloseType::None => Ok(()),
        SyncOnCloseType::Data => file.sync_data().traced_await().await,
        SyncOnCloseType::All => file.sync_all().traced_await().await,
    }
}
