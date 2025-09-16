use std::pin::Pin;
use std::sync::{Arc, LazyLock, Mutex};

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use polars_core::config;
use polars_core::frame::DataFrame;
use polars_core::prelude::Column;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;

use self::metrics::WriteMetrics;
use super::{ComputeNode, JoinHandle, Morsel, PortState, RecvPort, SendPort, TaskScope};
use crate::async_executor::{AbortOnDropHandle, spawn};
use crate::async_primitives::connector::{Receiver, Sender, connector};
use crate::async_primitives::distributor_channel;
use crate::async_primitives::linearizer::{Inserter, Linearizer};
use crate::async_primitives::wait_group::WaitGroup;
use crate::execute::StreamingExecutionState;
use crate::nodes::TaskPriority;

mod metrics;
mod phase;
use phase::PhaseOutcome;

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
static DEFAULT_SINK_LINEARIZER_BUFFER_SIZE: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("POLARS_DEFAULT_SINK_LINEARIZER_BUFFER_SIZE")
        .map(|x| x.parse().unwrap())
        .unwrap_or(1)
});

static DEFAULT_SINK_DISTRIBUTOR_BUFFER_SIZE: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("POLARS_DEFAULT_SINK_DISTRIBUTOR_BUFFER_SIZE")
        .map(|x| x.parse().unwrap())
        .unwrap_or(1)
});

pub enum SinkInputPort {
    Serial(Receiver<Morsel>),
    Parallel(Vec<Receiver<Morsel>>),
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

/// Spawn a task that linearizes and buffers morsels until a given a maximum chunk size is reached
/// and then distributes the columns amongst worker tasks.
fn buffer_and_distribute_columns_task(
    mut recv_port_rx: Receiver<(PhaseOutcome, SinkInputPort)>,
    mut dist_tx: distributor_channel::Sender<(usize, usize, Column)>,
    chunk_size: usize,
    schema: SchemaRef,
    metrics: Arc<Mutex<Option<WriteMetrics>>>,
) -> JoinHandle<PolarsResult<()>> {
    spawn(TaskPriority::High, async move {
        let mut seq = 0usize;
        let mut buffer = DataFrame::empty_with_schema(schema.as_ref());

        let mut metrics_ = metrics.lock().unwrap().take();
        while let Ok((outcome, rx)) = recv_port_rx.recv().await {
            let mut rx = rx.serial();
            while let Ok(morsel) = rx.recv().await {
                let (df, _, _, consume_token) = morsel.into_inner();

                if let Some(metrics) = metrics_.as_mut() {
                    metrics.append(&df)?;
                }

                // @NOTE: This also performs schema validation.
                buffer.vstack_mut(&df)?;

                while buffer.height() >= chunk_size {
                    let df;
                    (df, buffer) = buffer.split_at(buffer.height().min(chunk_size) as i64);

                    for (i, column) in df.take_columns().into_iter().enumerate() {
                        if dist_tx.send((seq, i, column)).await.is_err() {
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

            outcome.stopped();
        }
        if let Some(metrics_) = metrics_ {
            *metrics.lock().unwrap() = Some(metrics_);
        }

        // Don't write an empty row group at the end.
        if buffer.is_empty() {
            return Ok(());
        }

        // Flush the remaining rows.
        assert!(buffer.height() <= chunk_size);
        for (i, column) in buffer.take_columns().into_iter().enumerate() {
            if dist_tx.send((seq, i, column)).await.is_err() {
                return Ok(());
            }
        }

        PolarsResult::Ok(())
    })
}

#[allow(clippy::type_complexity)]
pub fn parallelize_receive_task<T: Ord + Send + 'static>(
    join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    mut recv_port_rx: Receiver<(PhaseOutcome, SinkInputPort)>,
    num_pipelines: usize,
    maintain_order: bool,
    mut io_tx: Sender<Linearizer<T>>,
) -> Vec<Receiver<(Receiver<Morsel>, Inserter<T>)>> {
    // Phase Handling Task -> Encode Tasks.
    let (mut pass_txs, pass_rxs) = (0..num_pipelines)
        .map(|_| connector())
        .collect::<(Vec<_>, Vec<_>)>();

    join_handles.push(spawn(TaskPriority::High, async move {
        while let Ok((outcome, port_rxs)) = recv_port_rx.recv().await {
            let port_rxs = port_rxs.parallel();
            let (lin_rx, lin_txs) = Linearizer::<T>::new_with_maintain_order(
                num_pipelines,
                *DEFAULT_SINK_LINEARIZER_BUFFER_SIZE,
                maintain_order,
            );

            for ((pass_tx, port_rx), lin_tx) in pass_txs.iter_mut().zip(port_rxs).zip(lin_txs) {
                if pass_tx.send((port_rx, lin_tx)).await.is_err() {
                    return Ok(());
                }
            }
            if io_tx.send(lin_rx).await.is_err() {
                return Ok(());
            }

            outcome.stopped();
        }

        Ok(())
    }));

    pass_rxs
}

pub trait SinkNode {
    fn name(&self) -> &str;

    fn is_sink_input_parallel(&self) -> bool;

    fn do_maintain_order(&self) -> bool {
        true
    }

    fn spawn_sink(
        &mut self,
        recv_ports_recv: Receiver<(PhaseOutcome, SinkInputPort)>,
        state: &StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    );

    /// Callback that gets called once before the sink is spawned.
    fn initialize(&mut self, state: &StreamingExecutionState) -> PolarsResult<()> {
        _ = state;
        Ok(())
    }

    /// Callback for when the query has finished successfully.
    ///
    /// This should only be called when the writing is finished and all the join handles have been
    /// awaited.
    fn finalize(
        &mut self,
        state: &StreamingExecutionState,
    ) -> Option<Pin<Box<dyn Future<Output = PolarsResult<()>> + Send>>> {
        _ = state;
        None
    }

    /// Fetch metrics for a specific sink.
    ///
    /// This should only be called when the writing is finished and all the join handles have been
    /// awaited.
    fn get_metrics(&self) -> PolarsResult<Option<WriteMetrics>> {
        Ok(None)
    }
}

/// The state needed to manage a spawned [`SinkNode`].
struct StartedSinkComputeNode {
    input_send: Sender<(PhaseOutcome, SinkInputPort)>,
    join_handles: FuturesUnordered<AbortOnDropHandle<PolarsResult<()>>>,
}

/// A [`ComputeNode`] to wrap a [`SinkNode`].
pub struct SinkComputeNode {
    sink: Box<dyn SinkNode + Send>,
    started: Option<StartedSinkComputeNode>,
    state: SinkState,
}

enum SinkState {
    /// Initial state of a [`SinkComputeNode`].
    ///
    /// This still requires `sink.initialize` to be called on the `SinkNode`.
    Uninitialized,

    /// Active state of a [`SinkComputeNode`].
    ///
    /// When finished, the `sink.finalize` method should be called.
    Initialized,

    /// Final state for the [`SinkComputeNode`].
    ///
    /// Receive port is Done and [`SinkNode`] is finalized.
    Finished,
}

impl SinkComputeNode {
    pub fn new(sink: Box<dyn SinkNode + Send>) -> Self {
        Self {
            sink,
            started: None,
            state: SinkState::Uninitialized,
        }
    }
}

impl<T: SinkNode + Send + 'static> From<T> for SinkComputeNode {
    fn from(value: T) -> Self {
        Self::new(Box::new(value))
    }
}

impl ComputeNode for SinkComputeNode {
    fn name(&self) -> &str {
        self.sink.name()
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        _send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        // Ensure that initialize is only called once.
        if matches!(self.state, SinkState::Uninitialized) {
            self.sink.initialize(state)?;
            self.state = SinkState::Initialized;
        }

        if recv[0] != PortState::Done {
            recv[0] = PortState::Ready;
        }

        if recv[0] == PortState::Done && !matches!(self.state, SinkState::Finished) {
            let started = self.started.take();
            let finalize = self.sink.finalize(state);

            state.spawn_subphase_task(async move {
                // We need to join on all started tasks before finalizing the node because the
                // unfinished tasks might still need access to the node.
                //
                // Note, that if the sink never received any data, this `started` might be None.
                // However, we do still need to finalize the node otherwise no file will be
                // created.
                if let Some(mut started) = started {
                    drop(started.input_send);
                    // Either the task finished or some error occurred.
                    while let Some(ret) = started.join_handles.next().await {
                        ret?;
                    }
                }

                if let Some(finalize) = finalize {
                    finalize.await?;
                }

                PolarsResult::Ok(())
            });

            self.state = SinkState::Finished;
        }

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert_eq!(recv_ports.len(), 1);
        assert!(send_ports.is_empty());

        let name = self.name().to_string();
        let started = self.started.get_or_insert_with(|| {
            let (tx, rx) = connector();
            let mut join_handles = Vec::new();

            self.sink.spawn_sink(rx, state, &mut join_handles);
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

    fn get_output(&mut self) -> PolarsResult<Option<DataFrame>> {
        Ok(None)
    }
}
