use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_error::{PolarsResult, polars_ensure};
use polars_io::metrics::IOMetrics;
use polars_io::pl_async;
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

use super::{ComputeNode, PortState};
use crate::async_executor;
use crate::async_primitives::connector;
use crate::execute::StreamingExecutionState;
use crate::metrics::MetricsBuilder;
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::nodes::TaskPriority;
use crate::nodes::io_sinks::components::partitioner::Partitioner;
use crate::nodes::io_sinks::config::{IOSinkNodeConfig, IOSinkTarget};
use crate::nodes::io_sinks::pipeline_initialization::partition_by::start_partition_sink_pipeline;
use crate::nodes::io_sinks::pipeline_initialization::single_file::start_single_file_sink_pipeline;
use crate::pipe::PortReceiver;
pub mod components;
pub mod config;
pub mod pipeline_initialization;
pub mod writers;

pub struct IOSinkNode {
    name: PlSmallStr,
    state: IOSinkNodeState,
    io_metrics: Option<Arc<IOMetrics>>,
    verbose: bool,
}

impl IOSinkNode {
    pub fn new(config: impl Into<Box<IOSinkNodeConfig>>) -> Self {
        let config = config.into();

        let target_type = match &config.target {
            IOSinkTarget::File(_) => "single-file",
            IOSinkTarget::Partitioned(p) => match &p.partitioner {
                Partitioner::Keyed(_) => "partition-keyed",
                Partitioner::FileSize => "partition-file-size",
            },
        };

        let extension = config.file_format.extension();

        let name = format_pl_smallstr!("io-sink[{target_type}[{extension}]]");
        let verbose = polars_core::config::verbose();

        IOSinkNode {
            name,
            state: IOSinkNodeState::Uninitialized { config },
            io_metrics: None,
            verbose,
        }
    }
}

impl ComputeNode for IOSinkNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn set_metrics_builder(&mut self, metrics_builder: MetricsBuilder) {
        self.io_metrics = Some(metrics_builder.new_io_metrics());
    }

    fn update_state(
        &mut self,
        recv: &mut [crate::graph::PortState],
        send: &mut [crate::graph::PortState],
        execution_state: &StreamingExecutionState,
    ) -> polars_error::PolarsResult<()> {
        assert_eq!(recv.len(), 1);
        assert!(send.is_empty());

        recv[0] = if recv[0] == PortState::Done {
            // Ensure initialize / writes empty file for empty output.
            self.state
                .initialize(&self.name, execution_state, self.io_metrics.clone())?;

            match std::mem::replace(&mut self.state, IOSinkNodeState::Finished) {
                IOSinkNodeState::Initialized {
                    phase_channel_tx,
                    task_handle,
                } => {
                    if self.verbose {
                        eprintln!(
                            "{}: Join on task_handle (recv PortState::Done)",
                            self.name()
                        );
                    }
                    drop(phase_channel_tx);
                    pl_async::get_runtime().block_on(task_handle)?;
                },
                IOSinkNodeState::Finished => {},
                IOSinkNodeState::Uninitialized { .. } => unreachable!(),
            };

            PortState::Done
        } else {
            polars_ensure!(
                !matches!(self.state, IOSinkNodeState::Finished),
                ComputeError:
                "unreachable: IO sink node state is 'Finished', but recv port \
                state is not 'Done'."
            );

            PortState::Ready
        };

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s crate::async_executor::TaskScope<'s, 'env>,
        recv_ports: &mut [Option<crate::pipe::RecvPort<'_>>],
        send_ports: &mut [Option<crate::pipe::SendPort<'_>>],
        execution_state: &'s StreamingExecutionState,
        join_handles: &mut Vec<crate::async_executor::JoinHandle<polars_error::PolarsResult<()>>>,
    ) {
        assert_eq!(recv_ports.len(), 1);
        assert!(send_ports.is_empty());

        let phase_morsel_rx = recv_ports[0].take().unwrap().serial();

        join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
            self.state
                .initialize(&self.name, execution_state, self.io_metrics.clone())?;

            let IOSinkNodeState::Initialized {
                phase_channel_tx, ..
            } = &mut self.state
            else {
                unreachable!()
            };

            if phase_channel_tx.send(phase_morsel_rx).await.is_err() {
                let IOSinkNodeState::Initialized {
                    phase_channel_tx,
                    task_handle,
                } = std::mem::replace(&mut self.state, IOSinkNodeState::Finished)
                else {
                    unreachable!()
                };

                if self.verbose {
                    eprintln!(
                        "{}: Join on task_handle (phase_channel_tx Err)",
                        self.name()
                    );
                }

                drop(phase_channel_tx);

                return Err(task_handle.await.unwrap_err());
            }

            Ok(())
        }));
    }
}

enum IOSinkNodeState {
    Uninitialized {
        config: Box<IOSinkNodeConfig>,
    },

    Initialized {
        phase_channel_tx: connector::Sender<PortReceiver>,
        /// Join handle for all background tasks.
        task_handle: async_executor::AbortOnDropHandle<PolarsResult<()>>,
    },

    Finished,
}

impl IOSinkNodeState {
    /// Initialize state if not yet initialized.
    fn initialize(
        &mut self,
        node_name: &PlSmallStr,
        execution_state: &StreamingExecutionState,
        io_metrics: Option<Arc<IOMetrics>>,
    ) -> PolarsResult<()> {
        use IOSinkNodeState::*;

        if !matches!(self, Self::Uninitialized { .. }) {
            return Ok(());
        }

        let Uninitialized { config } = std::mem::replace(self, Finished) else {
            unreachable!()
        };

        let (phase_channel_tx, mut phase_channel_rx) = connector::connector::<PortReceiver>();
        let (mut multi_phase_tx, multi_phase_rx) = connector::connector();

        let _ = multi_phase_tx.try_send(Morsel::new(
            DataFrame::empty_with_arc_schema(config.input_schema.clone()),
            MorselSeq::new(0),
            SourceToken::default(),
        ));

        async_executor::spawn(TaskPriority::High, async move {
            let mut morsel_seq: u64 = 1;

            while let Ok(mut phase_rx) = phase_channel_rx.recv().await {
                while let Ok(mut morsel) = phase_rx.recv().await {
                    morsel.set_seq(MorselSeq::new(morsel_seq));
                    morsel_seq = morsel_seq.saturating_add(1);

                    if multi_phase_tx.send(morsel).await.is_err() {
                        break;
                    }
                }
            }
        });

        let task_handle = match &config.target {
            IOSinkTarget::File(_) => start_single_file_sink_pipeline(
                node_name.clone(),
                multi_phase_rx,
                *config,
                execution_state,
                io_metrics,
            )?,

            IOSinkTarget::Partitioned { .. } => start_partition_sink_pipeline(
                node_name,
                multi_phase_rx,
                *config,
                execution_state,
                io_metrics,
            )?,
        };

        *self = Initialized {
            phase_channel_tx,
            task_handle,
        };

        Ok(())
    }
}
