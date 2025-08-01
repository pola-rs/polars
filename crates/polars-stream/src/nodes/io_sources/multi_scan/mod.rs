pub mod components;
pub mod config;
pub mod functions;
mod pipeline;
pub mod reader_interface;

use std::sync::{Arc, Mutex};

use pipeline::initialization::initialize_multi_scan_pipeline;
use polars_error::PolarsResult;
use polars_io::pl_async;
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

use crate::async_executor::{self, AbortOnDropHandle, TaskPriority};
use crate::async_primitives::connector;
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::Morsel;
use crate::nodes::ComputeNode;
use crate::nodes::io_sources::multi_scan::components::bridge::BridgeState;
use crate::nodes::io_sources::multi_scan::config::MultiScanConfig;
use crate::nodes::io_sources::multi_scan::functions::{
    calc_max_concurrent_scans, calc_n_readers_pre_init,
};
use crate::nodes::io_sources::multi_scan::pipeline::models::InitializedPipelineState;

pub struct MultiScan {
    name: PlSmallStr,
    state: MultiScanState,
    verbose: bool,
}

impl MultiScan {
    pub fn new(config: Arc<MultiScanConfig>) -> Self {
        let name = format_pl_smallstr!("multi-scan[{}]", config.file_reader_builder.reader_name());
        let verbose = config.verbose;

        MultiScan {
            name,
            state: MultiScanState::Uninitialized { config },
            verbose,
        }
    }
}

impl ComputeNode for MultiScan {
    fn name(&self) -> &str {
        &self.name
    }

    fn update_state(
        &mut self,
        recv: &mut [crate::graph::PortState],
        send: &mut [crate::graph::PortState],
        _state: &StreamingExecutionState,
    ) -> polars_error::PolarsResult<()> {
        use MultiScanState::*;
        assert!(recv.is_empty());
        assert_eq!(send.len(), 1);

        send[0] = if send[0] == PortState::Done {
            self.state = Finished;

            PortState::Done
        } else {
            // Refresh first - in case there is an error we end here instead of ending when we go
            // into spawn.
            async_executor::task_scope(|s| {
                pl_async::get_runtime()
                    .block_on(s.spawn_task(TaskPriority::High, self.state.refresh(self.verbose)))
            })?;

            match self.state {
                Uninitialized { .. } | Initialized { .. } => PortState::Ready,
                Finished => PortState::Done,
            }
        };

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s crate::async_executor::TaskScope<'s, 'env>,
        recv_ports: &mut [Option<crate::pipe::RecvPort<'_>>],
        send_ports: &mut [Option<crate::pipe::SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<crate::async_executor::JoinHandle<polars_error::PolarsResult<()>>>,
    ) {
        assert!(recv_ports.is_empty() && send_ports.len() == 1);

        let phase_morsel_tx = send_ports[0].take().unwrap().serial();
        let verbose = self.verbose;

        join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
            use MultiScanState::*;

            self.state.initialize(state);
            self.state.refresh(verbose).await?;

            match &mut self.state {
                Uninitialized { .. } => unreachable!(),

                Finished => return Ok(()),

                Initialized {
                    phase_channel_tx,
                    wait_group,
                    ..
                } => {
                    use crate::async_primitives::connector::SendError;

                    match phase_channel_tx.try_send((phase_morsel_tx, wait_group.token())) {
                        Ok(_) => wait_group.wait().await,

                        // Should never: We only send the next value once the wait token is dropped.
                        Err(SendError::Full(_)) => unreachable!(),

                        // Bridge has disconnected from the reader side. We know this because
                        // we are still holding `phase_channel_tx`.
                        Err(SendError::Closed(_)) => {
                            if verbose {
                                eprintln!("[MultiScan]: Bridge disconnected")
                            }

                            let Initialized { task_handle, .. } =
                                std::mem::replace(&mut self.state, Finished)
                            else {
                                unreachable!()
                            };

                            task_handle.await?;

                            return Ok(());
                        },
                    }
                },
            }

            self.state.refresh(verbose).await
        }));
    }
}

enum MultiScanState {
    Uninitialized {
        config: Arc<MultiScanConfig>,
    },

    Initialized {
        phase_channel_tx: connector::Sender<(connector::Sender<Morsel>, WaitToken)>,
        /// Wait group sent to the bridge, only dropped when a disconnect happens at the bridge.
        wait_group: WaitGroup,
        bridge_state: Arc<Mutex<BridgeState>>,
        /// Single join handle for all background tasks. Note, this does not include the bridge.
        task_handle: AbortOnDropHandle<PolarsResult<()>>,
    },

    Finished,
}

impl MultiScanState {
    /// Initialize state if not yet initialized.
    fn initialize(&mut self, execution_state: &StreamingExecutionState) {
        use MultiScanState::*;

        let slf = std::mem::replace(self, Finished);

        let Uninitialized { config } = slf else {
            *self = slf;
            return;
        };

        config
            .file_reader_builder
            .set_execution_state(execution_state);

        let num_pipelines = execution_state.num_pipelines;

        config.num_pipelines.store(num_pipelines);

        config.n_readers_pre_init.store(calc_n_readers_pre_init(
            num_pipelines,
            config.sources.len(),
            config.pre_slice.as_ref(),
        ));

        config.max_concurrent_scans.store(calc_max_concurrent_scans(
            num_pipelines,
            config.sources.len(),
        ));

        let InitializedPipelineState {
            task_handle,
            phase_channel_tx,
            bridge_state,
        } = initialize_multi_scan_pipeline(config);

        let wait_group = WaitGroup::default();

        *self = Initialized {
            phase_channel_tx,
            wait_group,
            bridge_state,
            task_handle,
        };
    }

    /// Refresh the state. This checks the bridge state if `self` is initialized and updates accordingly.
    async fn refresh(&mut self, verbose: bool) -> PolarsResult<()> {
        use MultiScanState::*;
        use components::bridge::StopReason;

        // Take, so that if we error below the state will be left as finished.
        let slf = std::mem::replace(self, MultiScanState::Finished);

        let slf = match slf {
            Uninitialized { .. } | Finished => slf,

            #[expect(clippy::blocks_in_conditions)]
            Initialized {
                phase_channel_tx,
                wait_group,
                bridge_state,
                task_handle,
            } => match { *bridge_state.lock().unwrap() } {
                BridgeState::NotYetStarted | BridgeState::Running => Initialized {
                    phase_channel_tx,
                    wait_group,
                    bridge_state,
                    task_handle,
                },

                // Never the case: holding `phase_channel_tx` guarantees this.
                BridgeState::Stopped(StopReason::ComputeNodeDisconnected) => unreachable!(),

                // If we are disconnected from the reader side, it could mean an error. Joining on
                // the handle should catch this.
                BridgeState::Stopped(StopReason::ReadersDisconnected) => {
                    if verbose {
                        eprintln!("[MultiScanState]: Readers disconnected")
                    }

                    *self = Finished;
                    task_handle.await?;
                    Finished
                },
            },
        };

        *self = slf;

        Ok(())
    }
}
