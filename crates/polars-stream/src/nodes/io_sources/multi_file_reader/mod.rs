pub mod bridge;
pub mod extra_ops;
pub mod initialization;
pub mod post_apply_pipeline;
pub mod reader_interface;
pub mod reader_pipelines;

use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, Mutex};

use bridge::BridgeState;
use initialization::MultiScanTaskInitializer;
use polars_core::config;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::cloud::CloudOptions;
use polars_io::predicates::ScanIOPredicate;
use polars_io::{RowIndex, pl_async};
use polars_plan::dsl::ScanSources;
use polars_plan::plans::hive::HivePartitionsDf;
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::slice_enum::Slice;
use reader_interface::builder::FileReaderBuilder;

use crate::async_executor::{self, AbortOnDropHandle, TaskPriority};
use crate::async_primitives::connector;
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::Morsel;
use crate::nodes::ComputeNode;

// Some parts are called MultiFileReader for now to avoid conflict with existing MultiScan.

pub struct MultiFileReaderConfig {
    sources: ScanSources,
    file_reader_builder: Arc<dyn FileReaderBuilder>,
    cloud_options: Option<Arc<CloudOptions>>,

    /// Final output schema of MultiScan node. Includes all e.g. row index / missing columns / file paths / hive etc.
    final_output_schema: SchemaRef,
    /// Columns to be projected from the file.
    projected_file_schema: SchemaRef,
    /// Full schema of the file. Used for complaining about extra columns.
    full_file_schema: SchemaRef,

    row_index: Option<RowIndex>,
    pre_slice: Option<Slice>,
    predicate: Option<ScanIOPredicate>,

    hive_parts: Option<Arc<HivePartitionsDf>>,
    include_file_paths: Option<PlSmallStr>,
    allow_missing_columns: bool,

    num_pipelines: AtomicUsize,
    /// Number of readers to initialize concurrently. e.g. Parquet will want to fetch metadata in this
    /// step.
    n_readers_pre_init: usize,

    verbose: AtomicBool,
}

impl MultiFileReaderConfig {
    fn num_pipelines(&self) -> usize {
        self.num_pipelines
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn verbose(&self) -> bool {
        self.verbose.load(std::sync::atomic::Ordering::Relaxed)
    }
}

enum MultiScanState {
    Uninitialized {
        config: Arc<MultiFileReaderConfig>,
    },

    Initialized {
        send_phase_tx_to_bridge: connector::Sender<(connector::Sender<Morsel>, WaitToken)>,
        /// Wait group sent to the bridge, only dropped when a disconnect happens at the bridge.
        wait_group: WaitGroup,
        bridge_state: Arc<Mutex<BridgeState>>,
        /// Single join handle for all background tasks. Note, this does not include the bridge.
        join_handle: AbortOnDropHandle<PolarsResult<()>>,
    },

    Finished,
}

pub struct MultiFileReader {
    name: PlSmallStr,
    state: MultiScanState,
    verbose: bool,
}

impl MultiFileReader {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        sources: ScanSources,
        file_reader_builder: Arc<dyn FileReaderBuilder>,
        cloud_options: Option<Arc<CloudOptions>>,

        final_output_schema: SchemaRef,
        projected_file_schema: SchemaRef,
        full_file_schema: SchemaRef,

        row_index: Option<RowIndex>,
        pre_slice: Option<Slice>,
        predicate: Option<ScanIOPredicate>,

        hive_parts: Option<Arc<HivePartitionsDf>>,
        include_file_paths: Option<PlSmallStr>,
        allow_missing_columns: bool,
    ) -> Self {
        let name = format_pl_smallstr!("MultiScan[{}]", file_reader_builder.reader_name());

        MultiFileReader {
            name,
            state: MultiScanState::Uninitialized {
                config: Arc::new(MultiFileReaderConfig {
                    sources,
                    file_reader_builder,
                    cloud_options,
                    final_output_schema,
                    projected_file_schema,
                    full_file_schema,
                    row_index,
                    pre_slice,
                    predicate,
                    hive_parts,
                    include_file_paths,
                    allow_missing_columns,
                    num_pipelines: AtomicUsize::new(0),
                    n_readers_pre_init: 3,
                    verbose: AtomicBool::new(false),
                }),
            },
            verbose: config::verbose(),
        }
    }
}

impl ComputeNode for MultiFileReader {
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
        assert!(send.len() == 1);

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
        state: &StreamingExecutionState,
        join_handles: &mut Vec<crate::async_executor::JoinHandle<polars_error::PolarsResult<()>>>,
    ) {
        assert!(recv_ports.is_empty() && send_ports.len() == 1);

        let phase_morsel_tx = send_ports[0].take().unwrap().serial();
        let num_pipelines = state.num_pipelines;
        let verbose = self.verbose;

        join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
            use MultiScanState::*;

            self.state.initialize(num_pipelines, verbose);
            self.state.refresh(verbose).await?;

            match &mut self.state {
                Uninitialized { .. } => unreachable!(),

                Finished => return Ok(()),

                Initialized {
                    send_phase_tx_to_bridge,
                    wait_group,
                    ..
                } => {
                    use crate::async_primitives::connector::SendError;

                    match send_phase_tx_to_bridge.try_send((phase_morsel_tx, wait_group.token())) {
                        Ok(_) => wait_group.wait().await,

                        // Should never: We only send the next value once the wait token is dropped.
                        Err(SendError::Full(_)) => unreachable!(),

                        // Bridge has disconnected from the reader side. We know this because
                        // we are still holding `send_phase_tx_to_bridge`.
                        Err(SendError::Closed(_)) => {
                            if verbose {
                                eprintln!("[MultiFileReader]: Bridge disconnected")
                            }

                            let Initialized { join_handle, .. } =
                                std::mem::replace(&mut self.state, Finished)
                            else {
                                unreachable!()
                            };

                            join_handle.await?;

                            return Ok(());
                        },
                    }
                },
            }

            self.state.refresh(verbose).await
        }));
    }
}

impl MultiScanState {
    /// Refresh the state. This checks the bridge state if `self` is initialized and updates accordingly.
    async fn refresh(&mut self, verbose: bool) -> PolarsResult<()> {
        use MultiScanState::*;
        use bridge::StopReason;

        // Take, so that if we error below the state will be left as finished.
        let slf = std::mem::replace(self, MultiScanState::Finished);

        let slf = match slf {
            Uninitialized { .. } | Finished => slf,

            #[expect(clippy::blocks_in_conditions)]
            Initialized {
                send_phase_tx_to_bridge,
                wait_group,
                bridge_state,
                join_handle,
            } => match { *bridge_state.lock().unwrap() } {
                BridgeState::NotYetStarted | BridgeState::Running => Initialized {
                    send_phase_tx_to_bridge,
                    wait_group,
                    bridge_state,
                    join_handle,
                },

                // Never the case: holding `send_phase_tx_to_bridge` guarantees this.
                BridgeState::Stopped(StopReason::ComputeNodeDisconnected) => unreachable!(),

                // If we are disconnected from the reader side, it could mean an error. Joining on
                // the handle should catch this.
                BridgeState::Stopped(StopReason::ReadersDisconnected) => {
                    if verbose {
                        eprintln!("[MultiScanState]: Readers disconnected")
                    }

                    *self = Finished;
                    join_handle.await?;
                    Finished
                },
            },
        };

        *self = slf;

        Ok(())
    }

    /// Initialize state if not yet initialized.
    fn initialize(&mut self, num_pipelines: usize, verbose: bool) {
        use MultiScanState::*;

        let slf = std::mem::replace(self, Finished);

        let Uninitialized { config } = slf else {
            *self = slf;
            return;
        };

        config
            .num_pipelines
            .store(num_pipelines, std::sync::atomic::Ordering::Relaxed);
        config
            .verbose
            .store(verbose, std::sync::atomic::Ordering::Relaxed);

        let (join_handle, send_phase_tx_to_bridge, bridge_state) =
            MultiScanTaskInitializer::new(config).spawn_background_tasks();

        let wait_group = WaitGroup::default();

        *self = Initialized {
            send_phase_tx_to_bridge,
            wait_group,
            bridge_state,
            join_handle,
        };
    }
}
