pub mod bridge;
pub mod extra_ops;
pub mod initialization;
pub mod reader_drivers;
pub mod reader_interface;

use std::sync::{Arc, Mutex};

use bridge::BridgeState;
use initialization::MultiScanTaskInitializer;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::cloud::CloudOptions;
use polars_io::predicates::ScanIOPredicate;
use polars_io::{RowIndex, pl_async};
use polars_plan::dsl::ScanSources;
use polars_plan::plans::hive::HivePartitionsDf;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::slice_enum::Slice;
use reader_interface::builder::FileReaderBuilder;

use crate::async_executor::{self, AbortOnDropHandle, TaskPriority};
use crate::async_primitives::connector;
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
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

    row_index: Option<RowIndex>,
    pre_slice: Option<Slice>,
    predicate: Option<ScanIOPredicate>,

    hive_parts: Option<Arc<HivePartitionsDf>>,
    include_file_paths: Option<PlSmallStr>,
    allow_missing_columns: bool,

    num_pipelines: usize,
    /// Number of readers to initialize concurrently. e.g. Parquet will want to fetch metadata in this
    /// step.
    n_readers_pre_init: usize,
}

enum MultiFileReaderState {
    Uninitialized {
        config: Box<MultiFileReaderConfig>,
    },

    Initialized {
        send_phase_tx_to_bridge: connector::Sender<(connector::Sender<Morsel>, WaitToken)>,
        bridge_state: Arc<Mutex<BridgeState>>,
        join_handle: AbortOnDropHandle<PolarsResult<()>>,
    },

    Finished,
}

pub struct MultiFileReader {
    name: String,
    state: MultiFileReaderState,
}

impl MultiFileReader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sources: ScanSources,
        file_reader_builder: Arc<dyn FileReaderBuilder>,
        cloud_options: Option<Arc<CloudOptions>>,

        final_output_schema: SchemaRef,
        projected_file_schema: SchemaRef,

        row_index: Option<RowIndex>,
        pre_slice: Option<Slice>,
        predicate: Option<ScanIOPredicate>,

        hive_parts: Option<Arc<HivePartitionsDf>>,
        include_file_paths: Option<PlSmallStr>,
        allow_missing_columns: bool,
    ) -> Self {
        // TODO: rename to multi-scan[]
        let name = format!("multi-file-reader[{:?}]", file_reader_builder.file_type());

        MultiFileReader {
            name,
            state: MultiFileReaderState::Uninitialized {
                config: Box::new(MultiFileReaderConfig {
                    sources,
                    file_reader_builder,
                    cloud_options,
                    final_output_schema,
                    projected_file_schema,
                    row_index,
                    pre_slice,
                    predicate,
                    hive_parts,
                    include_file_paths,
                    allow_missing_columns,
                    num_pipelines: 0,
                    n_readers_pre_init: 3,
                }),
            },
        }
    }
}

impl ComputeNode for MultiFileReader {
    fn name(&self) -> &str {
        &self.name
    }

    fn initialize(&mut self, num_pipelines: usize) {
        use MultiFileReaderState::*;

        match &mut self.state {
            Uninitialized { config } => config.num_pipelines = num_pipelines,
            Initialized { .. } | Finished => {},
        };
    }

    fn update_state(
        &mut self,
        recv: &mut [crate::graph::PortState],
        send: &mut [crate::graph::PortState],
    ) -> polars_error::PolarsResult<()> {
        use MultiFileReaderState::*;
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
                    .block_on(s.spawn_task(TaskPriority::High, self.state.refresh()))
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
        _state: &'s polars_expr::prelude::ExecutionState,
        join_handles: &mut Vec<crate::async_executor::JoinHandle<polars_error::PolarsResult<()>>>,
    ) {
        assert!(recv_ports.is_empty() && send_ports.len() == 1);

        let phase_morsel_tx = send_ports[0].take().unwrap().serial();

        join_handles.push(scope.spawn_task(TaskPriority::Low, async {
            use MultiFileReaderState::*;

            self.state.initialize();
            self.state.refresh().await?;

            match &mut self.state {
                Uninitialized { .. } => unreachable!(),

                Finished => return Ok(()),

                Initialized {
                    send_phase_tx_to_bridge,
                    ..
                } => {
                    let wait_group = WaitGroup::default();

                    if send_phase_tx_to_bridge
                        .send((phase_morsel_tx, wait_group.token()))
                        .await
                        .is_err()
                    {
                        // Bridge has disconnected
                        self.state.refresh().await?;
                        // In a panic scenario we can fall through to here with a non-finished state,
                        // so we forcibly set it to finished.
                        self.state = Finished;
                        return Ok(());
                    }

                    wait_group.wait().await;
                },
            }

            self.state.refresh().await
        }));
    }
}

impl MultiFileReaderState {
    /// Refresh the state. This checks the bridge state if `self` is initialized and updates accordingly.
    async fn refresh(&mut self) -> PolarsResult<()> {
        use MultiFileReaderState::*;

        // Take, so that if we error below the state will be left as finished.
        let slf = std::mem::replace(self, MultiFileReaderState::Finished);

        let slf = match slf {
            Uninitialized { .. } | Finished => slf,

            #[allow(clippy::blocks_in_conditions)]
            Initialized {
                send_phase_tx_to_bridge,
                bridge_state,
                join_handle,
            } => match { *bridge_state.lock().unwrap() } {
                BridgeState::NotYetStarted | BridgeState::Running => Initialized {
                    send_phase_tx_to_bridge,
                    bridge_state,
                    join_handle,
                },

                // Never the case: holding `send_phase_tx_to_bridge` guarantees this.
                BridgeState::Stopped(bridge::StopReason::ComputeNodeDisconnected) => unreachable!(),

                // If we are disconnected from the reader side, it could mean an error. Joining on
                // the handle will catch this.
                BridgeState::Stopped(bridge::StopReason::ReadersDisconnected) => {
                    join_handle.await?;
                    Finished
                },
            },
        };

        *self = slf;

        Ok(())
    }

    /// Initialize state if not yet initialized.
    fn initialize(&mut self) {
        use MultiFileReaderState::*;

        let slf = std::mem::replace(self, Finished);

        let Uninitialized { config } = slf else {
            *self = slf;
            return;
        };

        let (initializer, send_phase_tx_to_bridge, bridge_state) =
            MultiScanTaskInitializer::new(config);

        let join_handle = AbortOnDropHandle::new(initializer.spawn_background_tasks());

        *self = Initialized {
            send_phase_tx_to_bridge,
            bridge_state,
            join_handle,
        };
    }
}
