pub mod predicate;
pub mod projection;
pub mod slice;

use std::sync::{Arc, Mutex};

use polars_error::PolarsResult;

use super::MultiFileReaderConfig;
use super::bridge::BridgeState;
use crate::async_executor::{self, AbortOnDropHandle, TaskPriority};
use crate::async_primitives::connector::{self};
use crate::async_primitives::wait_group::WaitToken;
use crate::morsel::Morsel;
use crate::nodes::io_sources::multi_file_reader::bridge::spawn_bridge;

pub struct MultiScanTaskInitializer {
    pub(super) config: Arc<MultiFileReaderConfig>,
}

impl MultiScanTaskInitializer {
    pub fn new(config: Arc<MultiFileReaderConfig>) -> Self {
        Self { config }
    }

    #[expect(clippy::type_complexity)]
    pub fn spawn_background_tasks(
        self,
    ) -> (
        AbortOnDropHandle<PolarsResult<()>>,
        connector::Sender<(connector::Sender<Morsel>, WaitToken)>,
        Arc<Mutex<BridgeState>>,
    ) {
        assert!(self.config.num_pipelines() > 0);
        let verbose = self.config.verbose;

        if verbose {
            eprintln!(
                "[MultiScanTaskInitializer]: spawn_background_tasks(), {} sources, reader name: {}, {:?}",
                self.config.sources.len(),
                self.config.file_reader_builder.reader_name(),
                self.config.file_reader_builder.reader_capabilities(),
            );

            eprintln!(
                "[MultiScanTaskInitializer]: n_readers_pre_init: {}, max_concurrent_scans: {}",
                self.config.n_readers_pre_init(),
                self.config.max_concurrent_scans(),
            );
        }

        let bridge_state = Arc::new(Mutex::new(BridgeState::NotYetStarted));

        let (bridge_handle, bridge_recv_port_tx, send_phase_chan_to_bridge) =
            spawn_bridge(bridge_state.clone());

        let verbose = self.config.verbose;

        let background_tasks_handle = AbortOnDropHandle::new(async_executor::spawn(
            TaskPriority::Low,
            async move {
                let (skip_files_mask, predicate) = self.initialize_predicate()?;

                if verbose {
                    eprintln!(
                        "[MultiScanTaskInitializer]: \
                        predicate: {:?}, \
                        skip files mask: {:?}, \
                        predicate to reader: {:?}",
                        self.config.predicate.is_some().then_some("<predicate>"),
                        skip_files_mask.is_some().then_some("<skip_files>"),
                        predicate.is_some().then_some("<predicate>"),
                    )
                }

                #[expect(clippy::never_loop)]
                loop {
                    if skip_files_mask
                        .as_ref()
                        .is_some_and(|x| x.unset_bits() == 0)
                    {
                        if verbose {
                            eprintln!(
                                "[MultiScanTaskInitializer]: early return (skip_files_mask / predicate)"
                            )
                        }
                    } else if self.config.pre_slice.as_ref().is_some_and(|x| x.len() == 0) {
                        if cfg!(debug_assertions) {
                            panic!("should quit earlier");
                        }

                        if verbose {
                            eprintln!(
                                "[MultiScanTaskInitializer]: early return (pre_slice.len == 0)"
                            )
                        }
                    } else {
                        break;
                    }

                    return Ok(());
                }

                let predicate = predicate.cloned();

                self.init_and_run(bridge_recv_port_tx, skip_files_mask, predicate)
                    .await?
                    .await?;

                bridge_handle.await;

                Ok(())
            },
        ));

        (
            background_tasks_handle,
            send_phase_chan_to_bridge,
            bridge_state,
        )
    }
}
