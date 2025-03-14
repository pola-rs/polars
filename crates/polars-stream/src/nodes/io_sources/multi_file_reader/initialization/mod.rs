pub mod predicate;
pub mod projection;
pub mod slice;

use std::sync::{Arc, Mutex};

use polars_error::PolarsResult;

use super::MultiFileReaderConfig;
use super::bridge::BridgeState;
use crate::async_executor::{self, JoinHandle, TaskPriority};
use crate::async_primitives::connector::{self};
use crate::async_primitives::wait_group::WaitToken;
use crate::morsel::Morsel;

pub struct MultiScanTaskInitializer {
    pub(super) config: Box<MultiFileReaderConfig>,
    /// If we resolved a negative slice we store some of the readers here. For Parquet this can
    /// avoid a duplicate metadata fetch/decode.
    pub(super) phase_channel_receiver:
        Option<connector::Receiver<(connector::Sender<Morsel>, WaitToken)>>,
    pub(super) bridge_state: Arc<Mutex<BridgeState>>,
}

impl MultiScanTaskInitializer {
    #[allow(clippy::type_complexity)]
    pub fn new(
        config: Box<MultiFileReaderConfig>,
    ) -> (
        Self,
        connector::Sender<(connector::Sender<Morsel>, WaitToken)>,
        Arc<Mutex<BridgeState>>,
    ) {
        let (phase_channel_sender, phase_channel_receiver) = connector::connector();
        let bridge_state = Arc::new(Mutex::new(BridgeState::NotYetStarted));

        (
            Self {
                config,
                phase_channel_receiver: Some(phase_channel_receiver),
                bridge_state: bridge_state.clone(),
            },
            phase_channel_sender,
            bridge_state,
        )
    }

    pub fn spawn_background_tasks(self) -> JoinHandle<PolarsResult<()>> {
        assert!(self.config.num_pipelines > 0);

        async_executor::spawn(TaskPriority::Low, async move {
            self.init_and_run_generic_loop().await?.await
        })
    }
}
