use polars_error::PolarsResult;

use crate::async_executor::AbortOnDropHandle;
use crate::async_primitives::connector;
use crate::async_primitives::wait_group::WaitToken;
use crate::nodes::io_sources::multi_scan::components::bridge::BridgeRecvPort;
use crate::nodes::io_sources::multi_scan::pipeline::models::StartedReaderState;

pub struct AttachReaderToBridge {
    /// The size of the channel controls how many readers are run in parallel.
    pub started_reader_rx: tokio::sync::mpsc::Receiver<(
        AbortOnDropHandle<PolarsResult<StartedReaderState>>,
        WaitToken,
    )>,
    pub bridge_recv_port_tx: connector::Sender<BridgeRecvPort>,
    pub verbose: bool,
}

impl AttachReaderToBridge {
    pub async fn run(self) -> PolarsResult<()> {
        let AttachReaderToBridge {
            mut started_reader_rx,
            mut bridge_recv_port_tx,
            verbose,
        } = self;

        let mut n_readers_received: usize = 0;

        while let Some((init_task_handle, wait_token)) = started_reader_rx.recv().await {
            n_readers_received = n_readers_received.saturating_add(1);

            if verbose {
                eprintln!(
                    "[AttachReaderToBridge]: received reader (n_readers_received: {n_readers_received})",
                );
            }

            let StartedReaderState {
                bridge_recv_port,
                post_apply_pipeline_handle,
                reader_handle,
            } = init_task_handle.await?;

            if bridge_recv_port_tx.send(bridge_recv_port).await.is_err() {
                break;
            }

            drop(wait_token);
            reader_handle.await?;

            if let Some(handle) = post_apply_pipeline_handle {
                handle.await?;
            }
        }

        Ok(())
    }
}
