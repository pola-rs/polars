use std::sync::{Arc, Mutex};

use super::reader_interface::output::FileReaderOutputRecv;
use crate::async_executor::{self, TaskPriority};
use crate::async_primitives::connector;
use crate::async_primitives::wait_group::WaitToken;
use crate::morsel::{Morsel, MorselSeq, SourceToken};

pub fn spawn_bridge(
    bridge_state: Arc<Mutex<BridgeState>>,
) -> (
    // For attaching file reader output port
    connector::Sender<FileReaderOutputRecv>,
    // For attaching the multi scan node output port
    connector::Sender<(connector::Sender<Morsel>, WaitToken)>,
) {
    let (incoming_tx, incoming) = connector::connector();
    let (outgoing_tx, outgoing) = connector::connector();

    async_executor::spawn(
        TaskPriority::Low,
        Bridge {
            incoming,
            outgoing,
            bridge_state,
            source_token: SourceToken::new(),
        }
        .run(),
    );

    (incoming_tx, outgoing_tx)
}

/// Bridge to pass morsels from individual readers into the output port of the multi scan node.
struct Bridge {
    incoming: connector::Receiver<FileReaderOutputRecv>,
    outgoing: connector::Receiver<(connector::Sender<Morsel>, WaitToken)>,
    bridge_state: Arc<Mutex<BridgeState>>,
    source_token: SourceToken,
}

#[derive(Copy, Clone)]
pub enum BridgeState {
    NotYetStarted,
    Running,
    Stopped(StopReason),
}

#[derive(Copy, Clone)]
pub enum StopReason {
    /// Disconnected from the reader side. The driver handle should be joined on in this case to
    /// determine if the readers disconnected due to an error.
    ReadersDisconnected,
    /// Disconnected from the multi scan ComputeNode.
    ComputeNodeDisconnected,
}

/// Single lane bridge for morsels.
impl Bridge {
    async fn run(mut self) {
        let Ok(rx) = self.incoming.recv().await else {
            return;
        };

        // This token prevents the phase from ending until we have set any potential error states.
        let Ok((tx, current_phase_wait_token)) = self.outgoing.recv().await else {
            return;
        };

        {
            *self.bridge_state.lock().unwrap() = BridgeState::Running
        }

        let (stop_reason, opt_wait_token) = self.run_impl(rx, tx, current_phase_wait_token).await;

        {
            *self.bridge_state.lock().unwrap() = BridgeState::Stopped(stop_reason)
        }

        // Only after we set the Mutex state do we drop the wait token (if we have one), causing the phase to end.
        drop(opt_wait_token);
    }

    async fn run_impl(
        &mut self,
        mut rx: FileReaderOutputRecv,
        mut tx: connector::Sender<Morsel>,
        mut current_phase_wait_token: WaitToken,
    ) -> (StopReason, Option<WaitToken>) {
        let mut n_morsels_processed: u64 = 0;

        loop {
            let mut morsel = match rx.recv().await {
                Ok(v) => v,
                Err(_) => {
                    drop(rx);

                    rx = match self.incoming.recv().await {
                        Ok(v) => v,
                        Err(_) => {
                            return (
                                StopReason::ReadersDisconnected,
                                Some(current_phase_wait_token),
                            );
                        },
                    };

                    continue;
                },
            };

            morsel.replace_source_token(self.source_token.clone());
            morsel.set_seq(MorselSeq::new(n_morsels_processed));

            n_morsels_processed = n_morsels_processed.saturating_add(1);

            while let Err(v) = tx.send(morsel).await {
                drop(tx);
                drop(current_phase_wait_token);

                (tx, current_phase_wait_token) = match self.outgoing.recv().await {
                    Ok(v) => v,
                    Err(_) => return (StopReason::ComputeNodeDisconnected, None),
                };

                morsel = v;
            }

            if self.source_token.stop_requested() {
                drop(tx);
                drop(current_phase_wait_token);

                (tx, current_phase_wait_token) = match self.outgoing.recv().await {
                    Ok(v) => v,
                    Err(_) => return (StopReason::ComputeNodeDisconnected, None),
                };

                self.source_token = SourceToken::new();
            }
        }
    }
}
