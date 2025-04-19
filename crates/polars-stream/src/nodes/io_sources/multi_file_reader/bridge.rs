use std::sync::{Arc, Mutex};

use super::reader_interface::output::FileReaderOutputRecv;
use crate::async_executor::{self, JoinHandle, TaskPriority};
use crate::async_primitives::connector;
use crate::async_primitives::morsel_linearizer::MorselLinearizer;
use crate::async_primitives::wait_group::WaitToken;
use crate::morsel::{Morsel, MorselSeq, SourceToken};

#[expect(clippy::type_complexity)]
pub fn spawn_bridge(
    bridge_state: Arc<Mutex<BridgeState>>,
) -> (
    JoinHandle<()>,
    // For attaching file reader output port
    connector::Sender<BridgeRecvPort>,
    // For attaching the multi scan node output port
    connector::Sender<(connector::Sender<Morsel>, WaitToken)>,
) {
    let (incoming_tx, incoming) = connector::connector();
    let (outgoing_tx, outgoing) = connector::connector();

    let handle = async_executor::spawn(
        TaskPriority::Low,
        Bridge {
            incoming,
            outgoing,
            bridge_state,
            source_token: SourceToken::new(),
        }
        .run(),
    );

    (handle, incoming_tx, outgoing_tx)
}

/// Bridge that connects the reader pipeline to the compute node output. Handles
/// switching on both sides (i.e. file changes on incoming, phase changes on outgoing).
struct Bridge {
    incoming: connector::Receiver<BridgeRecvPort>,
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
    /// Disconnected from the reader side. The reader pipeline handle should be joined on in this case to
    /// determine if the readers disconnected due to an error.
    ReadersDisconnected,
    /// Disconnected from the multi scan ComputeNode.
    ComputeNodeDisconnected,
}

/// Port for the reader side.
///
/// Note: `first_morsel` is a residual from post-op initialization.
pub enum BridgeRecvPort {
    Direct {
        rx: FileReaderOutputRecv,
        first_morsel: Option<Morsel>,
    },
    /// Parallel post-apply ops will connect through this.
    Linearized { rx: MorselLinearizer },
}

impl BridgeRecvPort {
    async fn recv(&mut self) -> Result<Morsel, ()> {
        use BridgeRecvPort::*;
        match self {
            Direct { rx, first_morsel } => {
                if let Some(v) = first_morsel.take() {
                    Ok(v)
                } else {
                    rx.recv().await
                }
            },
            Linearized { rx } => rx.get().await.ok_or(()),
        }
    }
}

impl Bridge {
    async fn run(mut self) {
        let Ok(rx) = self.incoming.recv().await else {
            return;
        };

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

        // Only end the phase after we have set the mutex state.
        drop(opt_wait_token);
    }

    async fn run_impl(
        &mut self,
        mut rx: BridgeRecvPort,
        mut tx: connector::Sender<Morsel>,
        mut current_phase_wait_token: WaitToken,
    ) -> (StopReason, Option<WaitToken>) {
        let mut morsel_seq: u64 = 0;

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
            // Important: We override the sequence ID here, as when we receive from a new reader the
            // incoming ID begins again from 0.
            morsel.set_seq(MorselSeq::new(morsel_seq));

            morsel_seq = morsel_seq.saturating_add(1);

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
