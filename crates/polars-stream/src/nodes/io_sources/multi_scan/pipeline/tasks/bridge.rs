use std::sync::{Arc, Mutex};

use crate::async_executor;
use crate::async_executor::{JoinHandle, TaskPriority};
use crate::async_primitives::connector;
use crate::async_primitives::wait_group::WaitToken;
use crate::morsel::{MorselSeq, SourceToken};
use crate::nodes::io_sources::multi_scan::components::bridge::{
    BridgeRecvPort, BridgeState, StopReason,
};
use crate::pipe::PortSender;

pub fn spawn_bridge(
    bridge_state: Arc<Mutex<BridgeState>>,
) -> (
    JoinHandle<()>,
    // For attaching file reader output port
    connector::Sender<BridgeRecvPort>,
    // For attaching the multi scan node output port
    connector::Sender<(PortSender, WaitToken)>,
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
    outgoing: connector::Receiver<(PortSender, WaitToken)>,
    bridge_state: Arc<Mutex<BridgeState>>,
    source_token: SourceToken,
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
        mut tx: PortSender,
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
