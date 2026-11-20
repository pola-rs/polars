use std::sync::{Arc, Mutex};

use polars_async::executor;
use polars_async::executor::{JoinHandle, TaskMetricAggregator, TaskPriority};
use polars_async::primitives::connector;
use polars_async::primitives::wait_group::WaitToken;

use crate::morsel::{MorselSeq, SourceToken};
use crate::nodes::io_sources::multi_scan::components::bridge::{
    BridgeRecvPort, BridgeState, StopReason,
};
use crate::pipe::PortSender;

pub fn spawn_bridge(
    bridge_state: Arc<Mutex<BridgeState>>,
    task_metrics: Option<&TaskMetricAggregator>,
) -> (
    JoinHandle<()>,
    // For attaching file reader output port
    connector::Sender<BridgeRecvPort>,
    // For attaching the multi scan node output port
    connector::Sender<(PortSender, WaitToken)>,
) {
    let (incoming_tx, incoming) = connector::connector();
    let (outgoing_tx, outgoing) = connector::connector();

    let handle = executor::spawn(
        TaskPriority::Low,
        task_metrics,
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
        tx: PortSender,
        current_phase_wait_token: WaitToken,
    ) -> (StopReason, Option<WaitToken>) {
        // The current phase's port and wait token.
        let mut phase = Some((tx, current_phase_wait_token));
        let mut morsel_seq: u64 = 0;

        loop {
            let recv = recv_or_stop(
                rx.recv(),
                &mut self.outgoing,
                &mut self.source_token,
                &mut phase,
            )
            .await;
            let mut morsel = match recv {
                Ok(Ok(v)) => v,
                Ok(Err(_)) => {
                    drop(rx);

                    let recv = recv_or_stop(
                        self.incoming.recv(),
                        &mut self.outgoing,
                        &mut self.source_token,
                        &mut phase,
                    )
                    .await;
                    rx = match recv {
                        Ok(Ok(v)) => v,
                        Ok(Err(_)) => {
                            return (
                                StopReason::ReadersDisconnected,
                                phase.map(|(_, wait_token)| wait_token),
                            );
                        },
                        Err(reason) => return (reason, None),
                    };

                    continue;
                },
                Err(reason) => return (reason, None),
            };

            morsel.replace_source_token(self.source_token.clone());
            // Important: We override the sequence ID here, as when we receive from a new reader the
            // incoming ID begins again from 0.
            morsel.set_seq(MorselSeq::new(morsel_seq));

            morsel_seq = morsel_seq.saturating_add(1);

            while let Err(v) = phase.as_mut().unwrap().0.send(morsel).await {
                if let Err(reason) =
                    next_phase(&mut self.outgoing, &mut self.source_token, &mut phase).await
                {
                    return (reason, None);
                }

                morsel = v;
                // the new phase has a new token: a stop on this morsel must reach it
                morsel.replace_source_token(self.source_token.clone());
            }

            if self.source_token.stop_requested() {
                if let Err(reason) =
                    next_phase(&mut self.outgoing, &mut self.source_token, &mut phase).await
                {
                    return (reason, None);
                }
            }
        }
    }
}

/// Waits for `recv`. A stop requested meanwhile ends the phase at once, instead of after the next
/// morsel: a reader can wait long for its first rows (e.g. a slow database partition) while the
/// output already has all it needs. `recv` is not cancelled, but carries over into the next phase.
async fn recv_or_stop<T>(
    recv: impl Future<Output = T>,
    outgoing: &mut connector::Receiver<(PortSender, WaitToken)>,
    source_token: &mut SourceToken,
    phase: &mut Option<(PortSender, WaitToken)>,
) -> Result<T, StopReason> {
    let mut recv = std::pin::pin!(recv);
    loop {
        tokio::select! {
            biased;
            v = &mut recv => return Ok(v),
            _ = source_token.stop_requested_wait() => {},
        }
        next_phase(outgoing, source_token, phase).await?;
    }
}

/// Ends the current phase and waits for the next one.
async fn next_phase(
    outgoing: &mut connector::Receiver<(PortSender, WaitToken)>,
    source_token: &mut SourceToken,
    phase: &mut Option<(PortSender, WaitToken)>,
) -> Result<(), StopReason> {
    drop(phase.take());
    *phase = Some(
        outgoing
            .recv()
            .await
            .map_err(|_| StopReason::ComputeNodeDisconnected)?,
    );
    *source_token = SourceToken::new();
    Ok(())
}
