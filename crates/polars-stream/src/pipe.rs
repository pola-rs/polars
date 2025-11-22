use std::cmp::Reverse;
use std::sync::Arc;

use parking_lot::Mutex;
use polars_error::PolarsResult;
use polars_utils::priority::Priority;
use polars_utils::relaxed_cell::RelaxedCell;

use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::connector::{ReceiverExt, SenderExt, connector_with};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::Linearizer;
use crate::async_primitives::wait_group::WaitGroup;
use crate::graph::LogicalPipeKey;
use crate::metrics::GraphMetrics;
use crate::morsel::{Morsel, MorselSeq};
use crate::{DEFAULT_DISTRIBUTOR_BUFFER_SIZE, DEFAULT_LINEARIZER_BUFFER_SIZE};

pub fn port_channel(metrics: Option<Arc<PipeMetrics>>) -> (PortSender, PortReceiver) {
    let (send, recv) = connector_with(metrics);
    (PortSender(send), PortReceiver(recv))
}

pub struct PortSender(SenderExt<Morsel, Option<Arc<PipeMetrics>>>);
pub struct PortReceiver(ReceiverExt<Morsel, Option<Arc<PipeMetrics>>>);

impl PortSender {
    #[inline]
    pub async fn send(&mut self, morsel: Morsel) -> Result<(), Morsel> {
        let rows = morsel.df().height() as u64;
        self.0.send(morsel).await?;
        if let Some(metrics) = self.0.shared() {
            metrics.morsels_sent.fetch_add(1);
            metrics.rows_sent.fetch_add(rows);
            metrics.largest_morsel_sent.fetch_max(rows);
        }
        Ok(())
    }
}

impl PortReceiver {
    #[inline]
    pub async fn recv(&mut self) -> Result<Morsel, ()> {
        let morsel = self.0.recv().await?;
        let rows = morsel.df().height() as u64;
        if let Some(metrics) = self.0.shared() {
            metrics.morsels_received.fetch_add(1);
            metrics.rows_received.fetch_add(rows);
            metrics.largest_morsel_received.fetch_max(rows);
        }
        Ok(morsel)
    }
}

#[derive(Default)]
#[repr(align(128))]
pub struct PipeMetrics {
    pub morsels_sent: RelaxedCell<u64>,
    pub rows_sent: RelaxedCell<u64>,
    pub largest_morsel_sent: RelaxedCell<u64>,
    pub morsels_received: RelaxedCell<u64>,
    pub rows_received: RelaxedCell<u64>,
    pub largest_morsel_received: RelaxedCell<u64>,
}

pub struct PhysicalPipe {
    state: State,
    seq_offset: Arc<RelaxedCell<u64>>,
    metrics: Option<Arc<Mutex<GraphMetrics>>>,
    key: LogicalPipeKey,
}

impl PhysicalPipe {
    fn make_channel(&self) -> (PortSender, PortReceiver) {
        let metrics = self.metrics.as_ref().map(|m| {
            let pipe_metrics = Arc::<PipeMetrics>::default();
            m.lock().add_pipe(self.key, pipe_metrics.clone());
            pipe_metrics
        });
        port_channel(metrics)
    }
}

enum State {
    Invalid,
    Uninit {
        num_pipelines: usize,
    },
    SerialReceiver {
        num_pipelines: usize,
        send: PortSender,
        maintain_order: bool,
    },
    ParallelReceiver {
        senders: Vec<PortSender>,
    },
    NeedsLinearizer {
        receivers: Vec<PortReceiver>,
        send: PortSender,
        maintain_order: bool,
    },
    NeedsDistributor {
        recv: PortReceiver,
        senders: Vec<PortSender>,
    },
    NeedsOffset {
        senders: Vec<PortSender>,
        receivers: Vec<PortReceiver>,
    },
    Initialized,
}

pub struct SendPort<'a>(&'a mut PhysicalPipe);
pub struct RecvPort<'a>(&'a mut PhysicalPipe);

impl RecvPort<'_> {
    pub fn serial(self) -> PortReceiver {
        self.serial_with_maintain_order(true)
    }

    pub fn serial_with_maintain_order(self, maintain_order: bool) -> PortReceiver {
        let State::Uninit { num_pipelines } = self.0.state else {
            unreachable!()
        };
        let (send, recv) = self.0.make_channel();
        self.0.state = State::SerialReceiver {
            num_pipelines,
            send,
            maintain_order,
        };
        recv
    }

    pub fn parallel(self) -> Vec<PortReceiver> {
        let State::Uninit { num_pipelines } = self.0.state else {
            unreachable!()
        };
        let (senders, receivers): (Vec<PortSender>, Vec<PortReceiver>) =
            (0..num_pipelines).map(|_| self.0.make_channel()).unzip();
        self.0.state = State::ParallelReceiver { senders };
        receivers
    }
}

impl SendPort<'_> {
    #[allow(unused)]
    pub fn is_receiver_serial(&self) -> bool {
        matches!(self.0.state, State::SerialReceiver { .. })
    }

    pub fn serial(self) -> PortSender {
        match core::mem::replace(&mut self.0.state, State::Invalid) {
            State::SerialReceiver { send, .. } => {
                if self.0.seq_offset.load() == 0 {
                    self.0.state = State::Initialized;
                    send
                } else {
                    let (offset_send, offset_recv) = self.0.make_channel();
                    self.0.state = State::NeedsOffset {
                        senders: vec![send],
                        receivers: vec![offset_recv],
                    };
                    offset_send
                }
            },
            State::ParallelReceiver { senders } => {
                let (send, recv) = self.0.make_channel();
                self.0.state = State::NeedsDistributor { recv, senders };
                send
            },
            _ => unreachable!(),
        }
    }

    pub fn parallel(self) -> Vec<PortSender> {
        match core::mem::replace(&mut self.0.state, State::Invalid) {
            State::SerialReceiver {
                num_pipelines,
                send,
                maintain_order,
            } => {
                let (senders, receivers): (Vec<PortSender>, Vec<PortReceiver>) =
                    (0..num_pipelines).map(|_| self.0.make_channel()).unzip();
                self.0.state = State::NeedsLinearizer {
                    receivers,
                    send,
                    maintain_order,
                };
                senders
            },
            State::ParallelReceiver { senders } => {
                if self.0.seq_offset.load() == 0 {
                    self.0.state = State::Initialized;
                    senders
                } else {
                    let (offset_senders, offset_receivers): (Vec<PortSender>, Vec<PortReceiver>) =
                        senders.iter().map(|_| self.0.make_channel()).unzip();
                    self.0.state = State::NeedsOffset {
                        senders,
                        receivers: offset_receivers,
                    };
                    offset_senders
                }
            },
            _ => unreachable!(),
        }
    }
}

impl PhysicalPipe {
    pub fn new(
        num_pipelines: usize,
        key: LogicalPipeKey,
        seq_offset: Arc<RelaxedCell<u64>>,
        metrics: Option<Arc<Mutex<GraphMetrics>>>,
    ) -> Self {
        Self {
            state: State::Uninit { num_pipelines },
            key,
            seq_offset,
            metrics,
        }
    }

    pub fn recv_port(&mut self) -> RecvPort<'_> {
        assert!(
            matches!(self.state, State::Uninit { .. }),
            "PhysicalPipe::recv_port can only be called on an uninitialized pipe"
        );
        RecvPort(self)
    }

    pub fn send_port(&mut self) -> SendPort<'_> {
        assert!(
            matches!(
                self.state,
                State::SerialReceiver { .. } | State::ParallelReceiver { .. }
            ),
            "PhysicalPipe::send_port must be called on a pipe which only has its receive port initialized"
        );
        SendPort(self)
    }

    pub fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        match core::mem::replace(&mut self.state, State::Initialized) {
            State::Invalid
            | State::Uninit { .. }
            | State::SerialReceiver { .. }
            | State::ParallelReceiver { .. } => {
                panic!("PhysicalPipe::spawn called on (partially) initialized pipe");
            },

            State::Initialized => {},

            State::NeedsLinearizer {
                receivers,
                mut send,
                maintain_order,
            } => {
                let num_pipelines = receivers.len();
                let (mut linearizer, inserters) =
                    Linearizer::<Priority<Reverse<MorselSeq>, Morsel>>::new_with_maintain_order(
                        num_pipelines,
                        *DEFAULT_LINEARIZER_BUFFER_SIZE,
                        maintain_order,
                    );

                let seq_offset = self.seq_offset.load();
                handles.push(scope.spawn_task(TaskPriority::High, async move {
                    while let Some(Priority(_, mut morsel)) = linearizer.get().await {
                        morsel.set_seq(morsel.seq().offset_by_u64(seq_offset));
                        if send.send(morsel).await.is_err() {
                            break;
                        }
                    }

                    Ok(())
                }));

                for (mut recv, mut inserter) in receivers.into_iter().zip(inserters) {
                    handles.push(scope.spawn_task(TaskPriority::High, async move {
                        while let Ok(mut morsel) = recv.recv().await {
                            // Drop the consume token, but only after the send has succeeded. This
                            // ensures we have backpressure, but only once the channel fills up.
                            let consume_token = morsel.take_consume_token();
                            if inserter
                                .insert(Priority(Reverse(morsel.seq()), morsel))
                                .await
                                .is_err()
                            {
                                break;
                            }
                            drop(consume_token);
                        }

                        Ok(())
                    }));
                }
            },

            State::NeedsDistributor { mut recv, senders } => {
                let num_pipelines = senders.len();
                let (mut distributor, distr_receivers) =
                    distributor_channel(num_pipelines, *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

                let arc_seq_offset = self.seq_offset.clone();
                handles.push(scope.spawn_task(TaskPriority::High, async move {
                    let mut seq_offset = arc_seq_offset.load();
                    let mut prev_orig_seq = None;

                    while let Ok(mut morsel) = recv.recv().await {
                        // We have to relabel sequence ids to be unique before distributing.
                        // Normally within a single pipeline consecutive ids may repeat but
                        // when distributing this would destroy the order.
                        if Some(morsel.seq()) == prev_orig_seq {
                            seq_offset += 1;
                        }
                        prev_orig_seq = Some(morsel.seq());
                        morsel.set_seq(morsel.seq().offset_by_u64(seq_offset));

                        // Important: we have to drop the consume token before
                        // going into the buffered distributor.
                        drop(morsel.take_consume_token());
                        if distributor.send(morsel).await.is_err() {
                            break;
                        }
                    }

                    arc_seq_offset.store(seq_offset);

                    Ok(())
                }));

                for (mut send, mut recv) in senders.into_iter().zip(distr_receivers) {
                    handles.push(scope.spawn_task(TaskPriority::High, async move {
                        let wait_group = WaitGroup::default();
                        while let Ok(mut morsel) = recv.recv().await {
                            morsel.set_consume_token(wait_group.token());
                            if send.send(morsel).await.is_err() {
                                break;
                            }
                            wait_group.wait().await;
                        }

                        Ok(())
                    }));
                }
            },

            State::NeedsOffset { senders, receivers } => {
                let seq_offset = self.seq_offset.load();
                for (mut send, mut recv) in senders.into_iter().zip(receivers) {
                    handles.push(scope.spawn_task(TaskPriority::High, async move {
                        while let Ok(mut morsel) = recv.recv().await {
                            morsel.set_seq(morsel.seq().offset_by_u64(seq_offset));
                            if send.send(morsel).await.is_err() {
                                break;
                            }
                        }
                        Ok(())
                    }));
                }
            },
        }
    }
}
