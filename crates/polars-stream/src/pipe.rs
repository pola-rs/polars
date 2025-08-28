use std::cmp::Reverse;
use std::sync::Arc;

use polars_error::PolarsResult;
use polars_utils::priority::Priority;
use polars_utils::relaxed_cell::RelaxedCell;

use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::connector::{Receiver, Sender, connector};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::Linearizer;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::{Morsel, MorselSeq};
use crate::{DEFAULT_DISTRIBUTOR_BUFFER_SIZE, DEFAULT_LINEARIZER_BUFFER_SIZE};

pub struct PhysicalPipe {
    state: State,
    seq_offset: Arc<RelaxedCell<u64>>,
}

enum State {
    Invalid,
    Uninit {
        num_pipelines: usize,
    },
    SerialReceiver {
        num_pipelines: usize,
        send: Sender<Morsel>,
        maintain_order: bool,
    },
    ParallelReceiver {
        senders: Vec<Sender<Morsel>>,
    },
    NeedsLinearizer {
        receivers: Vec<Receiver<Morsel>>,
        send: Sender<Morsel>,
        maintain_order: bool,
    },
    NeedsDistributor {
        recv: Receiver<Morsel>,
        senders: Vec<Sender<Morsel>>,
    },
    NeedsOffset {
        senders: Vec<Sender<Morsel>>,
        receivers: Vec<Receiver<Morsel>>,
    },
    Initialized,
}

pub struct SendPort<'a>(&'a mut PhysicalPipe);
pub struct RecvPort<'a>(&'a mut PhysicalPipe);

impl RecvPort<'_> {
    pub fn serial(self) -> Receiver<Morsel> {
        self.serial_with_maintain_order(true)
    }

    pub fn serial_with_maintain_order(self, maintain_order: bool) -> Receiver<Morsel> {
        let State::Uninit { num_pipelines } = self.0.state else {
            unreachable!()
        };
        let (send, recv) = connector();
        self.0.state = State::SerialReceiver {
            num_pipelines,
            send,
            maintain_order,
        };
        recv
    }

    pub fn parallel(self) -> Vec<Receiver<Morsel>> {
        let State::Uninit { num_pipelines } = self.0.state else {
            unreachable!()
        };
        let (senders, receivers): (Vec<Sender<Morsel>>, Vec<Receiver<Morsel>>) =
            (0..num_pipelines).map(|_| connector()).unzip();
        self.0.state = State::ParallelReceiver { senders };
        receivers
    }
}

impl SendPort<'_> {
    #[allow(unused)]
    pub fn is_receiver_serial(&self) -> bool {
        matches!(self.0.state, State::SerialReceiver { .. })
    }

    pub fn serial(self) -> Sender<Morsel> {
        match core::mem::replace(&mut self.0.state, State::Invalid) {
            State::SerialReceiver { send, .. } => {
                if self.0.seq_offset.load() == 0 {
                    self.0.state = State::Initialized;
                    send
                } else {
                    let (offset_send, offset_recv) = connector();
                    self.0.state = State::NeedsOffset {
                        senders: vec![send],
                        receivers: vec![offset_recv],
                    };
                    offset_send
                }
            },
            State::ParallelReceiver { senders } => {
                let (send, recv) = connector();
                self.0.state = State::NeedsDistributor { recv, senders };
                send
            },
            _ => unreachable!(),
        }
    }

    pub fn parallel(self) -> Vec<Sender<Morsel>> {
        match core::mem::replace(&mut self.0.state, State::Invalid) {
            State::SerialReceiver {
                num_pipelines,
                send,
                maintain_order,
            } => {
                let (senders, receivers): (Vec<Sender<Morsel>>, Vec<Receiver<Morsel>>) =
                    (0..num_pipelines).map(|_| connector()).unzip();
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
                    let (offset_senders, offset_receivers): (
                        Vec<Sender<Morsel>>,
                        Vec<Receiver<Morsel>>,
                    ) = senders.iter().map(|_| connector()).unzip();
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
    pub fn new(num_pipelines: usize, seq_offset: Arc<RelaxedCell<u64>>) -> Self {
        Self {
            state: State::Uninit { num_pipelines },
            seq_offset,
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
