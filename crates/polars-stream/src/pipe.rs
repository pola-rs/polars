use std::cmp::Reverse;

use polars_error::PolarsResult;
use polars_utils::priority::Priority;

use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::connector::{Receiver, Sender, connector};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::Linearizer;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::{Morsel, MorselSeq};
use crate::{DEFAULT_DISTRIBUTOR_BUFFER_SIZE, DEFAULT_LINEARIZER_BUFFER_SIZE};

pub enum PhysicalPipe {
    Uninit(usize),
    /// (_, _, maintain_order)
    SerialReceiver(usize, Sender<Morsel>, bool),
    ParallelReceiver(Vec<Sender<Morsel>>),
    /// (_, _, maintain_order)
    NeedsLinearizer(Vec<Receiver<Morsel>>, Sender<Morsel>, bool),
    NeedsDistributor(Receiver<Morsel>, Vec<Sender<Morsel>>),
    Initialized,
}

pub struct SendPort<'a>(&'a mut PhysicalPipe);
pub struct RecvPort<'a>(&'a mut PhysicalPipe);

impl RecvPort<'_> {
    pub fn serial(self) -> Receiver<Morsel> {
        self.serial_with_maintain_order(true)
    }

    pub fn serial_with_maintain_order(self, maintain_order: bool) -> Receiver<Morsel> {
        let PhysicalPipe::Uninit(num_pipelines) = self.0 else {
            unreachable!()
        };
        let (send, recv) = connector();
        *self.0 = PhysicalPipe::SerialReceiver(*num_pipelines, send, maintain_order);
        recv
    }

    pub fn parallel(self) -> Vec<Receiver<Morsel>> {
        let PhysicalPipe::Uninit(num_pipelines) = self.0 else {
            unreachable!()
        };
        let (senders, receivers): (Vec<Sender<Morsel>>, Vec<Receiver<Morsel>>) =
            (0..*num_pipelines).map(|_| connector()).unzip();
        *self.0 = PhysicalPipe::ParallelReceiver(senders);
        receivers
    }
}

impl SendPort<'_> {
    #[allow(unused)]
    pub fn is_receiver_serial(&self) -> bool {
        matches!(self.0, PhysicalPipe::SerialReceiver(..))
    }

    pub fn serial(self) -> Sender<Morsel> {
        match core::mem::replace(self.0, PhysicalPipe::Uninit(0)) {
            PhysicalPipe::SerialReceiver(_, send, _) => {
                *self.0 = PhysicalPipe::Initialized;
                send
            },
            PhysicalPipe::ParallelReceiver(senders) => {
                let (send, recv) = connector();
                *self.0 = PhysicalPipe::NeedsDistributor(recv, senders);
                send
            },
            _ => unreachable!(),
        }
    }

    pub fn parallel(self) -> Vec<Sender<Morsel>> {
        match core::mem::replace(self.0, PhysicalPipe::Uninit(0)) {
            PhysicalPipe::SerialReceiver(num_pipelines, send, maintain_order) => {
                let (senders, receivers): (Vec<Sender<Morsel>>, Vec<Receiver<Morsel>>) =
                    (0..num_pipelines).map(|_| connector()).unzip();
                *self.0 = PhysicalPipe::NeedsLinearizer(receivers, send, maintain_order);
                senders
            },
            PhysicalPipe::ParallelReceiver(senders) => {
                *self.0 = PhysicalPipe::Initialized;
                senders
            },
            _ => unreachable!(),
        }
    }
}

impl PhysicalPipe {
    pub fn new(num_pipelines: usize) -> Self {
        Self::Uninit(num_pipelines)
    }

    pub fn recv_port(&mut self) -> RecvPort<'_> {
        assert!(
            matches!(self, Self::Uninit(_)),
            "PhysicalPipe::recv_port can only be called on an uninitialized pipe"
        );
        RecvPort(self)
    }

    pub fn send_port(&mut self) -> SendPort<'_> {
        assert!(
            matches!(self, Self::SerialReceiver(..) | Self::ParallelReceiver(..)),
            "PhysicalPipe::send_port must be called on a pipe which only has its receive port initialized"
        );
        SendPort(self)
    }

    pub fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        match core::mem::replace(self, Self::Initialized) {
            Self::Uninit(_) | Self::SerialReceiver(_, _, _) | Self::ParallelReceiver(_) => {
                panic!("PhysicalPipe::spawn called on (partially) initialized pipe");
            },

            Self::Initialized => {},

            Self::NeedsLinearizer(receivers, mut sender, maintain_order) => {
                let num_pipelines = receivers.len();
                let (mut linearizer, inserters) =
                    Linearizer::<Priority<Reverse<MorselSeq>, Morsel>>::new_with_maintain_order(
                        num_pipelines,
                        *DEFAULT_LINEARIZER_BUFFER_SIZE,
                        maintain_order,
                    );

                handles.push(scope.spawn_task(TaskPriority::High, async move {
                    while let Some(morsel) = linearizer.get().await {
                        if sender.send(morsel.1).await.is_err() {
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

            Self::NeedsDistributor(mut receiver, senders) => {
                let num_pipelines = senders.len();
                let (mut distributor, distr_receivers) =
                    distributor_channel(num_pipelines, *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

                handles.push(scope.spawn_task(TaskPriority::High, async move {
                    while let Ok(mut morsel) = receiver.recv().await {
                        // Important: we have to drop the consume token before
                        // going into the buffered distributor.
                        drop(morsel.take_consume_token());
                        if distributor.send(morsel).await.is_err() {
                            break;
                        }
                    }

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
        }
    }
}
