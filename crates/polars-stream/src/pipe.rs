use polars_error::PolarsResult;

use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::connector::{connector, Receiver, Sender};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::Morsel;
use crate::utils::linearizer::Linearizer;
use crate::{DEFAULT_DISTRIBUTOR_BUFFER_SIZE, DEFAULT_LINEARIZER_BUFFER_SIZE};

pub enum PhysicalPipe {
    Uninit(usize),
    SerialReceiver(usize, Sender<Morsel>),
    ParallelReceiver(Vec<Sender<Morsel>>),
    NeedsLinearizer(Vec<Receiver<Morsel>>, Sender<Morsel>),
    NeedsDistributor(Receiver<Morsel>, Vec<Sender<Morsel>>),
    Initialized,
}

pub struct SendPort<'a>(&'a mut PhysicalPipe);
pub struct RecvPort<'a>(&'a mut PhysicalPipe);

impl<'a> RecvPort<'a> {
    pub fn serial(self) -> Receiver<Morsel> {
        let PhysicalPipe::Uninit(num_pipelines) = self.0 else {
            unreachable!()
        };
        let (send, recv) = connector();
        *self.0 = PhysicalPipe::SerialReceiver(*num_pipelines, send);
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

impl<'a> SendPort<'a> {
    #[allow(unused)]
    pub fn is_receiver_serial(&self) -> bool {
        matches!(self.0, PhysicalPipe::SerialReceiver(..))
    }

    pub fn serial(self) -> Sender<Morsel> {
        match core::mem::replace(self.0, PhysicalPipe::Uninit(0)) {
            PhysicalPipe::SerialReceiver(_, send) => {
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
            PhysicalPipe::SerialReceiver(num_pipelines, send) => {
                let (senders, receivers): (Vec<Sender<Morsel>>, Vec<Receiver<Morsel>>) =
                    (0..num_pipelines).map(|_| connector()).unzip();
                *self.0 = PhysicalPipe::NeedsLinearizer(receivers, send);
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
            "PhysicalPipe::send_port must be called on a pipe which only has its send port initialized"
        );
        SendPort(self)
    }

    pub fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        match core::mem::replace(self, Self::Initialized) {
            Self::Uninit(_) | Self::SerialReceiver(_, _) | Self::ParallelReceiver(_) => {
                panic!("PhysicalPipe::spawn called on (partially) initialized pipe");
            },

            Self::Initialized => {},

            Self::NeedsLinearizer(receivers, mut sender) => {
                let num_pipelines = receivers.len();
                let (mut linearizer, inserters) =
                    Linearizer::new(num_pipelines, DEFAULT_LINEARIZER_BUFFER_SIZE);

                handles.push(scope.spawn_task(TaskPriority::High, async move {
                    while let Some(morsel) = linearizer.get().await {
                        if sender.send(morsel).await.is_err() {
                            break;
                        }
                    }

                    Ok(())
                }));

                for (mut recv, mut inserter) in receivers.into_iter().zip(inserters) {
                    handles.push(scope.spawn_task(TaskPriority::High, async move {
                        while let Ok(morsel) = recv.recv().await {
                            if inserter.insert(morsel).await.is_err() {
                                break;
                            }
                        }

                        Ok(())
                    }));
                }
            },

            Self::NeedsDistributor(mut receiver, senders) => {
                let num_pipelines = senders.len();
                let (mut distributor, distr_receivers) =
                    distributor_channel(num_pipelines, DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

                handles.push(scope.spawn_task(TaskPriority::High, async move {
                    while let Ok(morsel) = receiver.recv().await {
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
