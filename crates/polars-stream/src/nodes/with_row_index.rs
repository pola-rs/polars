use polars_core::prelude::*;
use polars_core::utils::Container;
use polars_utils::pl_str::PlSmallStr;

use super::compute_node_prelude::*;
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::wait_group::WaitGroup;
use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;

pub struct WithRowIndexNode {
    name: PlSmallStr,
    offset: IdxSize,
}

impl WithRowIndexNode {
    pub fn new(name: PlSmallStr, offset: Option<IdxSize>) -> Self {
        Self {
            name,
            offset: offset.unwrap_or(0),
        }
    }
}

impl ComputeNode for WithRowIndexNode {
    fn name(&self) -> &str {
        "with_row_index"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);
        recv.swap_with_slice(send);
        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv: &mut [Option<RecvPort<'_>>],
        send: &mut [Option<SendPort<'_>>],
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv.len() == 1 && send.len() == 1);
        let mut receiver = recv[0].take().unwrap().serial();
        let senders = send[0].take().unwrap().parallel();

        let (mut distributor, distr_receivers) =
            distributor_channel(senders.len(), DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

        let name = self.name.clone();

        // To figure out the correct offsets we need to be serial.
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = receiver.recv().await {
                let offset = self.offset;
                self.offset = self
                    .offset
                    .checked_add(morsel.df().len().try_into().unwrap())
                    .unwrap();
                if distributor.send((morsel, offset)).await.is_err() {
                    break;
                }
            }

            Ok(())
        }));

        // But adding the new row index column can be done in parallel.
        for (mut send, mut recv) in senders.into_iter().zip(distr_receivers) {
            let name = name.clone();
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let wait_group = WaitGroup::default();
                while let Ok((morsel, offset)) = recv.recv().await {
                    let mut morsel =
                        morsel.try_map(|df| df.with_row_index(name.clone(), Some(offset)))?;
                    morsel.set_consume_token(wait_group.token());
                    if send.send(morsel).await.is_err() {
                        break;
                    }
                    wait_group.wait().await;
                }

                Ok(())
            }));
        }
    }
}
