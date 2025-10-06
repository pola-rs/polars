use polars_error::polars_ensure;

use super::compute_node_prelude::*;
use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::wait_group::WaitGroup;

pub struct GatherEveryNode {
    n: usize,
    offset: usize,
}

impl GatherEveryNode {
    pub fn new(n: usize, offset: usize) -> PolarsResult<Self> {
        polars_ensure!(n > 0, InvalidOperation: "gather_every(n): n should be positive");

        assert!(i64::try_from(n).unwrap() > 0);
        assert!(i64::try_from(offset).unwrap() >= 0);

        Ok(Self { n, offset })
    }
}

impl ComputeNode for GatherEveryNode {
    fn name(&self) -> &str {
        "gather_every"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);
        recv.swap_with_slice(send);
        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);
        let mut receiver = recv_ports[0].take().unwrap().serial();
        let senders = send_ports[0].take().unwrap().parallel();

        let (mut distributor, distr_receivers) =
            distributor_channel(senders.len(), *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

        let n = self.n;

        // To figure out the correct offsets we need to be serial.
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = receiver.recv().await {
                let height = morsel.df().height();
                if self.offset >= height {
                    self.offset -= height;
                    continue;
                }

                if distributor.send((morsel, self.offset)).await.is_err() {
                    break;
                }

                // Calculates `offset = (offset - height) mod n` without under- and overflow.
                self.offset += height.next_multiple_of(self.n) - height;
                self.offset %= self.n;
            }

            Ok(())
        }));

        // But gathering the column can be done in parallel.
        for (mut send, mut recv) in senders.into_iter().zip(distr_receivers) {
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let wait_group = WaitGroup::default();
                while let Ok((morsel, offset)) = recv.recv().await {
                    let mut morsel = morsel.try_map(|mut df| {
                        let column = &df.get_columns()[0];
                        let out = column
                            .gather_every(n, offset)?
                            .with_name(column.name().clone());
                        unsafe { df.get_columns_mut()[0] = out };
                        PolarsResult::Ok(df)
                    })?;
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
