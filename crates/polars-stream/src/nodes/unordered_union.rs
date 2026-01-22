use std::sync::Arc;

use polars_core::schema::Schema;
use tokio::sync::mpsc;

use super::compute_node_prelude::*;

pub struct UnorderedUnionNode {
    max_morsel_seq_sent: MorselSeq,
    output_schema: Arc<Schema>,
}

impl UnorderedUnionNode {
    pub fn new(output_schema: Arc<Schema>) -> Self {
        Self {
            max_morsel_seq_sent: MorselSeq::new(0),
            output_schema,
        }
    }
}

impl ComputeNode for UnorderedUnionNode {
    fn name(&self) -> &str {
        "unordered-union"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert_eq!(send.len(), 1);

        let all_done = recv.iter().all(|r| *r == PortState::Done);

        if all_done {
            send[0] = PortState::Done;
        } else {
            let any_ready = recv.iter().any(|r| *r == PortState::Ready);
            send[0] = if any_ready {
                PortState::Ready
            } else {
                PortState::Blocked
            };
        }

        recv.fill(send[0]);

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert_eq!(send_ports.len(), 1);

        let output_schema = self.output_schema.clone();

        let output_senders = send_ports[0].take().unwrap().parallel();
        let n = output_senders.len();
        assert_eq!(n, state.num_pipelines);

        let (mpsc_senders, mpsc_receivers): (Vec<_>, Vec<_>) =
            (0..n).map(|_| mpsc::channel::<Morsel>(1)).unzip();

        for recv_port in recv_ports {
            if let Some(recv) = recv_port.take() {
                let receivers = recv.parallel();
                let mpsc_senders_clone = mpsc_senders.clone();

                for (mut receiver, sender) in receivers.into_iter().zip(mpsc_senders_clone) {
                    join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                        while let Ok(morsel) = receiver.recv().await {
                            if sender.send(morsel).await.is_err() {
                                break;
                            }
                        }
                        PolarsResult::Ok(())
                    }));
                }
            }
        }

        drop(mpsc_senders);

        let morsel_offset = self.max_morsel_seq_sent.successor();

        let mut inner_handles = Vec::new();
        for (lane_idx, (mut mpsc_receiver, mut output_sender)) in
            mpsc_receivers.into_iter().zip(output_senders).enumerate()
        {
            inner_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let mut local_seq = morsel_offset.offset_by_u64(lane_idx as u64);
                let seq_step = n as u64;
                let mut max_seq = MorselSeq::new(0);

                while let Some(mut morsel) = mpsc_receiver.recv().await {
                    morsel.set_seq(local_seq);
                    max_seq = max_seq.max(local_seq);
                    local_seq = local_seq.offset_by_u64(seq_step);

                    if output_sender.send(morsel).await.is_err() {
                        break;
                    }
                }

                PolarsResult::Ok(max_seq)
            }));
        }

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            for handle in inner_handles {
                self.max_morsel_seq_sent = self.max_morsel_seq_sent.max(handle.await?);
            }
            Ok(())
        }));
    }
}
