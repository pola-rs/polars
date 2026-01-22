use std::sync::Arc;

use polars_core::schema::Schema;
use tokio::sync::mpsc;

use super::compute_node_prelude::*;

pub struct UnorderedUnionNode {
    output_schema: Arc<Schema>,
}

impl UnorderedUnionNode {
    pub fn new(output_schema: Arc<Schema>) -> Self {
        Self { output_schema }
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

        if send[0] == PortState::Blocked {
            for r in recv.iter_mut() {
                if *r == PortState::Ready {
                    *r = PortState::Blocked;
                }
            }
        }

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
        assert_eq!(send_ports.len(), 1);

        let output_schema = self.output_schema.clone();

        let output_senders = send_ports[0].take().unwrap().parallel();
        let n = output_senders.len();
        assert_eq!(n, _state.num_pipelines);

        let mut mpsc_senders = Vec::new();
        let mut mpsc_receivers = Vec::new();

        for _ in 0..n {
            let (tx, rx) = mpsc::channel::<Morsel>(1000); // what should be? 1000 is just random
            mpsc_senders.push(tx);
            mpsc_receivers.push(rx);
        }

        for recv_port in recv_ports {
            let receivers = recv_port.take().unwrap().parallel();
            let output_schema = output_schema.clone();

            let mpsc_senders_clone: Vec<_> = mpsc_senders.iter().map(|s| s.clone()).collect();

            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let mut sub_tasks = Vec::new();

                for (lane_idx, mut receiver) in receivers.into_iter().enumerate() {
                    let mpsc_senders = mpsc_senders_clone.clone();
                    let output_schema = output_schema.clone();

                    sub_tasks.push(scope.spawn_task(TaskPriority::High, async move {
                        while let Ok(mut morsel) = receiver.recv().await {
                            morsel.df_mut().ensure_matches_schema(&output_schema)?;

                            if mpsc_senders[lane_idx].send(morsel).await.is_err() {
                                break;
                            }
                        }
                        PolarsResult::Ok(())
                    }));
                }

                for sub_task in sub_tasks {
                    sub_task.await?;
                }

                PolarsResult::Ok(())
            }));
        }

        drop(mpsc_senders);

        for (lane_idx, (mut mpsc_receiver, mut output_sender)) in
            mpsc_receivers.into_iter().zip(output_senders).enumerate()
        {
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let mut local_seq = lane_idx as u64;
                let seq_step = n as u64;

                while let Some(mut morsel) = mpsc_receiver.recv().await {
                    morsel.set_seq(MorselSeq::new(local_seq));
                    local_seq += seq_step;

                    if output_sender.send(morsel).await.is_err() {
                        break;
                    }
                }

                PolarsResult::Ok(())
            }));
        }
    }
}
