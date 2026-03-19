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

        let done = send[0] == PortState::Done || recv.iter().all(|r| *r == PortState::Done);
        if done {
            send[0] = PortState::Done;
            recv.fill(PortState::Done);
            return Ok(());
        }

        let any_ready = recv.contains(&PortState::Ready);
        recv.fill(send[0]);
        send[0] = if any_ready {
            PortState::Ready
        } else {
            PortState::Blocked
        };
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
        let output_senders = send_ports[0].take().unwrap().parallel();
        let num_pipelines = output_senders.len();
        assert_eq!(num_pipelines, state.num_pipelines);

        let (mpsc_senders, mpsc_receivers): (Vec<_>, Vec<_>) = (0..num_pipelines)
            .map(|_| mpsc::channel::<Morsel>(1))
            .unzip();

        for recv_port in recv_ports {
            if let Some(recv) = recv_port.take() {
                let receivers = recv.parallel();
                let mpsc_senders_clone = mpsc_senders.clone();

                for (mut receiver, sender) in receivers.into_iter().zip(mpsc_senders_clone) {
                    let output_schema = self.output_schema.clone();
                    join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                        while let Ok(mut morsel) = receiver.recv().await {
                            // Ensure the morsel matches the expected output schema,
                            // casting nulls to the appropriate output type.
                            morsel.df_mut().ensure_matches_schema(&output_schema)?;

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

        // Each pipeline relabels morsel sequences independently of the others.
        // We first compute the `morsel_offset` as (max morsel sequence sent so far + 1), so this
        // phase never reuses sequence numbers from earlier phases.
        //
        // Then, each pipeline assigns sequences by:
        // - starting at `morsel_offset + pipeline_idx` (so pipelines start at different values),
        // - advancing by `num_pipelines` each time it emits a morsel.
        //
        // Example with 2 pipelines (num_pipelines = 2) and morsel_offset = 1000:
        // pipeline 0: 1000, 1002, 1004, ...
        // pipeline 1: 1001, 1003, 1005, ...
        //
        // This guarantees:
        // - Global uniqueness: no collisions with earlier phases, and no collisions across pipelines.
        // - Per-pipeline non-decreasing: each pipeline only moves forward by a fixed positive step.
        let morsel_offset = self.max_morsel_seq_sent.successor();

        let mut inner_handles = Vec::new();
        for (lane_idx, (mut mpsc_receiver, mut output_sender)) in
            mpsc_receivers.into_iter().zip(output_senders).enumerate()
        {
            inner_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let mut local_seq = morsel_offset.offset_by_u64(lane_idx as u64);
                let seq_step = num_pipelines as u64;
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
