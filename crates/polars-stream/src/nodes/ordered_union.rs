use super::compute_node_prelude::*;

/// A node that first passes through all data from the first input, then the
/// second input, etc.
pub struct OrderedUnionNode {
    cur_input_idx: usize,
    max_morsel_seq_sent: MorselSeq,
    morsel_offset: MorselSeq,
}

impl OrderedUnionNode {
    pub fn new() -> Self {
        Self {
            cur_input_idx: 0,
            max_morsel_seq_sent: MorselSeq::new(0),
            morsel_offset: MorselSeq::new(0),
        }
    }
}

impl ComputeNode for OrderedUnionNode {
    fn name(&self) -> &str {
        "ordered_union"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(self.cur_input_idx <= recv.len() && send.len() == 1);

        // Skip inputs that are done.
        while self.cur_input_idx < recv.len() && recv[self.cur_input_idx] == PortState::Done {
            self.cur_input_idx += 1;
        }

        // Act like a normal pass-through node for the current input, or mark
        // ourselves as done if all inputs are handled.
        if self.cur_input_idx < recv.len() {
            core::mem::swap(&mut recv[self.cur_input_idx], &mut send[0]);
        } else {
            send[0] = PortState::Done;
        }

        // Mark all inputs after the current one as blocked.
        for r in recv.iter_mut().skip(self.cur_input_idx + 1) {
            *r = PortState::Blocked;
        }

        // Set the morsel offset one higher than any sent so far.
        self.morsel_offset = self.max_morsel_seq_sent.successor();
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
        let ready_count = recv.iter().filter(|r| r.is_some()).count();
        assert!(ready_count == 1 && send.len() == 1);
        let receivers = recv[self.cur_input_idx].take().unwrap().parallel();
        let senders = send[0].take().unwrap().parallel();

        let mut inner_handles = Vec::new();
        for (mut recv, mut send) in receivers.into_iter().zip(senders) {
            let morsel_offset = self.morsel_offset;
            inner_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let mut max_seq = MorselSeq::new(0);
                while let Ok(mut morsel) = recv.recv().await {
                    // Ensure the morsel sequence id stream is monotonic.
                    let seq = morsel.seq().offset_by(morsel_offset);
                    max_seq = max_seq.max(seq);

                    morsel.set_seq(seq);
                    if send.send(morsel).await.is_err() {
                        break;
                    }
                }
                max_seq
            }));
        }

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            // Update our global maximum.
            for handle in inner_handles {
                self.max_morsel_seq_sent = self.max_morsel_seq_sent.max(handle.await);
            }
            Ok(())
        }));
    }
}
