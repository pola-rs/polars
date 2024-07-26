use std::collections::VecDeque;

use polars_core::functions::concat_df_horizontal;

use super::compute_node_prelude::*;
use crate::morsel::SourceToken;

pub struct ZipNode {
    out_seq: MorselSeq,
    input_heads: Vec<VecDeque<Morsel>>,
}

impl ZipNode {
    pub fn new() -> Self {
        Self {
            out_seq: MorselSeq::new(0),
            input_heads: Vec::new(),
        }
    }
}

impl ComputeNode for ZipNode {
    fn name(&self) -> &str {
        "zip"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) {
        assert!(send.len() == 1);
        assert!(!recv.is_empty());

        let any_input_blocked = recv.iter().any(|s| *s == PortState::Blocked);

        let mut all_done = true;
        let mut at_least_one_done = false;
        let mut at_least_one_nonempty = false;
        for (recv_idx, recv_state) in recv.iter().enumerate() {
            let is_empty = self
                .input_heads
                .get(recv_idx)
                .map(|h| h.is_empty())
                .unwrap_or(true);
            at_least_one_nonempty |= !is_empty;
            if *recv_state == PortState::Done {
                all_done &= is_empty;
                at_least_one_done |= is_empty;
            } else {
                all_done = false;
            }
        }

        assert!(
            !(at_least_one_done && at_least_one_nonempty),
            "zip received non-equal length inputs"
        );

        let new_recv_state = if send[0] == PortState::Done || all_done {
            self.input_heads.clear();
            send[0] = PortState::Done;
            PortState::Done
        } else if send[0] == PortState::Blocked || any_input_blocked {
            send[0] = if any_input_blocked {
                PortState::Blocked
            } else {
                PortState::Ready
            };
            PortState::Blocked
        } else {
            send[0] = PortState::Ready;
            PortState::Ready
        };

        for r in recv {
            *r = new_recv_state;
        }
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv: &mut [Option<RecvPort<'_>>],
        send: &mut [Option<SendPort<'_>>],
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(send.len() == 1);
        assert!(!recv.is_empty());
        let mut sender = send[0].take().unwrap().serial();
        let mut receivers: Vec<_> = recv.iter_mut().map(|r| Some(r.take()?.serial())).collect();

        self.input_heads.resize_with(receivers.len(), VecDeque::new);

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let mut out = Vec::new();
            let source_token = SourceToken::new();
            loop {
                if source_token.stop_requested() {
                    break;
                }

                // Fill input heads with non-empty morsels.
                for (recv_idx, opt_recv) in receivers.iter_mut().enumerate() {
                    if let Some(recv) = opt_recv {
                        if self.input_heads[recv_idx].is_empty() {
                            while let Ok(morsel) = recv.recv().await {
                                if morsel.df().height() > 0 {
                                    self.input_heads[recv_idx].push_back(morsel);
                                    break;
                                }
                            }
                        }
                    }
                }

                // TODO: recombine morsels to make sure the concatenation is
                // close to the ideal morsel size.

                // Compute common size and send a combined morsel.
                let common_size = self
                    .input_heads
                    .iter()
                    .map(|h| h.front().map(|m| m.df().height()).unwrap_or(0))
                    .min()
                    .unwrap();
                if common_size == 0 {
                    // One or more of the input heads is exhausted (this phase).
                    break;
                }

                for input_head in &mut self.input_heads {
                    if input_head[0].df().height() == common_size {
                        out.push(input_head.pop_front().unwrap().into_df());
                    } else {
                        let (head, tail) = input_head[0].df().split_at(common_size as i64);
                        *input_head[0].df_mut() = tail;
                        out.push(head);
                    }
                }

                let out_df = concat_df_horizontal(&out, false)?;
                out.clear();

                let morsel = Morsel::new(out_df, self.out_seq, source_token.clone());
                self.out_seq = self.out_seq.successor();
                if sender.send(morsel).await.is_err() {
                    // Our receiver is no longer interested in any data, no
                    // need store the rest of the incoming stream, can directly
                    // return.
                    return Ok(());
                }
            }

            // We can't continue because one or more input heads is empty. We
            // must tell everyone to stop, unblock all pipes by consuming
            // all ConsumeTokens, and then store all data that was still flowing
            // through the pipelines into input_heads for the next phase.
            for input_head in &mut self.input_heads {
                for morsel in input_head {
                    morsel.source_token().stop();
                    drop(morsel.take_consume_token());
                }
            }

            for (recv_idx, opt_recv) in receivers.iter_mut().enumerate() {
                if let Some(recv) = opt_recv {
                    while let Ok(mut morsel) = recv.recv().await {
                        morsel.source_token().stop();
                        drop(morsel.take_consume_token());
                        if morsel.df().height() > 0 {
                            self.input_heads[recv_idx].push_back(morsel);
                        }
                    }
                }
            }

            Ok(())
        }));
    }
}
