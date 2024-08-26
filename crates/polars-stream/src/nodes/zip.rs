use std::collections::VecDeque;
use std::sync::Arc;

use polars_core::functions::concat_df_horizontal;
use polars_core::schema::Schema;
use polars_core::series::Series;
use polars_error::polars_ensure;

use super::compute_node_prelude::*;
use crate::morsel::SourceToken;

/// The head of an input stream.
#[derive(Debug)]
struct InputHead {
    /// The schema of the input, needed when creating full-null dataframes.
    schema: Arc<Schema>,

    // None when it is unknown whether this input stream is a broadcasting input or not.
    is_broadcast: Option<bool>,

    // True when there are no more morsels after the ones in the head.
    stream_exhausted: bool,

    // A FIFO queue of morsels belonging to this input stream.
    morsels: VecDeque<Morsel>,

    // The total length of the morsels in the input head.
    total_len: usize,
}

impl InputHead {
    fn new(schema: Arc<Schema>, may_broadcast: bool) -> Self {
        Self {
            schema,
            morsels: VecDeque::new(),
            is_broadcast: if may_broadcast { None } else { Some(false) },
            total_len: 0,
            stream_exhausted: false,
        }
    }

    fn add_morsel(&mut self, mut morsel: Morsel) {
        self.total_len += morsel.df().height();

        if self.is_broadcast.is_none() {
            if self.total_len > 1 {
                self.is_broadcast = Some(false);
            } else {
                // Make sure we don't deadlock trying to wait to clear our ambiguous
                // broadcast status.
                drop(morsel.take_consume_token());
            }
        }

        if morsel.df().height() > 0 {
            self.morsels.push_back(morsel);
        }
    }

    fn notify_no_more_morsels(&mut self) {
        if self.is_broadcast.is_none() {
            self.is_broadcast = Some(self.total_len == 1);
        }
        self.stream_exhausted = true;
    }

    fn ready_to_send(&self) -> bool {
        self.is_broadcast.is_some() && (self.total_len > 0 || self.stream_exhausted)
    }

    fn min_len(&self) -> Option<usize> {
        if self.is_broadcast == Some(false) {
            self.morsels.front().map(|m| m.df().height())
        } else {
            None
        }
    }

    fn take(&mut self, len: usize) -> DataFrame {
        if self.is_broadcast.unwrap() {
            self.morsels[0]
                .df()
                .iter()
                .map(|s| s.new_from_index(0, len))
                .collect()
        } else if self.total_len > 0 {
            self.total_len -= len;
            if self.morsels[0].df().height() == len {
                self.morsels.pop_front().unwrap().into_df()
            } else {
                let (head, tail) = self.morsels[0].df().split_at(len as i64);
                *self.morsels[0].df_mut() = tail;
                head
            }
        } else {
            self.schema
                .iter()
                .map(|(name, dtype)| Series::full_null(name, len, dtype))
                .collect()
        }
    }

    fn consume_broadcast(&mut self) -> DataFrame {
        assert!(self.is_broadcast == Some(true) && self.total_len == 1);
        let out = self.morsels.pop_front().unwrap().into_df();
        self.clear();
        out
    }

    fn clear(&mut self) {
        self.total_len = 0;
        self.is_broadcast = Some(false);
        self.morsels.clear();
    }
}

pub struct ZipNode {
    null_extend: bool,
    out_seq: MorselSeq,
    input_heads: Vec<InputHead>,
}

impl ZipNode {
    pub fn new(null_extend: bool, schemas: Vec<Arc<Schema>>) -> Self {
        let input_heads = schemas
            .into_iter()
            .map(|s| InputHead::new(s, !null_extend))
            .collect();
        Self {
            null_extend,
            out_seq: MorselSeq::new(0),
            input_heads,
        }
    }
}

impl ComputeNode for ZipNode {
    fn name(&self) -> &str {
        "zip"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(send.len() == 1);
        assert!(recv.len() == self.input_heads.len());

        let any_input_blocked = recv.iter().any(|s| *s == PortState::Blocked);

        let mut all_broadcast = true;
        let mut all_done_or_broadcast = true;
        let mut at_least_one_non_broadcast_done = false;
        let mut at_least_one_non_broadcast_nonempty = false;
        for (recv_idx, recv_state) in recv.iter().enumerate() {
            let input_head = &mut self.input_heads[recv_idx];
            if *recv_state == PortState::Done {
                input_head.notify_no_more_morsels();

                all_done_or_broadcast &=
                    input_head.is_broadcast == Some(true) || input_head.total_len == 0;
                at_least_one_non_broadcast_done |=
                    input_head.is_broadcast == Some(false) && input_head.total_len == 0;
            } else {
                all_done_or_broadcast = false;
            }

            all_broadcast &= input_head.is_broadcast == Some(true);
            at_least_one_non_broadcast_nonempty |=
                input_head.is_broadcast == Some(false) && input_head.total_len > 0;
        }

        if !self.null_extend {
            polars_ensure!(
                !(at_least_one_non_broadcast_done && at_least_one_non_broadcast_nonempty),
                ShapeMismatch: "zip node received non-equal length inputs"
            );
        }

        let all_output_sent = all_done_or_broadcast && !all_broadcast;

        let new_recv_state = if send[0] == PortState::Done || all_output_sent {
            for input_head in &mut self.input_heads {
                input_head.clear();
            }
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
        assert!(send.len() == 1);
        assert!(!recv.is_empty());
        let mut sender = send[0].take().unwrap().serial();
        let mut receivers: Vec<_> = recv.iter_mut().map(|r| Some(r.take()?.serial())).collect();

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let mut out = Vec::new();
            let source_token = SourceToken::new();
            loop {
                if source_token.stop_requested() {
                    break;
                }

                // Fill input heads until they are ready to send or the input is
                // exhausted (in this phase).
                let mut all_ready = true;
                for (recv_idx, opt_recv) in receivers.iter_mut().enumerate() {
                    if let Some(recv) = opt_recv {
                        while !self.input_heads[recv_idx].ready_to_send() {
                            if let Ok(morsel) = recv.recv().await {
                                self.input_heads[recv_idx].add_morsel(morsel);
                            } else {
                                break;
                            }
                        }
                    }
                    all_ready &= self.input_heads[recv_idx].ready_to_send();
                }

                if !all_ready {
                    // One or more of the input heads is exhausted (this phase).
                    break;
                }

                // TODO: recombine morsels to make sure the concatenation is
                // close to the ideal morsel size.

                // Compute common size and send a combined morsel.
                let Some(common_size) = self.input_heads.iter().flat_map(|h| h.min_len()).min()
                else {
                    // If all input heads are broadcasts we don't get a common size,
                    // we handle this below.
                    break;
                };

                for input_head in &mut self.input_heads {
                    out.push(input_head.take(common_size));
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

            // We can't continue because one or more input heads is empty or all
            // inputs are broadcasts. We must tell everyone to stop, unblock all
            // pipes by consuming all ConsumeTokens, and then store all data
            // that was still flowing through the pipelines into input_heads for
            // the next phase.
            for input_head in &mut self.input_heads {
                for morsel in &mut input_head.morsels {
                    morsel.source_token().stop();
                    drop(morsel.take_consume_token());
                }
            }

            for (recv_idx, opt_recv) in receivers.iter_mut().enumerate() {
                if let Some(recv) = opt_recv {
                    while let Ok(mut morsel) = recv.recv().await {
                        morsel.source_token().stop();
                        drop(morsel.take_consume_token());
                        self.input_heads[recv_idx].add_morsel(morsel);
                    }
                }
            }

            // If all our input heads are broadcasts we need to send a morsel
            // once with their output, consuming all broadcast inputs.
            let all_broadcast = self
                .input_heads
                .iter()
                .all(|h| h.is_broadcast == Some(true));
            if all_broadcast {
                for input_head in &mut self.input_heads {
                    out.push(input_head.consume_broadcast());
                }
                let out_df = concat_df_horizontal(&out, false)?;
                out.clear();

                let morsel = Morsel::new(out_df, self.out_seq, source_token.clone());
                self.out_seq = self.out_seq.successor();
                let _ = sender.send(morsel).await;
            }

            Ok(())
        }));
    }
}
