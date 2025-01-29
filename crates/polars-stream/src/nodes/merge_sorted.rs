use std::collections::VecDeque;
use std::sync::Arc;

use polars_core::prelude::ChunkCompareIneq;
use polars_core::schema::Schema;
use polars_ops::frame::_merge_sorted_dfs;
use polars_utils::pl_str::PlSmallStr;

use crate::async_primitives::connector::Receiver;
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::morsel::{get_ideal_morsel_size, SourceToken};
use crate::nodes::compute_node_prelude::*;
use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;

#[derive(Debug)]
enum State {
    /// Merging values from buffered or ports.
    Merging,
    /// Flushing buffer values.
    Flushing,
    /// Passing values along from one of the ports.
    Piping,
}

pub struct MergeSortedNode {
    num_pipelines: usize,
    key_column_idx: usize,
    state: State,

    seq: MorselSeq,
    max_seq_sent: MorselSeq,

    // Not yet merged buffers.
    left_unmerged: VecDeque<DataFrame>,
    right_unmerged: VecDeque<DataFrame>,
}

impl MergeSortedNode {
    pub fn new(schema: Arc<Schema>, key: PlSmallStr) -> Self {
        assert!(schema.contains(key.as_str()));
        let key_column_idx = schema.index_of(key.as_str()).unwrap();

        Self {
            num_pipelines: 1,
            key_column_idx,
            state: State::Merging,

            seq: MorselSeq::default(),
            max_seq_sent: MorselSeq::default(),

            left_unmerged: VecDeque::new(),
            right_unmerged: VecDeque::new(),
        }
    }

    fn has_buffered_data(&self) -> bool {
        !self.left_unmerged.is_empty() || !self.right_unmerged.is_empty()
    }
}

impl ComputeNode for MergeSortedNode {
    fn name(&self) -> &str {
        "merge_sorted"
    }

    fn initialize(&mut self, num_pipelines: usize) {
        self.num_pipelines = num_pipelines;
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert_eq!(send.len(), 1);
        assert_eq!(recv.len(), 2);

        // If the output doesn't want any more data, transition to being done.
        if send[0] == PortState::Done {
            recv[0] = PortState::Done;
            recv[1] = PortState::Done;

            self.left_unmerged.clear();
            self.right_unmerged.clear();
            self.state = State::Piping;

            return Ok(());
        }

        if recv[0] == PortState::Done || recv[1] == PortState::Done {
            if self.has_buffered_data() {
                self.state = State::Flushing;
            } else {
                self.state = State::Piping;
            }
        } else {
            self.state = State::Merging;
        }

        match self.state {
            State::Merging if send[0] == PortState::Blocked => {
                recv[0] = PortState::Blocked;
                recv[1] = PortState::Blocked;
            },
            State::Merging
                if recv[0] == PortState::Blocked
                    && recv[1] == PortState::Blocked
                    && !self.has_buffered_data() =>
            {
                send[0] = PortState::Blocked;
            },
            State::Merging => {
                if recv[0] != PortState::Blocked {
                    recv[0] = PortState::Ready;
                }
                if recv[1] != PortState::Blocked {
                    recv[1] = PortState::Ready;
                }
                send[0] = PortState::Ready;
            },

            State::Flushing => {
                if recv[0] != PortState::Done {
                    recv[0] = PortState::Blocked;
                }
                if recv[1] != PortState::Done {
                    recv[1] = PortState::Blocked;
                }
            },

            State::Piping if recv[0] == PortState::Done && recv[1] == PortState::Done => {
                send[0] = PortState::Done;
            },
            State::Piping if recv[0] == PortState::Blocked || recv[1] == PortState::Blocked => {
                send[0] = PortState::Blocked;
            },
            State::Piping if send[0] == PortState::Blocked => {
                if recv[0] != PortState::Done {
                    recv[0] = PortState::Blocked;
                }
                if recv[1] != PortState::Done {
                    recv[1] = PortState::Blocked;
                }
            },
            State::Piping => {
                if recv[0] != PortState::Done {
                    recv[0] = PortState::Ready;
                }
                if recv[1] != PortState::Done {
                    recv[1] = PortState::Ready;
                }
                send[0] = PortState::Ready;
            },
        }

        // Set the morsel offset one higher than any sent so far.
        self.seq = self.max_seq_sent.successor();

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert_eq!(recv_ports.len(), 2);
        assert_eq!(send_ports.len(), 1);

        let seq = &mut self.seq;
        let max_seq_sent = &mut self.max_seq_sent;
        let key_column_idx = self.key_column_idx;
        let left_unmerged = &mut self.left_unmerged;
        let right_unmerged = &mut self.right_unmerged;

        let source_token = SourceToken::new();

        match self.state {
            State::Merging => {
                /// Request that a port stops producing [`Morsel`]s and buffer all the remaining
                /// [`Morsel`]s.
                async fn buffer_port(
                    port: &mut Receiver<Morsel>,
                    buffer: &mut VecDeque<DataFrame>,
                ) {
                    // If a stop was requested, we need to buffer the remaining
                    // morsels and trigger a phase transition.
                    let Ok(morsel) = port.recv().await else {
                        return;
                    };

                    // Request the port stop producing morsels.
                    morsel.source_token().stop();

                    // Buffer all the morsels that were already produced.
                    buffer.push_back(morsel.into_df());
                    while let Ok(morsel) = port.recv().await {
                        buffer.push_back(morsel.into_df());
                    }
                }

                let send = send_ports[0].take().unwrap().parallel();

                let (mut distributor, dist_recv) =
                    distributor_channel(self.num_pipelines, DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

                let mut left = recv_ports[0].take().unwrap().serial();
                let mut right = recv_ports[1].take().unwrap().serial();

                let _source_token = source_token.clone();
                join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                    let mut is_stopped = false;
                    let source_token = _source_token;

                    loop {
                        if source_token.stop_requested() {
                            is_stopped = true;
                        }

                        // If we have morsels from both input ports, find until where we can merge
                        // them and send that on to be merged.
                        while !left_unmerged.is_empty() && !right_unmerged.is_empty() {
                            let mut left = left_unmerged.pop_front().unwrap();
                            let mut right = right_unmerged.pop_front().unwrap();

                            // Ensure that we have some data to merge.
                            if left.is_empty() || right.is_empty() {
                                if !left.is_empty() {
                                    left_unmerged.push_front(left);
                                }
                                if !right.is_empty() {
                                    right_unmerged.push_front(right);
                                }
                                continue;
                            }

                            let left_key = &left[key_column_idx];
                            let right_key = &right[key_column_idx];

                            let left_key_last = left_key.tail(Some(1));
                            let right_key_last = right_key.tail(Some(1));

                            // We already made sure we had data to work with.
                            assert!(!left_key_last.is_empty());
                            assert!(!right_key_last.is_empty());

                            // For both left and right, find row index of the minimum of the maxima
                            // of the left and right key columns. We can savely merge until this
                            // point.
                            let mut left_cutoff = left.height();
                            let mut right_cutoff = right.height();
                            if left_key_last.lt(&right_key_last)?.all() {
                                // @TODO: This is essentially search sorted, but that does not
                                // support categoricals at moment.
                                // @TODO: Does this work for nulls?
                                right_cutoff = right_key
                                    .gt(&left_key_last)?
                                    .downcast_into_array()
                                    .values()
                                    .leading_zeros();
                            } else if left_key_last.gt(&right_key_last)?.all() {
                                // @TODO: This is essentially search sorted, but that does not
                                // support categoricals at moment.
                                // @TODO: Does this work for nulls?
                                left_cutoff = left_key
                                    .gt(&right_key_last)?
                                    .downcast_into_array()
                                    .values()
                                    .leading_zeros();
                            }

                            let left_mergeable: DataFrame;
                            let right_mergeable: DataFrame;
                            (left_mergeable, left) = left.split_at(left_cutoff as i64);
                            (right_mergeable, right) = right.split_at(right_cutoff as i64);

                            if distributor
                                .send((*seq, left_mergeable, right_mergeable))
                                .await
                                .is_err()
                            {
                                return Ok(());
                            };
                            // The merging task might split the merged dataframe in two.
                            *seq = seq.successor().successor();

                            if !left.is_empty() {
                                left_unmerged.push_front(left);
                            }
                            if !right.is_empty() {
                                right_unmerged.push_front(right);
                            }
                        }

                        if is_stopped {
                            buffer_port(&mut left, left_unmerged).await;
                            buffer_port(&mut right, right_unmerged).await;
                            return Ok(());
                        }

                        let (empty_port, empty_unmerged) = if left_unmerged.is_empty() {
                            (&mut left, &mut *left_unmerged)
                        } else {
                            (&mut right, &mut *right_unmerged)
                        };

                        // Try to get a new morsel from the empty side.
                        let Ok(received_unmerged) = empty_port.recv().await else {
                            is_stopped = true;
                            continue;
                        };
                        empty_unmerged.push_back(received_unmerged.into_df());
                    }
                }));

                // Task that actually merges the two dataframes. Since this merge might be very
                // expensive, this is split over several tasks.
                join_handles.extend(dist_recv.into_iter().zip(send).map(|(mut recv, mut send)| {
                    let source_token = source_token.clone();
                    let ideal_morsel_size = get_ideal_morsel_size();
                    scope.spawn_task(TaskPriority::High, async move {
                        while let Ok((seq, left, right)) = recv.recv().await {
                            let left_s = left[key_column_idx].as_materialized_series();
                            let right_s = right[key_column_idx].as_materialized_series();

                            let merged = _merge_sorted_dfs(&left, &right, left_s, right_s, false)?;

                            if ideal_morsel_size > 1 && merged.height() > ideal_morsel_size {
                                // The merged dataframe will have at most doubled in size from the
                                // input so we can divide by half.
                                let (m1, m2) = merged.split_at((merged.height() / 2) as i64);

                                let morsel = Morsel::new(m1, seq, source_token.clone());
                                if send.send(morsel).await.is_err() {
                                    break;
                                }
                                let morsel = Morsel::new(m2, seq.successor(), source_token.clone());
                                if send.send(morsel).await.is_err() {
                                    break;
                                }
                            } else {
                                let morsel = Morsel::new(merged, seq, source_token.clone());
                                if send.send(morsel).await.is_err() {
                                    break;
                                }
                            }
                        }

                        Ok(())
                    })
                }));
            },

            // If we have data left over from merging. We block the input ports until the buffered
            // data is flushed out.
            State::Flushing => {
                assert!(recv_ports[0].is_none());
                assert!(recv_ports[1].is_none());

                assert!(left_unmerged.is_empty() || right_unmerged.is_empty());

                let mut send = send_ports[0].take().unwrap().serial();

                let buffer = if left_unmerged.is_empty() {
                    right_unmerged
                } else {
                    left_unmerged
                };

                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    while let Some(df) = buffer.pop_front() {
                        if send
                            .send(Morsel::new(df, *seq, source_token.clone()))
                            .await
                            .is_err()
                        {
                            break;
                        }
                        *seq = seq.successor();

                        if source_token.stop_requested() {
                            break;
                        }
                    }

                    Ok(())
                }));
            },

            // When one of the input ports is closed, we don't have to merge anymore so we start
            // passing the data along.
            State::Piping => {
                let send = send_ports[0].take().unwrap().parallel();

                assert!(left_unmerged.is_empty());
                assert!(right_unmerged.is_empty());

                assert!(recv_ports[0].is_none() || recv_ports[1].is_none());

                // When this gets spawned, exactly one port should be open.
                let port = recv_ports[0].take().or(recv_ports[1].take());
                let port = port.unwrap();

                let inner_handles = port
                    .parallel()
                    .into_iter()
                    .zip(send)
                    .map(|(mut recv, mut send)| {
                        let morsel_offset = *seq;
                        scope.spawn_task(TaskPriority::High, async move {
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
                        })
                    })
                    .collect::<Vec<_>>();

                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    // Update our global maximum.
                    for handle in inner_handles {
                        *max_seq_sent = (*max_seq_sent).max(handle.await);
                    }
                    Ok(())
                }));
            },
        }
    }
}
