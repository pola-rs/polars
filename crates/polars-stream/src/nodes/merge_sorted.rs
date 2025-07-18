use std::collections::VecDeque;
use std::sync::Arc;

use polars_core::prelude::ChunkCompareIneq;
use polars_core::schema::Schema;
use polars_ops::frame::_merge_sorted_dfs;
use polars_utils::pl_str::PlSmallStr;

use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_primitives::connector::Receiver;
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::morsel::{SourceToken, get_ideal_morsel_size};
use crate::nodes::compute_node_prelude::*;

pub struct MergeSortedNode {
    key_column_idx: usize,

    seq: MorselSeq,

    starting_nulls: bool,

    // Not yet merged buffers.
    left_unmerged: VecDeque<DataFrame>,
    right_unmerged: VecDeque<DataFrame>,
}

impl MergeSortedNode {
    pub fn new(schema: Arc<Schema>, key: PlSmallStr) -> Self {
        assert!(schema.contains(key.as_str()));
        let key_column_idx = schema.index_of(key.as_str()).unwrap();

        Self {
            key_column_idx,

            seq: MorselSeq::default(),

            starting_nulls: false,

            left_unmerged: VecDeque::new(),
            right_unmerged: VecDeque::new(),
        }
    }
}

/// Find a part amongst both unmerged buffers which is mergeable.
///
/// This returns `None` if there is nothing mergeable at this point.
fn find_mergeable(
    left_unmerged: &mut VecDeque<DataFrame>,
    right_unmerged: &mut VecDeque<DataFrame>,

    key_column_idx: usize,

    is_first: bool,
    starting_nulls: &mut bool,
) -> PolarsResult<Option<(DataFrame, DataFrame)>> {
    fn first_non_empty(vd: &mut VecDeque<DataFrame>) -> Option<DataFrame> {
        let mut df = vd.pop_front()?;
        while df.height() == 0 {
            df = vd.pop_front()?;
        }
        Some(df)
    }

    loop {
        let (mut left, mut right) = match (
            first_non_empty(left_unmerged),
            first_non_empty(right_unmerged),
        ) {
            (Some(l), Some(r)) => (l, r),
            (Some(l), None) => {
                left_unmerged.push_front(l);
                return Ok(None);
            },
            (None, Some(r)) => {
                right_unmerged.push_front(r);
                return Ok(None);
            },
            (None, None) => return Ok(None),
        };

        let left_key = &left[key_column_idx];
        let right_key = &right[key_column_idx];

        let left_null_count = left_key.null_count();
        let right_null_count = right_key.null_count();

        let has_nulls = left_null_count > 0 || right_null_count > 0;

        // If we are on the first morsel we need to decide whether we have
        // nulls first or not.
        if is_first
            && has_nulls
            && (left_key.head(Some(1)).has_nulls() || right_key.head(Some(1)).has_nulls())
        {
            *starting_nulls = true;
        }

        // For both left and right, find row index of the minimum of the maxima
        // of the left and right key columns. We can safely merge until this
        // point.
        let mut left_cutoff = left.height();
        let mut right_cutoff = right.height();

        let left_key_last = left_key.tail(Some(1));
        let right_key_last = right_key.tail(Some(1));

        // We already made sure we had data to work with.
        assert!(!left_key_last.is_empty());
        assert!(!right_key_last.is_empty());

        if has_nulls {
            if *starting_nulls {
                // If there are starting nulls do those first, then repeat
                // without the nulls.
                left_cutoff = left_null_count;
                right_cutoff = right_null_count;
            } else {
                // If there are ending nulls then first do things without the
                // nulls and then repeat with only the nulls the nulls.
                let left_is_all_nulls = left_null_count == left.height();
                let right_is_all_nulls = right_null_count == right.height();

                match (left_is_all_nulls, right_is_all_nulls) {
                    (false, false) => {
                        let left_nulls;
                        let right_nulls;
                        (left, left_nulls) =
                            left.split_at((left.height() - left_null_count) as i64);
                        (right, right_nulls) =
                            right.split_at((right.height() - right_null_count) as i64);

                        left_unmerged.push_front(left_nulls);
                        left_unmerged.push_front(left);
                        right_unmerged.push_front(right_nulls);
                        right_unmerged.push_front(right);
                        continue;
                    },
                    (true, false) => left_cutoff = 0,
                    (false, true) => right_cutoff = 0,
                    (true, true) => {},
                }
            }
        } else if left_key_last.lt(&right_key_last)?.all() {
            // @TODO: This is essentially search sorted, but that does not
            // support categoricals at moment.
            let gt_mask = right_key.gt(&left_key_last)?;
            right_cutoff = gt_mask.downcast_as_array().values().leading_zeros();
        } else if left_key_last.gt(&right_key_last)?.all() {
            // @TODO: This is essentially search sorted, but that does not
            // support categoricals at moment.
            let gt_mask = left_key.gt(&right_key_last)?;
            left_cutoff = gt_mask.downcast_as_array().values().leading_zeros();
        }

        let left_mergeable: DataFrame;
        let right_mergeable: DataFrame;
        (left_mergeable, left) = left.split_at(left_cutoff as i64);
        (right_mergeable, right) = right.split_at(right_cutoff as i64);

        if !left.is_empty() {
            left_unmerged.push_front(left);
        }
        if !right.is_empty() {
            right_unmerged.push_front(right);
        }

        return Ok(Some((left_mergeable, right_mergeable)));
    }
}

impl ComputeNode for MergeSortedNode {
    fn name(&self) -> &str {
        "merge-sorted"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert_eq!(send.len(), 1);
        assert_eq!(recv.len(), 2);

        // Abstraction: we merge buffer state with port state so we can map
        // to one three possible 'effective' states:
        // no data now (_blocked); data available (); or no data anymore (_done)
        let left_done = recv[0] == PortState::Done && self.left_unmerged.is_empty();
        let right_done = recv[1] == PortState::Done && self.right_unmerged.is_empty();

        // We're done as soon as one side is done.
        if send[0] == PortState::Done || (left_done && right_done) {
            recv[0] = PortState::Done;
            recv[1] = PortState::Done;
            send[0] = PortState::Done;
            return Ok(());
        }

        // Each port is ready to proceed unless one of the other ports is effectively
        // blocked. For example:
        // - [Blocked with empty buffer, Ready] [Ready] returns [Ready, Blocked] [Blocked]
        // - [Blocked with non-empty buffer, Ready] [Ready] returns [Ready, Ready, Ready]
        let send_blocked = send[0] == PortState::Blocked;
        let left_blocked = recv[0] == PortState::Blocked && self.left_unmerged.is_empty();
        let right_blocked = recv[1] == PortState::Blocked && self.right_unmerged.is_empty();
        send[0] = if left_blocked || right_blocked {
            PortState::Blocked
        } else {
            PortState::Ready
        };
        recv[0] = if send_blocked || right_blocked {
            PortState::Blocked
        } else {
            PortState::Ready
        };
        recv[1] = if send_blocked || left_blocked {
            PortState::Blocked
        } else {
            PortState::Ready
        };

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
        assert_eq!(recv_ports.len(), 2);
        assert_eq!(send_ports.len(), 1);

        let send = send_ports[0].take().unwrap().parallel();

        let seq = &mut self.seq;
        let starting_nulls = &mut self.starting_nulls;
        let key_column_idx = self.key_column_idx;
        let left_unmerged = &mut self.left_unmerged;
        let right_unmerged = &mut self.right_unmerged;

        match (recv_ports[0].take(), recv_ports[1].take()) {
            // If we do not need to merge or flush anymore, just start passing the port in
            // parallel.
            (Some(port), None) | (None, Some(port))
                if left_unmerged.is_empty() && right_unmerged.is_empty() =>
            {
                let recv = port.parallel();
                let inner_handles = recv
                    .into_iter()
                    .zip(send)
                    .map(|(mut recv, mut send)| {
                        let morsel_offset = *seq;
                        scope.spawn_task(TaskPriority::High, async move {
                            let mut max_seq = morsel_offset;
                            while let Ok(mut morsel) = recv.recv().await {
                                // Ensure the morsel sequence id stream is monotone non-decreasing.
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
                        *seq = (*seq).max(handle.await);
                    }
                    Ok(())
                }));
            },

            // This is the base case. Either:
            // - Both streams are still open and we still need to merge.
            // - One or both streams are closed stream is closed and we still have some buffered
            // data.
            (left, right) => {
                async fn buffer_unmerged(
                    port: &mut Receiver<Morsel>,
                    unmerged: &mut VecDeque<DataFrame>,
                ) {
                    // If a stop was requested, we need to buffer the remaining
                    // morsels and trigger a phase transition.
                    let Ok(morsel) = port.recv().await else {
                        return;
                    };

                    // Request the port stop producing morsels.
                    morsel.source_token().stop();

                    // Buffer all the morsels that were already produced.
                    unmerged.push_back(morsel.into_df());
                    while let Ok(morsel) = port.recv().await {
                        unmerged.push_back(morsel.into_df());
                    }
                }

                let (mut distributor, dist_recv) =
                    distributor_channel(send.len(), *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

                let mut left = left.map(|p| p.serial());
                let mut right = right.map(|p| p.serial());

                join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                    let source_token = SourceToken::new();

                    // While we can still load data for the empty side.
                    while (left.is_some() || right.is_some())
                        && !(left.is_none() && left_unmerged.is_empty())
                        && !(right.is_none() && right_unmerged.is_empty())
                    {
                        // If we have morsels from both input ports, find until where we can merge
                        // them and send that on to be merged.
                        while let Some((left_mergeable, right_mergeable)) = find_mergeable(
                            left_unmerged,
                            right_unmerged,
                            key_column_idx,
                            seq.to_u64() == 0,
                            starting_nulls,
                        )? {
                            let left_mergeable =
                                Morsel::new(left_mergeable, *seq, source_token.clone());
                            *seq = seq.successor();

                            if distributor
                                .send((left_mergeable, right_mergeable))
                                .await
                                .is_err()
                            {
                                return Ok(());
                            };
                        }

                        if source_token.stop_requested() {
                            // Request that a port stops producing morsels and buffers all the
                            // remaining morsels.
                            if let Some(p) = &mut left {
                                buffer_unmerged(p, left_unmerged).await;
                            }
                            if let Some(p) = &mut right {
                                buffer_unmerged(p, right_unmerged).await;
                            }
                            break;
                        }

                        assert!(left_unmerged.is_empty() || right_unmerged.is_empty());
                        let (empty_port, empty_unmerged) = match (
                            left_unmerged.is_empty(),
                            right_unmerged.is_empty(),
                            left.as_mut(),
                            right.as_mut(),
                        ) {
                            (true, _, Some(left), _) => (left, &mut *left_unmerged),
                            (_, true, _, Some(right)) => (right, &mut *right_unmerged),

                            // If the port that is empty is closed, we don't need to merge anymore.
                            _ => break,
                        };

                        // Try to get a new morsel from the empty side.
                        let Ok(m) = empty_port.recv().await else {
                            if let Some(p) = &mut left {
                                buffer_unmerged(p, left_unmerged).await;
                            }
                            if let Some(p) = &mut right {
                                buffer_unmerged(p, right_unmerged).await;
                            }
                            break;
                        };
                        empty_unmerged.push_back(m.into_df());
                    }

                    // Clear out buffers until we cannot anymore. This helps allows us to go to the
                    // parallel case faster.
                    while let Some((left_mergeable, right_mergeable)) = find_mergeable(
                        left_unmerged,
                        right_unmerged,
                        key_column_idx,
                        seq.to_u64() == 0,
                        starting_nulls,
                    )? {
                        let left_mergeable =
                            Morsel::new(left_mergeable, *seq, source_token.clone());
                        *seq = seq.successor();

                        if distributor
                            .send((left_mergeable, right_mergeable))
                            .await
                            .is_err()
                        {
                            return Ok(());
                        };
                    }

                    // If one of the ports is done and does not have buffered data anymore, we
                    // flush the data on the other side. After this point, this node just pipes
                    // data through.
                    let pass = if left.is_none() && left_unmerged.is_empty() {
                        Some((right.as_mut(), &mut *right_unmerged))
                    } else if right.is_none() && right_unmerged.is_empty() {
                        Some((left.as_mut(), &mut *left_unmerged))
                    } else {
                        None
                    };
                    if let Some((pass_port, pass_unmerged)) = pass {
                        for df in std::mem::take(pass_unmerged) {
                            let m = Morsel::new(df, *seq, source_token.clone());
                            *seq = seq.successor();
                            if distributor.send((m, DataFrame::empty())).await.is_err() {
                                return Ok(());
                            }
                        }

                        // Start passing on the port that is port that is still open.
                        if let Some(pass_port) = pass_port {
                            let Ok(mut m) = pass_port.recv().await else {
                                return Ok(());
                            };
                            if source_token.stop_requested() {
                                m.source_token().stop();
                            }
                            m.set_seq(*seq);
                            *seq = seq.successor();
                            if distributor.send((m, DataFrame::empty())).await.is_err() {
                                return Ok(());
                            }

                            while let Ok(mut m) = pass_port.recv().await {
                                m.set_seq(*seq);
                                *seq = seq.successor();
                                if distributor.send((m, DataFrame::empty())).await.is_err() {
                                    return Ok(());
                                }
                            }
                        }
                    }

                    Ok(())
                }));

                // Task that actually merges the two dataframes. Since this merge might be very
                // expensive, this is split over several tasks.
                join_handles.extend(dist_recv.into_iter().zip(send).map(|(mut recv, mut send)| {
                    let ideal_morsel_size = get_ideal_morsel_size();
                    scope.spawn_task(TaskPriority::High, async move {
                        while let Ok((left, right)) = recv.recv().await {
                            // When we are flushing the buffer, we will just send one morsel from
                            // the input. We don't want to mess with the source token or wait group
                            // and just pass it on.
                            if right.is_empty() {
                                if send.send(left).await.is_err() {
                                    return Ok(());
                                }
                                continue;
                            }

                            let (left, seq, source_token, wg) = left.into_inner();
                            assert!(wg.is_none());

                            let left_s = left[key_column_idx].as_materialized_series();
                            let right_s = right[key_column_idx].as_materialized_series();

                            let merged = _merge_sorted_dfs(&left, &right, left_s, right_s, false)?;

                            if ideal_morsel_size > 1 && merged.height() > ideal_morsel_size {
                                // The merged dataframe will have at most doubled in size from the
                                // input so we can divide by half.
                                let (m1, m2) = merged.split_at((merged.height() / 2) as i64);

                                // MorselSeq have to be monotonely non-decreasing so we can
                                // pass the same sequence token twice.
                                let morsel = Morsel::new(m1, seq, source_token.clone());
                                if send.send(morsel).await.is_err() {
                                    break;
                                }
                                let morsel = Morsel::new(m2, seq, source_token.clone());
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
        }
    }
}
