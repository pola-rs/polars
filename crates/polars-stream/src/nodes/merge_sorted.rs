use std::sync::Arc;

use polars_core::prelude::ChunkCompareIneq;
use polars_core::schema::Schema;
use polars_ops::frame::_merge_sorted_dfs;
use polars_utils::pl_str::PlSmallStr;

use crate::morsel::SourceToken;
use crate::nodes::compute_node_prelude::*;

#[derive(Clone, Copy)]
enum Side {
    Left,
    Right,
}

#[derive(Debug)]
enum State {
    /// Merging values from buffered or ports.
    Merging,
    /// Passing values along from one of the ports.
    Passing,
}

pub struct MergeSortedNode {
    key_column_idx: usize,
    state: State,

    seq: MorselSeq,

    /// Already merged buffer.
    merged: DataFrame,

    /// Not yet merged buffer.
    unmerged: DataFrame,
    /// The port-side the unmerged buffer belongs to.
    unmerged_side: Side,
}

impl MergeSortedNode {
    pub fn new(schema: Arc<Schema>, key: PlSmallStr) -> Self {
        assert!(schema.contains(key.as_str()));
        let key_column_idx = schema.index_of(key.as_str()).unwrap();

        Self {
            key_column_idx,
            state: State::Merging,

            seq: MorselSeq::default(),

            merged: DataFrame::empty_with_schema(&schema),
            unmerged: DataFrame::empty_with_schema(&schema),
            unmerged_side: Side::Left,
        }
    }
}

impl ComputeNode for MergeSortedNode {
    fn name(&self) -> &str {
        "merge_sorted"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert_eq!(send.len(), 1);
        assert_eq!(recv.len(), 2);

        // If the output doesn't want any more data, transition to being done.
        if send[0] == PortState::Done {
            recv[0] = PortState::Done;
            recv[1] = PortState::Done;
        }

        self.state = match (recv[0], recv[1]) {
            _ if !self.merged.is_empty() || !self.unmerged.is_empty() => State::Merging,

            // If one of the ports is closed and the buffers are empty, we can just start passing
            // the morsels along the port.
            (PortState::Done, PortState::Done) => {
                send[0] = PortState::Done;
                State::Passing
            },
            (PortState::Done, _) | (_, PortState::Done) => State::Passing,

            _ => State::Merging,
        };

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
        let key_column_idx = self.key_column_idx;
        let merged = &mut self.merged;
        let unmerged = &mut self.unmerged;
        let unmerged_side = &mut self.unmerged_side;

        match self.state {
            State::Passing => {
                let mut send = send_ports[0].take().unwrap().serial();

                match (recv_ports[0].take(), recv_ports[1].take()) {
                    (None, None) => {},
                    (Some(port), None) | (None, Some(port)) => {
                        // @TODO: Turn into parallel passing.
                        let mut recv = port.serial();
                        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                            while let Ok(morsel) = recv.recv().await {
                                match send.send(morsel).await {
                                    Ok(_) => *seq = seq.successor(),
                                    Err(m) => {
                                        *merged = m.into_df();
                                        return Ok(());
                                    },
                                }
                            }

                            Ok(())
                        }));
                    },
                    (Some(_), Some(_)) => unreachable!(),
                }
            },
            State::Merging => {
                let source_token = SourceToken::new();
                let mut send = send_ports[0].take().unwrap().serial();

                let mut left = recv_ports[0].take().map(|p| p.serial());
                let mut right = recv_ports[1].take().map(|p| p.serial());

                join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                    loop {
                        if source_token.stop_requested() {
                            return Ok(());
                        }

                        if !merged.is_empty() {
                            // @TODO: Break unmerged up in smaller chunks as it might be very
                            // big.
                            let morsel = Morsel::new(merged.clone(), *seq, source_token.clone());
                            if send.send(morsel).await.is_err() {
                                return Ok(());
                            }

                            *merged = merged.clear();
                            *seq = seq.successor();
                        }

                        let opposite_port = match *unmerged_side {
                            Side::Left => right.as_mut(),
                            Side::Right => left.as_mut(),
                        };

                        let other_unmerged = match opposite_port {
                            None => {
                                if unmerged.is_empty() {
                                    return Ok(());
                                }

                                // @TODO: Break unmerged up in smaller chunks as it might be very
                                // big.
                                if send
                                    .send(Morsel::new(unmerged.clone(), *seq, source_token.clone()))
                                    .await
                                    .is_err()
                                {
                                    return Ok(());
                                };

                                *unmerged = unmerged.clear();
                                *seq = seq.successor();
                                return Ok(());
                            },
                            Some(port) => port.recv().await,
                        };

                        let Ok(other_unmerged) = other_unmerged else {
                            return Ok(());
                        };
                        let other_unmerged = other_unmerged.into_df();
                        let taken_unmerged = std::mem::take(unmerged);
                        let (left_unmerged, right_unmerged) = match *unmerged_side {
                            Side::Left => (taken_unmerged, other_unmerged),
                            Side::Right => (other_unmerged, taken_unmerged),
                        };

                        if left_unmerged.is_empty() {
                            *unmerged = right_unmerged;
                            *unmerged_side = Side::Right;
                            continue;
                        }
                        if right_unmerged.is_empty() {
                            *unmerged = left_unmerged;
                            *unmerged_side = Side::Left;
                            continue;
                        }

                        let left_key = &left_unmerged.get_columns()[key_column_idx];
                        let right_key = &right_unmerged.get_columns()[key_column_idx];

                        let left_key_last = left_key.slice(-1, 1);
                        let right_key_last = right_key.slice(-1, 1);

                        let (left_mergable, right_mergable) = if left_key_last.eq(&right_key_last) {
                            *unmerged = left_unmerged.slice(0, 0);
                            *unmerged_side = Side::Left;
                            (left_unmerged, right_unmerged)
                        } else if left_key_last.lt(&right_key_last)?.all() {
                            // @TODO: This is essentially search sorted, it should be optimized as such.
                            // @TODO: Does this work for nulls?
                            let cutoff = right_key
                                .gt(&left_key_last)?
                                .downcast_into_array()
                                .values()
                                .leading_zeros();

                            let (right_mergable, right_unmerged) =
                                right_unmerged.split_at(cutoff as i64);
                            *unmerged = right_unmerged;
                            *unmerged_side = Side::Right;
                            (left_unmerged, right_mergable)
                        } else {
                            // @TODO: This is essentially search sorted, it should be optimized as such.
                            // @TODO: Does this work for nulls?
                            let cutoff = left_key
                                .gt(&right_key_last)?
                                .downcast_into_array()
                                .values()
                                .leading_zeros();

                            let (left_mergable, left_unmerged) =
                                left_unmerged.split_at(cutoff as i64);
                            *unmerged = left_unmerged;
                            *unmerged_side = Side::Left;
                            (left_mergable, right_unmerged)
                        };

                        let left_s =
                            left_mergable.get_columns()[key_column_idx].as_materialized_series();
                        let right_s =
                            right_mergable.get_columns()[key_column_idx].as_materialized_series();

                        *merged = _merge_sorted_dfs(
                            &left_mergable,
                            &right_mergable,
                            left_s,
                            right_s,
                            false,
                        )?;
                    }
                }));
            },
        }
    }
}
