use std::sync::Arc;

use polars_core::prelude::ChunkCompareIneq;
use polars_core::schema::Schema;
use polars_ops::frame::_merge_sorted_dfs;
use polars_utils::pl_str::PlSmallStr;

use crate::async_primitives::connector::connector;
use crate::morsel::{get_ideal_morsel_size, SourceToken};
use crate::nodes::compute_node_prelude::*;

#[derive(Clone, Copy)]
enum Side {
    Left,
    Right,
}

pub struct MergeSortedNode {
    key_column_idx: usize,

    seq: MorselSeq,

    merged: DataFrame,
    mergable: (DataFrame, DataFrame),

    unmerged: DataFrame,
    unmerged_side: Side,
}

impl MergeSortedNode {
    pub fn new(schema: Arc<Schema>, key: PlSmallStr) -> Self {
        assert!(schema.contains(key.as_str()));
        let key_column_idx = schema.index_of(key.as_str()).unwrap();

        Self {
            key_column_idx,
            seq: MorselSeq::default(),

            merged: DataFrame::empty_with_schema(&schema),
            mergable: (
                DataFrame::empty_with_schema(&schema),
                DataFrame::empty_with_schema(&schema),
            ),

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
        assert!(recv.len() == 2 && send.len() == 1);

        // If the output doesn't want any more data, transition to being done.
        if dbg!(send[0]) == PortState::Done {
            recv[0] = PortState::Done;
            recv[1] = PortState::Done;
        }

        dbg!(&self.unmerged);
        dbg!(&self.mergable.0);
        dbg!(&self.mergable.1);
        dbg!(&self.merged);

        if dbg!(matches!(recv[0], PortState::Done))
            && dbg!(matches!(recv[1], PortState::Done))
            && self.unmerged.is_empty()
            && self.mergable.0.is_empty()
            && self.mergable.1.is_empty()
            && self.merged.is_empty()
        {
            send[0] = PortState::Done;
        }

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
        assert!(recv_ports.len() == 2);
        assert!(send_ports.len() == 1);

        let mut send = send_ports[0].take().unwrap().serial();

        let left = recv_ports[0].take().map(|p| p.serial());
        let right = recv_ports[1].take().map(|p| p.serial());

        let (mut emit_send, mut emit_recv) = connector();

        let ideal_morsel_size = get_ideal_morsel_size();

        let seq = &mut self.seq;
        let key_column_idx = self.key_column_idx;
        let merged = &mut self.merged;
        let mergable = &mut self.mergable;
        let unmerged = &mut self.unmerged;
        let unmerged_side = &mut self.unmerged_side;

        join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
            if !mergable.0.is_empty() || !mergable.1.is_empty() {
                let l;
                let r;
                (mergable.0, l) = mergable.0.split_at(0);
                (mergable.1, r) = mergable.1.split_at(0);
                if let Err(m) = emit_send.send((l, r)).await {
                    *mergable = m;
                    return Ok(());
                }
            }

            match (left, right) {
                (None, None) => {
                    if !unmerged.is_empty() {
                        let um;
                        (*unmerged, um) = unmerged.split_at(0);
                        if let Err((m, _)) = emit_send.send((um, DataFrame::empty())).await {
                            *unmerged = m;
                            return Ok(());
                        }
                    }
                },
                (mut left, mut right) => {
                    loop {
                        if left.is_none() && matches!(*unmerged_side, Side::Right) {
                            loop {
                                if !unmerged.is_empty() {
                                    let um;
                                    (*unmerged, um) = unmerged.split_at(0);
                                    if let Err((m, _)) = emit_send.send((um, DataFrame::empty())).await {
                                        *unmerged = m;
                                        return Ok(());
                                    }
                                }

                                let Ok(um) = right.as_mut().unwrap().recv().await else {
                                    break;
                                };
                                *unmerged = dbg!(um.into_df());
                            }

                            return Ok(())
                        } else if right.is_none() && matches!(*unmerged_side, Side::Left) {
                            loop {
                                if !unmerged.is_empty() {
                                    let um;
                                    (*unmerged, um) = unmerged.split_at(0);
                                    if let Err((m, _)) = emit_send.send((um, DataFrame::empty())).await {
                                        *unmerged = m;
                                        return Ok(());
                                    }
                                }

                                let Ok(um) = left.as_mut().unwrap().recv().await else {
                                    break;
                                };
                                *unmerged = dbg!(um.into_df());
                            }

                            return Ok(())
                        }

                        let other_unmerged = match *unmerged_side {
                            Side::Left => right.as_mut().unwrap().recv().await,
                            Side::Right => left.as_mut().unwrap().recv().await,
                        };
                        let Ok(other_unmerged) = other_unmerged else {
                            return Ok(());
                        };
                        let other_unmerged = dbg!(other_unmerged.into_df());
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

                        dbg!(&left_unmerged);
                        dbg!(&right_unmerged);

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

                        if let Err(m) = emit_send.send((left_mergable, right_mergable)).await {
                            *mergable = m;
                            return Ok(());
                        }
                    }
                },
            }

            Ok(())
        }));

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let source_token = SourceToken::new();

            loop {
                if !merged.is_empty() {
                    let num_morsels = merged.height().div_ceil(ideal_morsel_size);
                    let max_morsel_size = merged.height() / num_morsels;

                    for _ in 0..num_morsels {
                        let df = merged.head(Some(max_morsel_size));
                        dbg!(&df);
                        let morsel_height = df.height();
                        let morsel = Morsel::new(df, *seq, source_token.clone());

                        if send.send(morsel).await.is_err() {
                            return Ok(());
                        }

                        *merged =
                            merged.slice(morsel_height as i64, merged.height() - morsel_height);
                        *seq = seq.successor();

                        if source_token.stop_requested() {
                            return Ok(());
                        }
                    }
                }

                let Ok((left_mergable, right_mergable)) = emit_recv.recv().await else {
                    break;
                };

                if left_mergable.is_empty() {
                    *merged = right_mergable;
                } else if right_mergable.is_empty() {
                    *merged = left_mergable;
                } else {
                    let left_s =
                        left_mergable.get_columns()[key_column_idx].as_materialized_series();
                    let right_s =
                        right_mergable.get_columns()[key_column_idx].as_materialized_series();

                    *merged =
                        _merge_sorted_dfs(&left_mergable, &right_mergable, left_s, right_s, false)?;
                }
            }

            Ok(())
        }));
    }
}
