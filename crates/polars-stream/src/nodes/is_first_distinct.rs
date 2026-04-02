use std::sync::Arc;

use arrow::array::BooleanArray;
use arrow::bitmap::BitmapBuilder;
use polars_core::prelude::*;
use polars_expr::groups::{Grouper, new_hash_grouper};
use polars_expr::hash_keys::HashKeys;
use polars_utils::IdxSize;

use super::compute_node_prelude::*;

/// A node which adds for each row whether it's the first time this row is seen, based on key cols.
pub struct IsFirstDistinctNode {
    key_schema: Arc<Schema>,
    out_name: PlSmallStr,
    grouper: Box<dyn Grouper>,
    subset: Vec<IdxSize>,
    group_idxs: Vec<IdxSize>,
    max_uniq_group_idx: IdxSize,
    random_state: PlRandomState,
}

impl IsFirstDistinctNode {
    pub fn new(key_schema: Arc<Schema>, out_name: PlSmallStr, random_state: PlRandomState) -> Self {
        let grouper = new_hash_grouper(key_schema.clone());
        Self {
            key_schema,
            out_name,
            grouper,
            subset: Vec::new(),
            group_idxs: Vec::new(),
            max_uniq_group_idx: 0,
            random_state,
        }
    }
}

impl ComputeNode for IsFirstDistinctNode {
    fn name(&self) -> &str {
        "is_first_distinct"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);
        recv.swap_with_slice(send);
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
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);
        let mut recv = recv_ports[0].take().unwrap().serial();
        let mut send = send_ports[0].take().unwrap().serial();

        let slf = &mut *self;
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = recv.recv().await {
                let morsel = morsel.map(|mut df| {
                    let key_df = df.select(slf.key_schema.iter_names()).unwrap();
                    let hash_keys =
                        HashKeys::from_df(&key_df, slf.random_state.clone(), true, false);
                    let mut distinct = BitmapBuilder::with_capacity(df.height());
                    unsafe {
                        slf.subset
                            .extend(slf.subset.len() as IdxSize..df.height() as IdxSize);
                        slf.grouper.insert_keys_subset(
                            &hash_keys,
                            &slf.subset[..df.height()],
                            Some(&mut slf.group_idxs),
                        );

                        for g in slf.group_idxs.drain(..) {
                            let new = g == slf.max_uniq_group_idx;
                            distinct.push_unchecked(new);
                            slf.max_uniq_group_idx += new as IdxSize;
                        }
                    }

                    let arr = BooleanArray::from(distinct.freeze());
                    let col = BooleanChunked::with_chunk(slf.out_name.clone(), arr).into_column();
                    df.with_column(col).unwrap();
                    df
                });
                if send.send(morsel).await.is_err() {
                    break;
                }
            }

            Ok(())
        }));
    }
}
