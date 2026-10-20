use std::sync::Arc;

use polars_core::prelude::*;
use polars_expr::groups::{Grouper, new_hash_grouper};
use polars_expr::hash_keys::HashKeys;
use polars_utils::IdxSize;

use super::compute_node_prelude::*;

/// A node which assigns a unique, dense ID to each distinct value, where equal
/// values receive the same ID. IDs are assigned in order of first occurrence,
/// starting from 0.
///
/// This uses a shared [`Grouper`] (see `is_first_distinct.rs` for the same
/// pattern): the group index a value is mapped to is itself a collision-free
/// unique ID. Because a single grouper is shared across all morsels the node
/// runs serially, which also preserves first-occurrence order.
pub struct UniqueIdNode {
    key_schema: Arc<Schema>,
    out_name: PlSmallStr,
    grouper: Box<dyn Grouper>,
    /// Cached `0..n` identity subset, the only way to feed whole morsels to
    /// [`Grouper::insert_keys_subset`]. Grows to the largest morsel seen.
    subset: Vec<IdxSize>,
    random_state: PlRandomState,
}

impl UniqueIdNode {
    pub fn new(key_schema: Arc<Schema>, random_state: PlRandomState) -> Self {
        let out_name = key_schema.get_at_index(0).unwrap().0.clone();
        let grouper = new_hash_grouper(key_schema.clone());
        Self {
            key_schema,
            out_name,
            grouper,
            subset: Vec::new(),
            random_state,
        }
    }
}

impl ComputeNode for UniqueIdNode {
    fn name(&self) -> &str {
        "unique_id"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);
        recv.swap_with_slice(send);
        if recv[0] == PortState::Done || send[0] == PortState::Done {
            self.grouper = self.grouper.new_empty();
            self.subset = Vec::new();
        }
        Ok(())
    }

    fn memory_usage(&self) -> NodeMemoryUsage {
        // The grouper grows with the number of distinct values.
        NodeMemoryUsage::Unbounded
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
                let morsel = morsel
                    .map(|df| {
                        let key_df = df.select(slf.key_schema.iter_names()).unwrap();
                        let hash_keys =
                            HashKeys::from_df(&key_df, slf.random_state.clone(), true, false);
                        // `insert_keys_subset` reserves exactly `df.height()`, so this
                        // is a single allocation that is then moved into the output.
                        let mut group_idxs = Vec::new();
                        unsafe {
                            slf.subset
                                .extend(slf.subset.len() as IdxSize..df.height() as IdxSize);
                            slf.grouper.insert_keys_subset(
                                &hash_keys,
                                &slf.subset[..df.height()],
                                Some(&mut group_idxs),
                            );
                        }

                        // The group index is a dense, collision-free unique ID.
                        let col = IdxCa::new_vec(slf.out_name.clone(), group_idxs).into_column();
                        unsafe { DataFrame::new_unchecked(col.len(), vec![col]) }
                    })
                    .await;
                if send.send(morsel).await.is_err() {
                    break;
                }
            }

            Ok(())
        }));
    }
}
