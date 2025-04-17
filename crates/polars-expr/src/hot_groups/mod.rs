use std::any::Any;

use polars_core::prelude::*;
use polars_utils::IdxSize;

use crate::EvictIdx;
use crate::hash_keys::HashKeys;

mod fixed_index_table;
mod row_encoded;

/// A HotGrouper maps keys to groups, such that duplicate keys map to the same
/// group. Unlike a Grouper it has a fixed size and will cause evictions rather
/// than growing.
pub trait HotGrouper: Any + Send + Sync {
    /// Creates a new empty HotGrouper similar to this one, with the given size.
    fn new_empty(&self, groups: usize) -> Box<dyn HotGrouper>;

    /// Returns the number of groups in this HotGrouper.
    fn num_groups(&self) -> IdxSize;

    /// Inserts the given keys into this Grouper, extending groups_idxs with
    /// the group index of keys[i].
    fn insert_keys(
        &mut self,
        keys: &HashKeys,
        hot_idxs: &mut Vec<IdxSize>,
        hot_group_idxs: &mut Vec<EvictIdx>,
        cold_idxs: &mut Vec<IdxSize>,
    );

    /// Get all the current hot keys, in group order.
    fn keys(&self) -> HashKeys;

    /// Get the number of evicted keys stored.
    fn num_evictions(&self) -> usize;

    /// Consume all the evicted keys from this HotGrouper.
    fn take_evicted_keys(&mut self) -> HashKeys;

    fn as_any(&self) -> &dyn Any;
}

pub fn new_hash_hot_grouper(key_schema: Arc<Schema>, num_groups: usize) -> Box<dyn HotGrouper> {
    Box::new(row_encoded::RowEncodedHashHotGrouper::new(
        key_schema, num_groups,
    ))
}
