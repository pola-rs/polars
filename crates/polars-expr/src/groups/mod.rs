use std::any::Any;
use std::path::Path;

use polars_core::prelude::*;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::hashing::HashPartitioner;
use polars_utils::IdxSize;

use crate::hash_keys::HashKeys;

mod row_encoded;

/// A Grouper maps keys to groups, such that duplicate keys map to the same group.
pub trait Grouper: Any + Send + Sync {
    /// Creates a new empty Grouper similar to this one.
    fn new_empty(&self) -> Box<dyn Grouper>;

    /// Reserves space for the given number additional groups.
    fn reserve(&mut self, additional: usize);

    /// Returns the number of groups in this Grouper.
    fn num_groups(&self) -> IdxSize;

    /// Inserts the given keys into this Grouper, mutating groups_idxs such
    /// that group_idxs[i] is the group index of keys[..][i].
    fn insert_keys(&mut self, keys: HashKeys, group_idxs: &mut Vec<IdxSize>);

    /// Adds the given Grouper into this one, mutating groups_idxs such that
    /// the ith group of other now has group index group_idxs[i] in self.
    fn combine(&mut self, other: &dyn Grouper, group_idxs: &mut Vec<IdxSize>);

    /// Adds the given Grouper into this one, mutating groups_idxs such that
    /// the group subset[i] of other now has group index group_idxs[i] in self.
    ///
    /// # Safety
    /// For all i, subset[i] < other.len().
    unsafe fn gather_combine(
        &mut self,
        other: &dyn Grouper,
        subset: &[IdxSize],
        group_idxs: &mut Vec<IdxSize>,
    );

    /// Generate partition indices.
    ///
    /// After this function partitions_idxs[i] will contain the indices for
    /// partition i, and sketches[i] will contain a cardinality sketch for
    /// partition i.
    fn gen_partition_idxs(
        &self,
        partitioner: &HashPartitioner,
        partition_idxs: &mut [Vec<IdxSize>],
        sketches: &mut [CardinalitySketch],
    );

    /// Returns the keys in this Grouper in group order, that is the key for
    /// group i is returned in row i.
    fn get_keys_in_group_order(&self) -> DataFrame;

    /// Stores this Grouper at the given path.
    fn store_ooc(&self, _path: &Path) {
        unimplemented!();
    }

    /// Loads this Grouper from the given path.
    fn load_ooc(&mut self, _path: &Path) {
        unimplemented!();
    }

    fn as_any(&self) -> &dyn Any;
}

pub fn new_hash_grouper(key_schema: Arc<Schema>) -> Box<dyn Grouper> {
    Box::new(row_encoded::RowEncodedHashGrouper::new(key_schema))
}
