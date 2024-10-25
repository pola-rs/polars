use std::any::Any;
use std::path::Path;

use polars_core::prelude::*;
use polars_utils::aliases::PlRandomState;
use polars_utils::IdxSize;

mod row_encoded;

/// A Grouper maps keys to groups, such that duplicate keys map to the same group.
pub trait Grouper: Any + Send {
    /// Creates a new empty Grouper similar to this one.
    fn new_empty(&self) -> Box<dyn Grouper>;

    /// Returns the number of groups in this Grouper.
    fn num_groups(&self) -> IdxSize;

    /// Inserts the given keys into this Grouper, mutating groups_idxs such
    /// that group_idxs[i] is the group index of keys[..][i].
    fn insert_keys(&mut self, keys: &DataFrame, group_idxs: &mut Vec<IdxSize>);

    /// Adds the given Grouper into this one, mutating groups_idxs such that
    /// the ith group of other now has group index group_idxs[i] in self.
    fn combine(&mut self, other: &dyn Grouper, group_idxs: &mut Vec<IdxSize>);

    /// Partitions this Grouper into the given partitions.
    ///
    /// Updates partition_idxs and group_idxs such that the ith group of self
    /// has group index group_idxs[i] in partition partition_idxs[i].
    ///
    /// It is guaranteed that two equal keys in two independent partition_into
    /// calls map to the same partition index if the seed and the number of
    /// partitions is equal.
    fn partition_into(
        &self,
        seed: u64,
        partitions: &mut [Box<dyn Grouper>],
        partition_idxs: &mut Vec<IdxSize>,
        group_idxs: &mut Vec<IdxSize>,
    );

    /// Returns the keys in this Grouper in group order, that is the key for
    /// group i is returned in row i.
    fn get_keys_in_group_order(&self) -> DataFrame;

    /// Returns the keys in this Grouper, mutating group_idxs such that the ith
    /// key returned corresponds to group group_idxs[i].
    fn get_keys_groups(&self, group_idxs: &mut Vec<IdxSize>) -> DataFrame;

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

pub fn new_hash_grouper(key_schema: Arc<Schema>, random_state: PlRandomState) -> Box<dyn Grouper> {
    Box::new(row_encoded::RowEncodedHashGrouper::new(
        key_schema,
        random_state,
    ))
}
