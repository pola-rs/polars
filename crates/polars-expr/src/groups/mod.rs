use std::any::Any;

use arrow::bitmap::BitmapBuilder;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::IdxSize;
use polars_utils::hashing::HashPartitioner;

use crate::hash_keys::HashKeys;

mod binview;
mod row_encoded;
mod single_key;

/// A Grouper maps keys to groups, such that duplicate keys map to the same group.
pub trait Grouper: Any + Send + Sync {
    /// Creates a new empty Grouper similar to this one.
    fn new_empty(&self) -> Box<dyn Grouper>;

    /// Reserves space for the given number additional groups.
    fn reserve(&mut self, additional: usize);

    /// Returns the number of groups in this Grouper.
    fn num_groups(&self) -> IdxSize;

    /// Inserts the given subset of keys into this Grouper. If groups_idxs is
    /// passed it is extended such with the group index of keys[subset[i]].
    ///
    /// # Safety
    /// The subset indexes must be in-bounds.
    unsafe fn insert_keys_subset(
        &mut self,
        keys: &HashKeys,
        subset: &[IdxSize],
        group_idxs: Option<&mut Vec<IdxSize>>,
    );

    /// Returns the keys in this Grouper in group order, that is the key for
    /// group i is returned in row i.
    fn get_keys_in_group_order(&self, schema: &Schema) -> DataFrame;

    /// Returns the (indices of the) keys found in the groupers. If
    /// invert is true it instead returns the keys not found in the groupers.
    /// # Safety
    /// All groupers must have the same schema.
    unsafe fn probe_partitioned_groupers(
        &self,
        groupers: &[Box<dyn Grouper>],
        keys: &HashKeys,
        partitioner: &HashPartitioner,
        invert: bool,
        probe_matches: &mut Vec<IdxSize>,
    );

    /// Returns for each key if it is found in the groupers. If invert is true
    /// it returns true if it isn't found.
    /// # Safety
    /// All groupers must have the same schema.
    unsafe fn contains_key_partitioned_groupers(
        &self,
        groupers: &[Box<dyn Grouper>],
        keys: &HashKeys,
        partitioner: &HashPartitioner,
        invert: bool,
        contains_key: &mut BitmapBuilder,
    );

    fn as_any(&self) -> &dyn Any;
}

pub fn new_hash_grouper(key_schema: Arc<Schema>) -> Box<dyn Grouper> {
    if key_schema.len() > 1 {
        Box::new(row_encoded::RowEncodedHashGrouper::new())
    } else {
        let (_name, dt) = key_schema.get_at_index(0).unwrap();
        match dt {
            dt if dt.is_primitive_numeric() | dt.is_temporal() => {
                with_match_physical_numeric_polars_type!(dt.to_physical(), |$T| {
                    Box::new(single_key::SingleKeyHashGrouper::<$T>::new())
                })
            },

            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(_, _) => {
                Box::new(single_key::SingleKeyHashGrouper::<Int128Type>::new())
            },
            #[cfg(feature = "dtype-categorical")]
            DataType::Enum(_, _) => Box::new(single_key::SingleKeyHashGrouper::<UInt32Type>::new()),

            DataType::String | DataType::Binary => Box::new(binview::BinviewHashGrouper::new()),

            _ => Box::new(row_encoded::RowEncodedHashGrouper::new()),
        }
    }
}
