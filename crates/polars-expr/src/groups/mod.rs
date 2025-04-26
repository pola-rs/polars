use std::any::Any;

use arrow::bitmap::BitmapBuilder;
use polars_core::prelude::*;
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
    fn get_keys_in_group_order(&self) -> DataFrame;

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
        Box::new(row_encoded::RowEncodedHashGrouper::new(key_schema))
    } else {
        use single_key::SingleKeyHashGrouper as SK;
        let (name, dt) = key_schema.get_at_index(0).unwrap();
        let (name, dt) = (name.clone(), dt.clone());
        match dt {
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => Box::new(SK::<UInt8Type>::new(name, dt)),
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => Box::new(SK::<UInt16Type>::new(name, dt)),
            DataType::UInt32 => Box::new(SK::<UInt32Type>::new(name, dt)),
            DataType::UInt64 => Box::new(SK::<UInt64Type>::new(name, dt)),
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => Box::new(SK::<Int8Type>::new(name, dt)),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => Box::new(SK::<Int16Type>::new(name, dt)),
            DataType::Int32 => Box::new(SK::<Int32Type>::new(name, dt)),
            DataType::Int64 => Box::new(SK::<Int64Type>::new(name, dt)),
            #[cfg(feature = "dtype-i128")]
            DataType::Int128 => Box::new(SK::<Int128Type>::new(name, dt)),
            DataType::Float32 => Box::new(SK::<Float32Type>::new(name, dt)),
            DataType::Float64 => Box::new(SK::<Float64Type>::new(name, dt)),

            #[cfg(feature = "dtype-date")]
            DataType::Date => Box::new(SK::<Int32Type>::new(name, dt)),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => Box::new(SK::<Int64Type>::new(name, dt)),
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(_) => Box::new(SK::<Int64Type>::new(name, dt)),
            #[cfg(feature = "dtype-time")]
            DataType::Time => Box::new(SK::<Int64Type>::new(name, dt)),

            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(_, _) => Box::new(SK::<Int128Type>::new(name, dt)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Enum(_, _) => Box::new(SK::<UInt32Type>::new(name, dt)),

            DataType::String | DataType::Binary => {
                Box::new(binview::BinviewHashGrouper::new(name, dt))
            },

            _ => Box::new(row_encoded::RowEncodedHashGrouper::new(key_schema)),
        }
    }
}
