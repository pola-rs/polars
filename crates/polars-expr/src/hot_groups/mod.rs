use std::any::Any;

use polars_core::prelude::*;
use polars_utils::IdxSize;

use crate::EvictIdx;
use crate::hash_keys::HashKeys;

mod binview;
mod fixed_index_table;
mod row_encoded;
mod single_key;

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
    if key_schema.len() > 1 {
        Box::new(row_encoded::RowEncodedHashHotGrouper::new(
            key_schema, num_groups,
        ))
    } else {
        use single_key::SingleKeyHashHotGrouper as SK;
        let dt = key_schema.get_at_index(0).unwrap().1.clone();
        let ng = num_groups;
        match dt {
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => Box::new(SK::<UInt8Type>::new(dt, ng)),
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => Box::new(SK::<UInt16Type>::new(dt, ng)),
            DataType::UInt32 => Box::new(SK::<UInt32Type>::new(dt, ng)),
            DataType::UInt64 => Box::new(SK::<UInt64Type>::new(dt, ng)),
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => Box::new(SK::<Int8Type>::new(dt, ng)),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => Box::new(SK::<Int16Type>::new(dt, ng)),
            DataType::Int32 => Box::new(SK::<Int32Type>::new(dt, ng)),
            DataType::Int64 => Box::new(SK::<Int64Type>::new(dt, ng)),
            #[cfg(feature = "dtype-i128")]
            DataType::Int128 => Box::new(SK::<Int128Type>::new(dt, ng)),
            DataType::Float32 => Box::new(SK::<Float32Type>::new(dt, ng)),
            DataType::Float64 => Box::new(SK::<Float64Type>::new(dt, ng)),

            #[cfg(feature = "dtype-date")]
            DataType::Date => Box::new(SK::<Int32Type>::new(dt, ng)),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => Box::new(SK::<Int64Type>::new(dt, ng)),
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(_) => Box::new(SK::<Int64Type>::new(dt, ng)),
            #[cfg(feature = "dtype-time")]
            DataType::Time => Box::new(SK::<Int64Type>::new(dt, ng)),

            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(_, _) => Box::new(SK::<Int128Type>::new(dt, ng)),
            #[cfg(feature = "dtype-categorical")]
            DataType::Enum(_, _) => Box::new(SK::<UInt32Type>::new(dt, ng)),

            DataType::String | DataType::Binary => {
                Box::new(binview::BinviewHashHotGrouper::new(ng))
            },

            _ => Box::new(row_encoded::RowEncodedHashHotGrouper::new(
                key_schema, num_groups,
            )),
        }
    }
}
