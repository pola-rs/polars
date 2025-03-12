use std::any::Any;

use polars_core::prelude::*;
use polars_utils::IdxSize;

use crate::hash_keys::HashKeys;

mod row_encoded;
mod single_key;

pub trait IdxTable: Any + Send + Sync {
    /// Creates a new empty IdxTable similar to this one.
    fn new_empty(&self) -> Box<dyn IdxTable>;

    /// Reserves space for the given number additional keys.
    fn reserve(&mut self, additional: usize);

    /// Returns the number of unique keys in this IdxTable.
    fn num_keys(&self) -> IdxSize;

    /// Inserts the given keys into this IdxTable.
    fn insert_keys(&mut self, keys: &HashKeys, track_unmatchable: bool);

    /// Inserts a subset of the given keys into this IdxTable.
    /// # Safety
    /// The provided subset indices must be in-bounds.
    unsafe fn insert_keys_subset(
        &mut self,
        keys: &HashKeys,
        subset: &[IdxSize],
        track_unmatchable: bool,
    );

    /// Probe the table, adding an entry to table_match and probe_match for each
    /// match. Will stop processing new keys once limit matches have been
    /// generated, returning the number of keys processed.
    ///
    /// If mark_matches is true, matches are marked in the table as such.
    ///
    /// If emit_unmatched is true, for keys that do not have a match we emit a
    /// match with ChunkId::null() on the table match.
    fn probe(
        &self,
        hash_keys: &HashKeys,
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
        mark_matches: bool,
        emit_unmatched: bool,
        limit: IdxSize,
    ) -> IdxSize;

    /// The same as probe, except it will only apply to the specified subset of keys.
    /// # Safety
    /// The provided subset indices must be in-bounds.
    #[allow(clippy::too_many_arguments)]
    unsafe fn probe_subset(
        &self,
        hash_keys: &HashKeys,
        subset: &[IdxSize],
        table_match: &mut Vec<IdxSize>,
        probe_match: &mut Vec<IdxSize>,
        mark_matches: bool,
        emit_unmatched: bool,
        limit: IdxSize,
    ) -> IdxSize;

    /// Get the ChunkIds for each key which was never marked during probing.
    fn unmarked_keys(&self, out: &mut Vec<IdxSize>, offset: IdxSize, limit: IdxSize) -> IdxSize;
}

pub fn new_idx_table(key_schema: Arc<Schema>) -> Box<dyn IdxTable> {
    if key_schema.len() > 1 {
        Box::new(row_encoded::RowEncodedIdxTable::new())
    } else {
        use single_key::SingleKeyIdxTable as SKIT;
        match key_schema.get_at_index(0).unwrap().1 {
            DataType::UInt8 => Box::new(SKIT::<UInt8Type>::new()),
            DataType::UInt16 => Box::new(SKIT::<UInt16Type>::new()),
            DataType::UInt32 => Box::new(SKIT::<UInt32Type>::new()),
            DataType::UInt64 => Box::new(SKIT::<UInt64Type>::new()),
            DataType::Int8 => Box::new(SKIT::<Int8Type>::new()),
            DataType::Int16 => Box::new(SKIT::<Int16Type>::new()),
            DataType::Int32 => Box::new(SKIT::<Int32Type>::new()),
            DataType::Int64 => Box::new(SKIT::<Int64Type>::new()),
            DataType::Int128 => Box::new(SKIT::<Int128Type>::new()),
            DataType::Float32 => Box::new(SKIT::<Float32Type>::new()),
            DataType::Float64 => Box::new(SKIT::<Float64Type>::new()),

            DataType::Date => Box::new(SKIT::<Int32Type>::new()),
            DataType::Datetime(_, _) => Box::new(SKIT::<Int64Type>::new()),
            DataType::Duration(_) => Box::new(SKIT::<Int64Type>::new()),
            DataType::Time => Box::new(SKIT::<Int64Type>::new()),

            DataType::Decimal(_, _) => Box::new(SKIT::<Int128Type>::new()),
            DataType::Enum(_, _) => Box::new(SKIT::<UInt32Type>::new()),

            DataType::String
            | DataType::Binary
            | DataType::Boolean
            | DataType::Null
            | DataType::BinaryOffset
            | DataType::Array(_, _)
            | DataType::List(_)
            | DataType::Object(_)
            | DataType::Categorical(_, _)
            | DataType::Struct(_)
            | DataType::Unknown(_) => Box::new(row_encoded::RowEncodedIdxTable::new()),
        }
    }
}
