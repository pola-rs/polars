use std::any::Any;

use polars_core::prelude::*;
use polars_utils::IdxSize;

use crate::hash_keys::HashKeys;

mod row_encoded;

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

pub fn new_idx_table(_key_schema: Arc<Schema>) -> Box<dyn IdxTable> {
    Box::new(row_encoded::RowEncodedIdxTable::new())
}
