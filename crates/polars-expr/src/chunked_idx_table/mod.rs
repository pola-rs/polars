use std::any::Any;

use polars_core::prelude::*;
use polars_utils::index::ChunkId;
use polars_utils::IdxSize;

use crate::hash_keys::HashKeys;

mod row_encoded;


pub trait ChunkedIdxTable: Any + Send + Sync {
    /// Creates a new empty ChunkedIdxTable similar to this one.
    fn new_empty(&self) -> Box<dyn ChunkedIdxTable>;

    /// Reserves space for the given number additional keys.
    fn reserve(&mut self, additional: usize);

    /// Returns the number of unique keys in this ChunkedIdxTable.
    fn num_keys(&self) -> IdxSize;

    /// Inserts the given key chunk into this ChunkedIdxTable.
    fn insert_key_chunk(&mut self, keys: HashKeys);
    
    /// Probe the table, updating table_match and probe_match with
    /// (ChunkId, IdxSize) pairs for each match. Will stop processing new keys
    /// once limit matches have been generated, returning the number of keys processed.
    fn probe(&self, hash_keys: &HashKeys, table_match: &mut Vec<ChunkId>, probe_match: &mut Vec<IdxSize>, mark_matches: bool, limit: usize) -> usize;
    
    /// Get the ChunkIds for each key which was never marked during probing.
    fn unmarked_keys(&self, out: &mut Vec<ChunkId>);
}

pub fn new_chunked_idx_table(key_schema: Arc<Schema>) -> Box<dyn ChunkedIdxTable> {
    // Box::new(row_encoded::BytesIndexMap::new(key_schema))
    todo!()
}