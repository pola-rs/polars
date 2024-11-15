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
    
    /// Probe the table, returning a ChunkId per key.
    fn probe(&self, keys: &HashKeys, out: &mut Vec<ChunkId>);
    
    /// Get the ChunkIds for each key which was never probed.
    fn unprobed_keys(&self, out: &mut Vec<ChunkId>);
}