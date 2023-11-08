pub use arrow::legacy::index::{IdxArr, IdxSize};
pub use polars_utils::aliases::{InitHashMaps, PlHashMap, PlHashSet, PlIndexMap, PlIndexSet};

use super::*;
use crate::hashing::IdBuildHasher;

/// [ChunkIdx, DfIdx]
pub type ChunkId = [IdxSize; 2];

#[cfg(not(feature = "bigidx"))]
pub type IdxCa = UInt32Chunked;
#[cfg(feature = "bigidx")]
pub type IdxCa = UInt64Chunked;

#[cfg(not(feature = "bigidx"))]
pub const IDX_DTYPE: DataType = DataType::UInt32;
#[cfg(feature = "bigidx")]
pub const IDX_DTYPE: DataType = DataType::UInt64;

#[cfg(not(feature = "bigidx"))]
pub type IdxType = UInt32Type;
#[cfg(feature = "bigidx")]
pub type IdxType = UInt64Type;

/// This hashmap uses an IdHasher
pub type PlIdHashMap<K, V> = hashbrown::HashMap<K, V, IdBuildHasher>;

pub trait InitHashMaps2 {
    type HashMap;

    fn new() -> Self::HashMap;

    fn with_capacity(capacity: usize) -> Self::HashMap;
}

impl<K, V> InitHashMaps2 for PlIdHashMap<K, V> {
    type HashMap = Self;

    fn new() -> Self::HashMap {
        Self::with_capacity_and_hasher(0, Default::default())
    }

    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, Default::default())
    }
}
