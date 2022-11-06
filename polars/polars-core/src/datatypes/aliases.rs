use super::*;

#[cfg(not(feature = "bigidx"))]
pub type IdxCa = UInt32Chunked;
#[cfg(feature = "bigidx")]
pub type IdxCa = UInt64Chunked;
pub use polars_arrow::index::{IdxArr, IdxSize};

use crate::vector_hasher::IdBuildHasher;

#[cfg(not(feature = "bigidx"))]
pub const IDX_DTYPE: DataType = DataType::UInt32;
#[cfg(feature = "bigidx")]
pub const IDX_DTYPE: DataType = DataType::UInt64;

#[cfg(not(feature = "bigidx"))]
pub type IdxType = UInt32Type;
#[cfg(feature = "bigidx")]
pub type IdxType = UInt64Type;

pub const NULL_DTYPE: DataType = DataType::Int32;

#[cfg(feature = "private")]
pub type PlHashMap<K, V> = hashbrown::HashMap<K, V, RandomState>;
#[cfg(feature = "private")]

/// This hashmap has the uses an IdHasher
pub type PlIdHashMap<K, V> = hashbrown::HashMap<K, V, IdBuildHasher>;
#[cfg(feature = "private")]
pub type PlHashSet<V> = hashbrown::HashSet<V, RandomState>;
#[cfg(feature = "private")]
pub type PlIndexMap<K, V> = indexmap::IndexMap<K, V, RandomState>;
#[cfg(feature = "private")]
pub type PlIndexSet<K> = indexmap::IndexSet<K, RandomState>;

pub trait InitHashMaps {
    type HashMap;

    fn new() -> Self::HashMap;

    fn with_capacity(capacity: usize) -> Self::HashMap;
}

impl<K, V> InitHashMaps for PlHashMap<K, V> {
    type HashMap = Self;

    fn new() -> Self::HashMap {
        Self::with_capacity_and_hasher(0, Default::default())
    }

    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, Default::default())
    }
}
impl<K> InitHashMaps for PlHashSet<K> {
    type HashMap = Self;

    fn new() -> Self::HashMap {
        Self::with_capacity_and_hasher(0, Default::default())
    }

    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, Default::default())
    }
}

impl<K> InitHashMaps for PlIndexSet<K> {
    type HashMap = Self;

    fn new() -> Self::HashMap {
        Self::with_capacity_and_hasher(0, Default::default())
    }

    fn with_capacity(capacity: usize) -> Self::HashMap {
        Self::with_capacity_and_hasher(capacity, Default::default())
    }
}

impl<K, V> InitHashMaps for PlIndexMap<K, V> {
    type HashMap = Self;

    fn new() -> Self::HashMap {
        Self::with_capacity_and_hasher(0, Default::default())
    }

    fn with_capacity(capacity: usize) -> Self::HashMap {
        Self::with_capacity_and_hasher(capacity, Default::default())
    }
}
impl<K, V> InitHashMaps for PlIdHashMap<K, V> {
    type HashMap = Self;

    fn new() -> Self::HashMap {
        Self::with_capacity_and_hasher(0, Default::default())
    }

    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, Default::default())
    }
}
