#[cfg(feature = "cross_join")]
mod cross;
mod generic_build;
mod generic_probe_inner_left;
mod row_values;

use std::hash::{BuildHasherDefault, Hash, Hasher};

#[cfg(feature = "cross_join")]
pub(crate) use cross::*;
pub(crate) use generic_build::GenericBuild;
use polars_core::hashing::IdHasher;
use polars_core::prelude::IdxSize;
use polars_ops::prelude::JoinType;
use polars_utils::idx_vec::UnitVec;
use polars_utils::index::ChunkId;
use polars_utils::partitioned::PartitionedHashMap;

trait ToRow {
    fn get_row(&self) -> &[u8];
}

impl ToRow for &[u8] {
    #[inline(always)]
    fn get_row(&self) -> &[u8] {
        self
    }
}

impl ToRow for Option<&[u8]> {
    #[inline(always)]
    fn get_row(&self) -> &[u8] {
        self.unwrap()
    }
}

// This is the hash and the Index offset in the chunks and the index offset in the dataframe
#[derive(Copy, Clone, Debug)]
pub(super) struct Key {
    pub(super) hash: u64,
    chunk_idx: IdxSize,
    df_idx: IdxSize,
}

impl Key {
    #[inline]
    fn new(hash: u64, chunk_idx: IdxSize, df_idx: IdxSize) -> Self {
        Key {
            hash,
            chunk_idx,
            df_idx,
        }
    }
}

impl Hash for Key {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash)
    }
}

type PartitionedMap = PartitionedHashMap<Key, UnitVec<ChunkId>, BuildHasherDefault<IdHasher>>;
