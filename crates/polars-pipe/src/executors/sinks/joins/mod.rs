#[cfg(feature = "cross_join")]
mod cross;
mod generic_build;
mod generic_probe_inner_left;
mod generic_probe_outer;
mod row_values;

use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::atomic::AtomicBool;

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
#[repr(C)]
pub(super) struct Key {
    pub(super) hash: u64,
    /// We use the MSB as tracker for outer join matches
    /// So the 25th bit of the chunk_idx will be used for that.
    idx: ChunkId,
}

impl Key {
    #[inline]
    fn new(hash: u64, chunk_idx: IdxSize, df_idx: IdxSize) -> Self {
        let idx = ChunkId::store(chunk_idx, df_idx);
        Key { hash, idx }
    }
}

impl Hash for Key {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash)
    }
}

pub(crate) trait ExtraPayload: Clone + Sync + Send + Default + 'static {
    /// Tracker used in the outer join.
    fn get_tracker(&self) -> &AtomicBool {
        panic!()
    }
}
impl ExtraPayload for () {}

#[repr(transparent)]
pub(crate) struct Tracker {
    inner: AtomicBool,
}

impl Default for Tracker {
    #[inline]
    fn default() -> Self {
        Self {
            inner: AtomicBool::new(false),
        }
    }
}

// Needed for the trait resolving. We should never hit this.
impl Clone for Tracker {
    fn clone(&self) -> Self {
        panic!()
    }
}

impl ExtraPayload for Tracker {
    #[inline(always)]
    fn get_tracker(&self) -> &AtomicBool {
        &self.inner
    }
}

type PartitionedMap<V> =
    PartitionedHashMap<Key, (UnitVec<ChunkId>, V), BuildHasherDefault<IdHasher>>;
