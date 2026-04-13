use polars_core::prelude::DataFrame;

use super::memory_manager::mm;

/// Ownerless slot identity — `(index, generation)` without the `Drop`
/// that frees the slot. Used in the spill-tracking stack as a weak
/// lookup key; stale entries are harmlessly skipped via generation checks.
#[derive(Clone, Copy)]
pub(crate) struct SlotId {
    pub(crate) index: u32,
    pub(crate) generation: u32,
}

/// Handle to a [`DataFrame`] stored in the [`MemoryManager`](crate::MemoryManager).
///
/// 8 bytes: `(index: u32, generation: u32)`. The index locates the slot in the
/// [`DataFrameStore`](crate::df_store::DataFrameStore); the generation prevents ABA reuse.
///
/// Dropping the token releases the slot and its memory accounting. If the
/// frame was spilled to disk the spill file is deleted.
#[derive(Debug, Eq, PartialEq, Hash)]
pub struct Token {
    index: u32,
    generation: u32,
}

const _: () = assert!(size_of::<Token>() == 8);

impl Drop for Token {
    fn drop(&mut self) {
        mm().drop_token(self);
    }
}

impl Token {
    #[inline]
    pub(crate) fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Ownerless copy of this token's slot identity for spill tracking.
    #[inline]
    pub(crate) fn id(&self) -> SlotId {
        SlotId {
            index: self.index,
            generation: self.generation,
        }
    }

    /// Return the row count of the stored [`DataFrame`].
    pub fn height(&self) -> usize {
        mm().height(self)
    }

    /// Pin the entry so it has lower priority during spill collection.
    /// Unpinned entries are always spilled first; pinned entries are only
    /// spilled if freeing unpinned entries alone is not enough.
    pub fn pin(&self) -> &Self {
        mm().pin(self);
        self
    }

    /// Unpin the entry so it has higher priority during spill collection.
    pub fn unpin(&self) -> &Self {
        mm().unpin(self);
        self
    }

    /// Clone the stored [`DataFrame`] without consuming the token.
    pub async fn df(&self) -> DataFrame {
        mm().df(self).await
    }

    /// Take the stored [`DataFrame`], consuming the token.
    pub async fn into_df(self) -> DataFrame {
        mm().take_df(self).await
    }
}
