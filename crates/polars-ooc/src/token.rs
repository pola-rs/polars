use polars_core::prelude::DataFrame;

use crate::memory_manager::{DfKey, mm};

/// Handle to a [`DataFrame`] stored in the [`MemoryManager`](crate::MemoryManager).
///
/// Dropping the token releases the slot and its memory accounting. If the
/// frame was spilled to disk the spill file is deleted.
#[derive(Debug, Eq, PartialEq, Hash)]
pub struct Token {
    thread_idx: u64,
    pub(crate) key: DfKey,
}

const _: () = assert!(size_of::<Token>() == 16);

impl Drop for Token {
    fn drop(&mut self) {
        mm().drop_token(self);
    }
}

impl Token {
    #[inline]
    pub(crate) fn new(thread_idx: u64, key: DfKey) -> Self {
        Self { thread_idx, key }
    }

    #[inline]
    pub(crate) fn thread_idx(&self) -> u64 {
        self.thread_idx
    }

    /// Return the row count of the stored [`DataFrame`].
    pub fn height(&self) -> usize {
        mm().height(self)
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
