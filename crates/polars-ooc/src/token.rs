/// A lightweight token referencing a DataFrame stored in the MemoryManager.
///
/// Tokens are 16 bytes and `Copy`.
/// A token encodes three pieces of information:
/// - `thread_idx` — which per-thread store in the `MemoryManager` array holds the data.
/// - `height` -- an optimization: the number of rows in the DataFrame at store
///   time (stored as `u32`, sufficient for any single morsel or buffered chunk).
///   Storing the height avoids loading the DataFrame from the MM (or from disk
///   if spilled) in many cases where nodes only need the row count to make
///   size-based decisions (e.g. trimming buffers, capacity estimation).
/// - `key_ffi` — the slotmap key (as a u64 FFI value) identifying the entry within that store.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Token {
    thread_idx: u32,
    height: u32,
    key_ffi: u64,
}

impl Token {
    #[inline]
    pub(crate) fn new(thread_idx: u32, height: u32, key_ffi: u64) -> Self {
        Self {
            thread_idx,
            height,
            key_ffi,
        }
    }

    /// The number of rows in the stored DataFrame (captured at store time).
    #[inline]
    pub fn height(self) -> usize {
        self.height as usize
    }

    #[inline]
    pub(crate) fn thread_idx(self) -> u32 {
        self.thread_idx
    }

    #[inline]
    pub(crate) fn key_ffi(self) -> u64 {
        self.key_ffi
    }

    #[inline]
    pub(crate) fn set_height(&mut self, height: usize) {
        self.height = height as u32;
    }
}
