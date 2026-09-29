use std::sync::Arc;

use polars_utils::IdxSize;

use super::keys::KeyRowKeys;
use super::layout::{KeyRowLayout, long_slice};
use super::push_long_bytes;

fn append_long_bytes(buffer: &mut Vec<u8>, bytes: &[u8]) -> u32 {
    let offset = buffer.len() as u32;
    buffer.extend_from_slice(bytes);
    offset
}

/// The keys of a hot grouper as rows, each owning the bytes of its long views. A long
/// view of hot key `k` has `buffer_idx == k` and its bytes in `long[k]`.
pub(crate) struct HotKeyRows {
    layout: Arc<KeyRowLayout>,
    hashes: Vec<u64>,
    rows: Vec<u64>,
    long: Vec<Vec<u8>>,
}

impl HotKeyRows {
    pub(crate) fn new(layout: Arc<KeyRowLayout>) -> Self {
        Self {
            layout,
            hashes: Vec::new(),
            rows: Vec::new(),
            long: Vec::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.hashes.len()
    }

    /// Whether hot key `k` is key `i` of `keys`.
    ///
    /// # Safety
    /// `k` and `i` must be in-bounds, and `keys` must have the layout of these keys.
    #[inline(always)]
    pub(crate) unsafe fn eq_key(&self, k: IdxSize, keys: &KeyRowKeys, i: usize) -> bool {
        let (k, stride_words) = (k as usize, self.layout.stride_words);
        let row = self
            .rows
            .get_unchecked(k * stride_words..(k + 1) * stride_words);
        keys.eq_stored(i, row, |view| view.get_external_slice_unchecked(&self.long))
    }

    /// Clears `ok[r]` when key `start + r` of `keys` is not hot key `hot_key_idxs[r]`.
    /// `rows` is scratch space.
    ///
    /// # Safety
    /// The indices must be in-bounds, and `keys` must have the layout of these keys.
    /// The row pointers are only used during this call.
    pub(crate) unsafe fn verify(
        &self,
        keys: &KeyRowKeys,
        start: usize,
        hot_key_idxs: &[IdxSize],
        rows: &mut Vec<*const u64>,
        ok: &mut [bool],
    ) {
        let stride_words = self.layout.stride_words;
        rows.clear();
        rows.extend(
            hot_key_idxs
                .iter()
                .map(|k| self.rows.as_ptr().add(*k as usize * stride_words)),
        );
        keys.verify(start..start + hot_key_idxs.len(), rows, ok, |view| {
            view.get_external_slice_unchecked(&self.long)
        });
    }

    /// Adds key `i` of `keys`, returning its index.
    ///
    /// # Safety
    /// `i` must be in-bounds, and `keys` must have the layout of these keys.
    #[inline(always)]
    pub(crate) unsafe fn push(&mut self, keys: &KeyRowKeys, i: usize) -> IdxSize {
        let k = self.hashes.len();
        self.hashes.push(0);
        self.rows
            .resize(self.rows.len() + self.layout.stride_words, 0);
        if self.layout.has_views() {
            self.long.push(Vec::new());
        }
        self.write(k, keys, i);
        k as IdxSize
    }

    /// Replaces hot key `k` by key `i` of `keys`.
    ///
    /// # Safety
    /// `k` and `i` must be in-bounds, and `keys` must have the layout of these keys.
    #[inline(always)]
    pub(crate) unsafe fn replace(&mut self, k: IdxSize, keys: &KeyRowKeys, i: usize) {
        let (k, stride_words) = (k as usize, self.layout.stride_words);
        self.rows
            .get_unchecked_mut(k * stride_words..(k + 1) * stride_words)
            .fill(0);
        if self.layout.has_views() {
            self.long.get_unchecked_mut(k).clear();
        }
        self.write(k, keys, i);
    }

    unsafe fn write(&mut self, k: usize, keys: &KeyRowKeys, i: usize) {
        let stride_words = self.layout.stride_words;
        *self.hashes.get_unchecked_mut(k) = keys.hashes.value_unchecked(i);
        let row = self
            .rows
            .get_unchecked_mut(k * stride_words..(k + 1) * stride_words);
        if self.layout.has_views() {
            let long = self.long.get_unchecked_mut(k);
            keys.write_row(i, row, |bytes| (k as u32, append_long_bytes(long, bytes)));
        } else {
            keys.write_row(i, row, |_| unreachable!());
        }
    }

    /// Adds hot key `k` to `collector`.
    ///
    /// # Safety
    /// `k` must be in-bounds, and `collector` must have the layout of these keys.
    pub(crate) unsafe fn collect(&self, k: IdxSize, collector: &mut KeyRowCollector) {
        debug_assert!(Arc::ptr_eq(&self.layout, &collector.layout));
        let (k, stride_words) = (k as usize, self.layout.stride_words);
        let row = self
            .rows
            .get_unchecked(k * stride_words..(k + 1) * stride_words);
        let long = self.long.get(k).map_or(&[][..], |l| l.as_slice());
        collector.push(*self.hashes.get_unchecked(k), row, long);
    }

    /// Returns all hot keys, in key order.
    pub(crate) fn keys(&self) -> KeyRowKeys {
        let mut collector = KeyRowCollector::new(self.layout.clone());
        for k in 0..self.len() {
            unsafe { self.collect(k as IdxSize, &mut collector) };
        }
        collector.take()
    }
}

/// Collects key rows, re-pointing their long views into its own buffers.
pub(crate) struct KeyRowCollector {
    layout: Arc<KeyRowLayout>,
    hashes: Vec<u64>,
    rows: Vec<u64>,
    buffers: Vec<Vec<u8>>,
}

impl KeyRowCollector {
    pub(crate) fn new(layout: Arc<KeyRowLayout>) -> Self {
        Self {
            layout,
            hashes: Vec::new(),
            rows: Vec::new(),
            buffers: Vec::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.hashes.len()
    }

    /// # Safety
    /// `row` must be a row of this layout whose long views are at their offset in
    /// `long`.
    unsafe fn push(&mut self, hash: u64, row: &[u64], long: &[u8]) {
        self.hashes.push(hash);
        let start = self.rows.len();
        self.rows.extend_from_slice(row);
        let buffers = &mut self.buffers;
        self.layout.repoint_long_views(
            &mut self.rows[start..],
            |view| long_slice(long, view),
            |b| push_long_bytes(buffers, b),
        );
    }

    /// Returns the collected keys, leaving this collector empty.
    pub(crate) fn take(&mut self) -> KeyRowKeys {
        KeyRowKeys::from_rows(
            self.layout.clone(),
            std::mem::take(&mut self.hashes),
            std::mem::take(&mut self.rows),
            std::mem::take(&mut self.buffers),
        )
    }
}
