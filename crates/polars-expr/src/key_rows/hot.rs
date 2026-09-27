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

/// The keys of a hot grouper as rows, each owning the bytes of its long views.
pub struct HotKeyRows {
    layout: Arc<KeyRowLayout>,
    hashes: Vec<u64>,
    rows: Vec<u64>,
    long: Vec<Vec<u8>>,
}

impl HotKeyRows {
    pub fn new(layout: Arc<KeyRowLayout>) -> Self {
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

    /// # Safety
    /// `k` must be in-bounds.
    #[inline(always)]
    pub unsafe fn hash(&self, k: IdxSize) -> u64 {
        *self.hashes.get_unchecked(k as usize)
    }

    /// Whether hot key `k` is key `i` of `keys`.
    ///
    /// # Safety
    /// `k` and `i` must be in-bounds, and `keys` must have the layout of these keys.
    #[inline(always)]
    pub unsafe fn eq_key(&self, k: IdxSize, keys: &KeyRowKeys, i: usize) -> bool {
        let (k, stride) = (k as usize, self.layout.stride);
        let row = self.rows.get_unchecked(k * stride..(k + 1) * stride);
        keys.eq_stored(i, row, |view| view.get_external_slice_unchecked(&self.long))
    }

    /// Clears `ok[r]` when key `idxs[r]` of `keys` is not hot key `ks[r]`.
    ///
    /// # Safety
    /// The indices must be in-bounds, and `keys` must have the layout of these keys.
    pub unsafe fn verify(
        &self,
        keys: &KeyRowKeys,
        idxs: &[IdxSize],
        ks: &[IdxSize],
        rows: &mut Vec<*const u64>,
        ok: &mut [bool],
    ) {
        let stride = self.layout.stride;
        rows.clear();
        rows.extend(
            ks.iter()
                .map(|k| self.rows.as_ptr().add(*k as usize * stride)),
        );
        keys.verify(idxs, rows, ok, |view| {
            view.get_external_slice_unchecked(&self.long)
        });
    }

    /// Adds key `i` of `keys`, returning its index.
    ///
    /// # Safety
    /// `i` must be in-bounds, and `keys` must have the layout of these keys.
    #[inline(always)]
    pub unsafe fn push(&mut self, keys: &KeyRowKeys, i: usize) -> IdxSize {
        let k = self.hashes.len();
        self.hashes.push(0);
        self.rows.resize(self.rows.len() + self.layout.stride, 0);
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
    pub unsafe fn replace(&mut self, k: IdxSize, keys: &KeyRowKeys, i: usize) {
        let (k, stride) = (k as usize, self.layout.stride);
        self.rows
            .get_unchecked_mut(k * stride..(k + 1) * stride)
            .fill(0);
        if self.layout.has_views() {
            self.long.get_unchecked_mut(k).clear();
        }
        self.write(k, keys, i);
    }

    unsafe fn write(&mut self, k: usize, keys: &KeyRowKeys, i: usize) {
        let stride = self.layout.stride;
        *self.hashes.get_unchecked_mut(k) = keys.hashes.value_unchecked(i);
        let row = self.rows.get_unchecked_mut(k * stride..(k + 1) * stride);
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
    /// `k` must be in-bounds.
    pub unsafe fn collect(&self, k: IdxSize, collector: &mut KeyRowCollector) {
        let (k, stride) = (k as usize, self.layout.stride);
        let row = self.rows.get_unchecked(k * stride..(k + 1) * stride);
        let long = self.long.get(k).map_or(&[][..], |l| l.as_slice());
        collector.push(&self.layout, *self.hashes.get_unchecked(k), row, long);
    }

    /// Returns all hot keys, in key order.
    pub fn keys(&self) -> KeyRowKeys {
        let mut collector = KeyRowCollector::default();
        for k in 0..self.len() {
            unsafe { self.collect(k as IdxSize, &mut collector) };
        }
        collector.take(self.layout.clone())
    }
}

/// Collects key rows, re-pointing their long views into shared buffers.
#[derive(Default)]
pub struct KeyRowCollector {
    hashes: Vec<u64>,
    rows: Vec<u64>,
    buffers: Vec<Vec<u8>>,
}

impl KeyRowCollector {
    pub(crate) fn len(&self) -> usize {
        self.hashes.len()
    }

    /// # Safety
    /// `row` must be a row of `layout` whose long views are at their offset in `long`.
    unsafe fn push(&mut self, layout: &KeyRowLayout, hash: u64, row: &[u64], long: &[u8]) {
        self.hashes.push(hash);
        let start = self.rows.len();
        self.rows.extend_from_slice(row);
        let buffers = &mut self.buffers;
        layout.repoint_long_views(
            &mut self.rows[start..],
            |view| long_slice(long, view),
            |b| push_long_bytes(buffers, b),
        );
    }

    pub fn take(&mut self, layout: Arc<KeyRowLayout>) -> KeyRowKeys {
        KeyRowKeys::from_rows(
            layout,
            std::mem::take(&mut self.hashes),
            std::mem::take(&mut self.rows),
            std::mem::take(&mut self.buffers),
        )
    }
}
