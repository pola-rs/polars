use crate::bitmap::{Bitmap, MutableBitmap};
use crate::storage::SharedStorage;

/// Used to build bitmaps bool-by-bool in sequential order.
#[derive(Default, Clone)]
pub struct BitmapBuilder {
    buf: u64,
    len: usize,
    cap: usize,
    set_bits: usize,
    bytes: Vec<u8>,
}

impl BitmapBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn with_capacity(bits: usize) -> Self {
        let bytes = Vec::with_capacity(bits.div_ceil(64) * 8);
        let words_available = bytes.capacity() / 8;
        Self {
            buf: 0,
            len: 0,
            cap: words_available * 64,
            set_bits: 0,
            bytes,
        }
    }

    #[inline(always)]
    pub fn reserve(&mut self, additional: usize) {
        if self.len + additional > self.cap {
            self.reserve_slow(additional)
        }
    }

    #[cold]
    #[inline(never)]
    fn reserve_slow(&mut self, additional: usize) {
        let bytes_needed = (self.len + additional).div_ceil(64) * 8;
        self.bytes.reserve(bytes_needed - self.bytes.capacity());
        let words_available = self.bytes.capacity() / 8;
        self.cap = words_available * 64;
    }

    #[inline(always)]
    pub fn push(&mut self, x: bool) {
        self.reserve(1);
        unsafe { self.push_unchecked(x) }
    }

    /// # Safety
    /// self.len() < self.capacity() must hold.
    #[inline(always)]
    pub unsafe fn push_unchecked(&mut self, x: bool) {
        debug_assert!(self.len < self.cap);
        self.buf |= (x as u64) << (self.len % 64);
        self.len += 1;
        if self.len % 64 == 0 {
            let p = self.bytes.as_mut_ptr().add(self.bytes.len()).cast::<u64>();
            p.write_unaligned(self.buf.to_le());
            self.bytes.set_len(self.bytes.len() + 8);
            self.set_bits += self.buf.count_ones() as usize;
            self.buf = 0;
        }
    }

    /// # Safety
    /// May only be called once at the end.
    unsafe fn finish(&mut self) {
        if self.len % 64 != 0 {
            self.bytes.extend_from_slice(&self.buf.to_le_bytes());
            self.set_bits += self.buf.count_ones() as usize;
        }
    }

    pub fn into_mut(mut self) -> MutableBitmap {
        unsafe {
            self.finish();
            MutableBitmap::from_vec(self.bytes, self.len)
        }
    }

    pub fn freeze(mut self) -> Bitmap {
        unsafe {
            self.finish();
            let storage = SharedStorage::from_vec(self.bytes);
            Bitmap::from_inner_unchecked(storage, 0, self.len, Some(self.len - self.set_bits))
        }
    }
}
