use polars_utils::slice::load_padded_le_u64;

use super::bitmask::BitMask;
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::storage::SharedStorage;
use crate::trusted_len::TrustedLen;

/// Used to build bitmaps bool-by-bool in sequential order.
#[derive(Default, Clone)]
pub struct BitmapBuilder {
    buf: u64,                 // A buffer containing the last self.bit_len % 64 bits.
    bit_len: usize,           // Length in bits.
    bit_cap: usize,           // Capacity in bits (always multiple of 64).
    set_bits_in_bytes: usize, // The number of bits set in self.bytes, not including self.buf.
    bytes: Vec<u8>,
}

impl BitmapBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.bit_len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.bit_len == 0
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.bit_cap
    }

    #[inline(always)]
    pub fn set_bits(&self) -> usize {
        self.set_bits_in_bytes + self.buf.count_ones() as usize
    }

    #[inline(always)]
    pub fn unset_bits(&self) -> usize {
        self.bit_len - self.set_bits()
    }

    pub fn with_capacity(bits: usize) -> Self {
        let bytes = Vec::with_capacity(bits.div_ceil(64) * 8);
        let words_available = bytes.capacity() / 8;
        Self {
            buf: 0,
            bit_len: 0,
            bit_cap: words_available * 64,
            set_bits_in_bytes: 0,
            bytes,
        }
    }

    #[inline(always)]
    pub fn reserve(&mut self, additional: usize) {
        if self.bit_len + additional > self.bit_cap {
            self.reserve_slow(additional)
        }
    }

    #[cold]
    #[inline(never)]
    fn reserve_slow(&mut self, additional: usize) {
        let bytes_needed = (self.bit_len + additional).div_ceil(64) * 8;
        self.bytes.reserve(bytes_needed - self.bytes.len());
        let words_available = self.bytes.capacity() / 8;
        self.bit_cap = words_available * 64;
    }

    #[inline(always)]
    pub fn push(&mut self, x: bool) {
        self.reserve(1);
        unsafe { self.push_unchecked(x) }
    }

    /// Does not update len/set_bits, simply writes to the output buffer.
    /// # Safety
    /// self.bytes.len() + 8 <= self.bytes.capacity() must hold.
    #[inline(always)]
    unsafe fn flush_word_unchecked(&mut self, w: u64) {
        let cur_len = self.bytes.len();
        let p = self.bytes.as_mut_ptr().add(cur_len).cast::<u64>();
        p.write_unaligned(w.to_le());
        self.bytes.set_len(cur_len + 8);
    }

    /// # Safety
    /// self.len() < self.capacity() must hold.
    #[inline(always)]
    pub unsafe fn push_unchecked(&mut self, x: bool) {
        debug_assert!(self.bit_len < self.bit_cap);
        self.buf |= (x as u64) << (self.bit_len % 64);
        self.bit_len += 1;
        if self.bit_len % 64 == 0 {
            self.flush_word_unchecked(self.buf);
            self.set_bits_in_bytes += self.buf.count_ones() as usize;
            self.buf = 0;
        }
    }

    #[inline(always)]
    pub fn extend_constant(&mut self, length: usize, value: bool) {
        // Fast path if the extension still fits in buf with room left to spare.
        let bits_in_buf = self.bit_len % 64;
        if bits_in_buf + length < 64 {
            let bit_block = ((value as u64) << length) - (value as u64);
            self.buf |= bit_block << bits_in_buf;
            self.bit_len += length;
        } else {
            self.extend_constant_slow(length, value);
        }
    }

    #[cold]
    fn extend_constant_slow(&mut self, length: usize, value: bool) {
        unsafe {
            let value_spread = if value { u64::MAX } else { 0 }; // Branchless neg.

            // Extend and flush current buf.
            self.reserve(length);
            let bits_in_buf = self.bit_len % 64;
            let ext_buf = self.buf | (value_spread << bits_in_buf);
            self.flush_word_unchecked(ext_buf);
            self.set_bits_in_bytes += ext_buf.count_ones() as usize;

            // Write complete words.
            let remaining_bits = length - (64 - bits_in_buf);
            let remaining_words = remaining_bits / 64;
            for _ in 0..remaining_words {
                self.flush_word_unchecked(value_spread);
            }
            self.set_bits_in_bytes += (remaining_words * 64) & value_spread as usize;

            // Put remainder in buf and update length.
            self.buf = ((value as u64) << (remaining_bits % 64)) - (value as u64);
            self.bit_len += length;
        }
    }

    /// Pushes the first length bits from the given word, assuming the rest of
    /// the bits are zero.
    /// # Safety
    /// self.len + length <= self.cap and length <= 64 must hold.
    pub unsafe fn push_word_with_len_unchecked(&mut self, word: u64, length: usize) {
        debug_assert!(self.bit_len + length <= self.bit_cap);
        debug_assert!(length <= 64);
        debug_assert!(length == 64 || (word >> length) == 0);
        let bits_in_buf = self.bit_len % 64;
        self.buf |= word << bits_in_buf;
        if bits_in_buf + length >= 64 {
            self.flush_word_unchecked(self.buf);
            self.set_bits_in_bytes += self.buf.count_ones() as usize;
            self.buf = if bits_in_buf > 0 {
                word >> (64 - bits_in_buf)
            } else {
                0
            };
        }
        self.bit_len += length;
    }

    /// # Safety
    /// self.len() + length <= self.capacity() must hold, as well as
    /// offset + length <= 8 * slice.len().
    unsafe fn extend_from_slice_unchecked(
        &mut self,
        mut slice: &[u8],
        mut offset: usize,
        mut length: usize,
    ) {
        if length == 0 {
            return;
        }

        // Deal with slice offset so it's aligned to bytes.
        let slice_bit_offset = offset % 8;
        if slice_bit_offset > 0 {
            let bits_in_first_byte = (8 - slice_bit_offset).min(length);
            let first_byte = *slice.get_unchecked(offset / 8) >> slice_bit_offset;
            self.push_word_with_len_unchecked(
                first_byte as u64 & ((1 << bits_in_first_byte) - 1),
                bits_in_first_byte,
            );
            length -= bits_in_first_byte;
            offset += bits_in_first_byte;
        }
        slice = slice.get_unchecked(offset / 8..);

        // Write word-by-word.
        let bits_in_buf = self.bit_len % 64;
        if bits_in_buf > 0 {
            while length >= 64 {
                let word = u64::from_le_bytes(slice.get_unchecked(0..8).try_into().unwrap());
                self.buf |= word << bits_in_buf;
                self.flush_word_unchecked(self.buf);
                self.set_bits_in_bytes += self.buf.count_ones() as usize;
                self.buf = word >> (64 - bits_in_buf);
                self.bit_len += 64;
                length -= 64;
                slice = slice.get_unchecked(8..);
            }
        } else {
            while length >= 64 {
                let word = u64::from_le_bytes(slice.get_unchecked(0..8).try_into().unwrap());
                self.flush_word_unchecked(word);
                self.set_bits_in_bytes += word.count_ones() as usize;
                self.bit_len += 64;
                length -= 64;
                slice = slice.get_unchecked(8..);
            }
        }

        // Just the last word left.
        if length > 0 {
            let word = load_padded_le_u64(slice);
            self.push_word_with_len_unchecked(word & ((1 << length) - 1), length);
        }
    }

    pub fn extend_from_slice(&mut self, slice: &[u8], offset: usize, length: usize) {
        assert!(8 * slice.len() >= offset + length);
        self.reserve(length);
        unsafe {
            self.extend_from_slice_unchecked(slice, offset, length);
        }
    }

    pub fn extend_from_bitmap(&mut self, bitmap: &Bitmap) {
        // TODO: we can perhaps use the bitmaps bitcount here instead of
        // recomputing it if it has a known bitcount.
        let (slice, offset, length) = bitmap.as_slice();
        self.extend_from_slice(slice, offset, length);
    }

    pub fn extend_from_bitmask(&mut self, bitmap: BitMask<'_>) {
        let (slice, offset, length) = bitmap.inner();
        self.extend_from_slice(slice, offset, length);
    }

    /// # Safety
    /// May only be called once at the end.
    unsafe fn finish(&mut self) {
        if self.bit_len % 64 != 0 {
            self.bytes.extend_from_slice(&self.buf.to_le_bytes());
            self.set_bits_in_bytes += self.buf.count_ones() as usize;
            self.buf = 0;
        }
    }

    /// Converts this BitmapBuilder into a mutable bitmap.
    pub fn into_mut(mut self) -> MutableBitmap {
        unsafe {
            self.finish();
            MutableBitmap::from_vec(self.bytes, self.bit_len)
        }
    }

    /// The same as into_mut, but returns None if the bitmap is all-ones.
    pub fn into_opt_mut_validity(mut self) -> Option<MutableBitmap> {
        unsafe {
            self.finish();
            if self.set_bits_in_bytes == self.bit_len {
                return None;
            }
            Some(MutableBitmap::from_vec(self.bytes, self.bit_len))
        }
    }

    /// Freezes this BitmapBuilder into an immutable Bitmap.
    pub fn freeze(mut self) -> Bitmap {
        unsafe {
            self.finish();
            let storage = SharedStorage::from_vec(self.bytes);
            Bitmap::from_inner_unchecked(
                storage,
                0,
                self.bit_len,
                Some(self.bit_len - self.set_bits_in_bytes),
            )
        }
    }

    /// The same as freeze, but returns None if the bitmap is all-ones.
    pub fn into_opt_validity(mut self) -> Option<Bitmap> {
        unsafe {
            self.finish();
            if self.set_bits_in_bytes == self.bit_len {
                return None;
            }
            let storage = SharedStorage::from_vec(self.bytes);
            let bitmap = Bitmap::from_inner_unchecked(
                storage,
                0,
                self.bit_len,
                Some(self.bit_len - self.set_bits_in_bytes),
            );
            Some(bitmap)
        }
    }

    pub fn extend_trusted_len_iter<I>(&mut self, iterator: I)
    where
        I: Iterator<Item = bool> + TrustedLen,
    {
        self.reserve(iterator.size_hint().1.unwrap());
        for b in iterator {
            // SAFETY: we reserved and the iterator's length is trusted.
            unsafe {
                self.push_unchecked(b);
            }
        }
    }

    #[inline]
    pub fn from_trusted_len_iter<I>(iterator: I) -> Self
    where
        I: Iterator<Item = bool> + TrustedLen,
    {
        let mut builder = Self::new();
        builder.extend_trusted_len_iter(iterator);
        builder
    }
}
