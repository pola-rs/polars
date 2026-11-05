use polars_utils::slice::load_padded_le_u64;

use super::Bitmap;
use crate::trusted_len::TrustedLen;

/// Calculates how many iterations are remaining, assuming:
///  - We have length elements left.
///  - We need max(consume, min_length_for_iter) elements to start a new iteration.
///  - On each iteration we consume the given amount of elements.
fn calc_iters_remaining(length: usize, min_length_for_iter: usize, consume: usize) -> usize {
    let min_length_for_iter = min_length_for_iter.max(consume);
    if length < min_length_for_iter {
        return 0;
    }

    let obvious_part = length - min_length_for_iter;
    let obvious_iters = obvious_part / consume;
    // let obvious_part_remaining = obvious_part % consume;
    // let total_remaining = min_length_for_iter + obvious_part_remaining;
    // assert!(total_remaining >= min_length_for_iter);          // We have at least 1 more iter.
    // assert!(obvious_part_remaining < consume);                // Basic modulo property.
    // assert!(total_remaining < min_length_for_iter + consume); // Add min_length_for_iter to both sides.
    // assert!(total_remaining - consume < min_length_for_iter); // Not enough remaining after 1 iter.
    1 + obvious_iters // Thus always exactly 1 more iter.
}

/// Iterator over the indices of the set bits.
pub type TrueIdxIter<'a> = SetBitIdxIter<'a, false>;

/// Iterator over the indices of the set bits, with an exact length.
pub type ExactTrueIdxIter<'a> = SetBitIdxIter<'a, true>;

/// Iterator over the indices of the set bits.
///
/// Consecutive words with all bits set are iterated as a plain range, so dense
/// stretches cost about as much as counting.
///
/// If `EXACT` is true the number of set bits is known, making the size hint
/// exact.
#[derive(Clone)]
pub struct SetBitIdxIter<'a, const EXACT: bool> {
    /// Starts at the current word.
    bytes: &'a [u8],
    /// Number of bits from the start of the current word to the end.
    bits_left: usize,
    /// Set bits of the current word not yet returned.
    word: u64,
    /// Output index of bit 0 of the current word. Wraps below zero for the
    /// first word if the offset is not byte-aligned.
    word_start: usize,
    /// Range of indices of set bits before the current word.
    run_pos: usize,
    run_end: usize,
    /// Number of indices not yet returned. Only maintained if `EXACT`.
    remaining: usize,
}

impl<'a> SetBitIdxIter<'a, false> {
    pub fn new(bytes: &'a [u8], offset: usize, len: usize) -> Self {
        Self::new_impl(bytes, offset, len, 0)
    }
}

impl<'a> SetBitIdxIter<'a, true> {
    /// # Safety
    /// `set_count` must be the number of set bits in the iterated range.
    pub unsafe fn new_with_set_count(
        bytes: &'a [u8],
        offset: usize,
        len: usize,
        set_count: usize,
    ) -> Self {
        Self::new_impl(bytes, offset, len, set_count)
    }

    /// Iterates over the indices of the valid entries, or `0..len` if there
    /// is no validity.
    #[inline]
    pub fn from_validity(len: usize, validity: Option<&'a Bitmap>) -> Self {
        let Some(validity) = validity else {
            return Self::all_set(len);
        };
        assert_eq!(len, validity.len());
        let (bytes, offset, len) = validity.as_slice();
        // SAFETY: the count is taken from the same bitmap.
        unsafe { Self::new_with_set_count(bytes, offset, len, validity.set_bits()) }
    }

    /// Iterates over `0..len`, as if all bits are set.
    #[inline]
    pub fn all_set(len: usize) -> Self {
        Self {
            bytes: &[],
            bits_left: 0,
            word: 0,
            word_start: 0,
            run_pos: 0,
            run_end: len,
            remaining: len,
        }
    }
}

impl<'a, const EXACT: bool> SetBitIdxIter<'a, EXACT> {
    fn new_impl(bytes: &'a [u8], offset: usize, len: usize, set_count: usize) -> Self {
        assert!(bytes.len() * 8 >= offset + len);
        let bytes = &bytes[offset / 8..];
        let bit_offset = offset % 8;
        let bits_left = bit_offset + len;
        let mut iter = Self {
            bytes,
            bits_left,
            word: load_word_to_end(bytes, bits_left) & (u64::MAX << bit_offset),
            word_start: 0usize.wrapping_sub(bit_offset),
            run_pos: 0,
            run_end: 0,
            remaining: set_count,
        };
        if iter.word == u64::MAX {
            iter.start_run();
        }
        iter
    }

    /// Moves to the next word, returns false if there is none.
    #[inline(always)]
    fn advance(&mut self) -> bool {
        if self.bits_left <= 64 {
            return false;
        }
        self.bytes = &self.bytes[8..];
        self.bits_left -= 64;
        self.word_start = self.word_start.wrapping_add(64);
        self.word = load_word_to_end(self.bytes, self.bits_left);
        true
    }

    /// Starts a run at the current word, which must be full, extending it over
    /// all directly following full words.
    #[inline(always)]
    fn start_run(&mut self) {
        self.run_pos = self.word_start;
        self.run_end = self.word_start;
        loop {
            self.run_end += 64;
            if !self.advance() {
                self.word = 0;
                return;
            }
            if self.word != u64::MAX {
                return;
            }
        }
    }
}

/// Loads the next 64 bits, with the bits at or after `bits_left` cleared.
#[inline(always)]
fn load_word_to_end(bytes: &[u8], bits_left: usize) -> u64 {
    let word = load_padded_le_u64(bytes);
    if bits_left < 64 {
        word & ((1 << bits_left) - 1)
    } else {
        word
    }
}

impl<const EXACT: bool> Iterator for SetBitIdxIter<'_, EXACT> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.run_pos < self.run_end {
                let idx = self.run_pos;
                self.run_pos += 1;
                if EXACT {
                    self.remaining -= 1;
                }
                return Some(idx);
            }
            if self.word != 0 {
                let idx = self
                    .word_start
                    .wrapping_add(self.word.trailing_zeros() as usize);
                self.word &= self.word - 1;
                if EXACT {
                    self.remaining -= 1;
                }
                return Some(idx);
            }
            // Checking `loaded` after starting the run gives notably faster
            // code for dense bitmaps.
            let loaded = self.advance();
            if self.word == u64::MAX {
                self.start_run();
            }
            if !loaded {
                return None;
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if EXACT {
            return (self.remaining, Some(self.remaining));
        }
        let in_hand = (self.run_end - self.run_pos) + self.word.count_ones() as usize;
        (in_hand, Some(in_hand + self.bits_left.saturating_sub(64)))
    }

    // All-ones fast path in fold, hoists branch out of inner loop. Sum,
    // for_each, etc, dispatch to fold.
    #[inline]
    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;
        for idx in self.run_pos..self.run_end {
            acc = f(acc, idx);
        }
        loop {
            if self.word == u64::MAX {
                for i in 0..64 {
                    acc = f(acc, self.word_start + i);
                }
            } else {
                while self.word != 0 {
                    let idx = self
                        .word_start
                        .wrapping_add(self.word.trailing_zeros() as usize);
                    acc = f(acc, idx);
                    self.word &= self.word - 1;
                }
            }
            if !self.advance() {
                return acc;
            }
        }
    }
}

unsafe impl TrustedLen for SetBitIdxIter<'_, true> {}

pub struct FastU32BitmapIter<'a> {
    bytes: &'a [u8],
    shift: u32,
    bits_left: usize,
}

impl<'a> FastU32BitmapIter<'a> {
    pub fn new(bytes: &'a [u8], offset: usize, len: usize) -> Self {
        assert!(bytes.len() * 8 >= offset + len);
        let shift = (offset % 8) as u32;
        let bytes = &bytes[offset / 8..];
        Self {
            bytes,
            shift,
            bits_left: len,
        }
    }

    // The iteration logic that would normally follow the fast-path.
    fn next_remainder(&mut self) -> Option<u32> {
        if self.bits_left > 0 {
            let word = load_padded_le_u64(self.bytes);
            let mask;
            if self.bits_left >= 32 {
                mask = u32::MAX;
                self.bits_left -= 32;
                self.bytes = unsafe { self.bytes.get_unchecked(4..) };
            } else {
                mask = (1 << self.bits_left) - 1;
                self.bits_left = 0;
            }

            return Some((word >> self.shift) as u32 & mask);
        }

        None
    }

    /// Returns the remainder bits and how many there are,
    /// assuming the iterator was fully consumed.
    pub fn remainder(mut self) -> (u64, usize) {
        let bits_left = self.bits_left;
        let lo = self.next_remainder().unwrap_or(0);
        let hi = self.next_remainder().unwrap_or(0);
        (((hi as u64) << 32) | (lo as u64), bits_left)
    }
}

impl Iterator for FastU32BitmapIter<'_> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Fast path, can load a whole u64.
        if self.bits_left >= 64 {
            let chunk;
            unsafe {
                // SAFETY: bits_left ensures this is in-bounds.
                chunk = self.bytes.get_unchecked(0..8);
                self.bytes = self.bytes.get_unchecked(4..);
            }
            self.bits_left -= 32;
            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            return Some((word >> self.shift) as u32);
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let hint = calc_iters_remaining(self.bits_left, 64, 32);
        (hint, Some(hint))
    }
}

unsafe impl TrustedLen for FastU32BitmapIter<'_> {}

#[derive(Clone)]
pub struct FastU56BitmapIter<'a> {
    bytes: &'a [u8],
    shift: u32,
    bits_left: usize,
}

impl<'a> FastU56BitmapIter<'a> {
    pub fn new(bytes: &'a [u8], offset: usize, len: usize) -> Self {
        assert!(bytes.len() * 8 >= offset + len);
        let shift = (offset % 8) as u32;
        let bytes = &bytes[offset / 8..];
        Self {
            bytes,
            shift,
            bits_left: len,
        }
    }

    // The iteration logic that would normally follow the fast-path.
    fn next_remainder(&mut self) -> Option<u64> {
        if self.bits_left > 0 {
            let word = load_padded_le_u64(self.bytes);
            let mask;
            if self.bits_left >= 56 {
                mask = (1 << 56) - 1;
                self.bits_left -= 56;
                self.bytes = unsafe { self.bytes.get_unchecked(7..) };
            } else {
                mask = (1 << self.bits_left) - 1;
                self.bits_left = 0;
            };

            return Some((word >> self.shift) & mask);
        }

        None
    }

    /// Returns the remainder bits and how many there are,
    /// assuming the iterator was fully consumed. Output is safe but
    /// not specified if the iterator wasn't fully consumed.
    pub fn remainder(mut self) -> (u64, usize) {
        let bits_left = self.bits_left;
        let lo = self.next_remainder().unwrap_or(0);
        let hi = self.next_remainder().unwrap_or(0);
        ((hi << 56) | lo, bits_left)
    }
}

impl Iterator for FastU56BitmapIter<'_> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Fast path, can load a whole u64.
        if self.bits_left >= 64 {
            let chunk;
            unsafe {
                // SAFETY: bits_left ensures this is in-bounds.
                chunk = self.bytes.get_unchecked(0..8);
                self.bytes = self.bytes.get_unchecked(7..);
                self.bits_left -= 56;
            }

            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            let mask = (1 << 56) - 1;
            return Some((word >> self.shift) & mask);
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let hint = calc_iters_remaining(self.bits_left, 64, 56);
        (hint, Some(hint))
    }
}

unsafe impl TrustedLen for FastU56BitmapIter<'_> {}

pub struct FastU64BitmapIter<'a> {
    bytes: &'a [u8],
    shift: u32,
    bits_left: usize,
    next_word: u64,
}

impl<'a> FastU64BitmapIter<'a> {
    pub fn new(bytes: &'a [u8], offset: usize, len: usize) -> Self {
        assert!(bytes.len() * 8 >= offset + len);
        let shift = (offset % 8) as u32;
        let bytes = &bytes[offset / 8..];
        let next_word = load_padded_le_u64(bytes);
        let bytes = bytes.get(8..).unwrap_or(&[]);
        Self {
            bytes,
            shift,
            bits_left: len,
            next_word,
        }
    }

    #[inline]
    fn combine(&self, lo: u64, hi: u64) -> u64 {
        // Compiles to 128-bit SHRD instruction on x86-64.
        // Yes, the % 64 is important for the compiler to generate optimal code.
        let wide = ((hi as u128) << 64) | lo as u128;
        (wide >> (self.shift % 64)) as u64
    }

    // The iteration logic that would normally follow the fast-path.
    fn next_remainder(&mut self) -> Option<u64> {
        if self.bits_left > 0 {
            let lo = self.next_word;
            let hi = load_padded_le_u64(self.bytes);
            let mask;
            if self.bits_left >= 64 {
                mask = u64::MAX;
                self.bits_left -= 64;
                self.bytes = self.bytes.get(8..).unwrap_or(&[]);
            } else {
                mask = (1 << self.bits_left) - 1;
                self.bits_left = 0;
            };
            self.next_word = hi;

            return Some(self.combine(lo, hi) & mask);
        }

        None
    }

    /// Returns the remainder bits and how many there are,
    /// assuming the iterator was fully consumed. Output is safe but
    /// not specified if the iterator wasn't fully consumed.
    pub fn remainder(mut self) -> ([u64; 2], usize) {
        let bits_left = self.bits_left;
        let lo = self.next_remainder().unwrap_or(0);
        let hi = self.next_remainder().unwrap_or(0);
        ([lo, hi], bits_left)
    }
}

impl Iterator for FastU64BitmapIter<'_> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Fast path: can load two u64s in a row.
        // (Note that we already loaded one in the form of self.next_word).
        if self.bits_left >= 128 {
            let chunk;
            unsafe {
                // SAFETY: bits_left ensures this is in-bounds.
                chunk = self.bytes.get_unchecked(0..8);
                self.bytes = self.bytes.get_unchecked(8..);
            }
            let lo = self.next_word;
            let hi = u64::from_le_bytes(chunk.try_into().unwrap());
            self.next_word = hi;
            self.bits_left -= 64;

            return Some(self.combine(lo, hi));
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let hint = calc_iters_remaining(self.bits_left, 128, 64);
        (hint, Some(hint))
    }
}

unsafe impl TrustedLen for FastU64BitmapIter<'_> {}

/// This crates' equivalent of [`std::vec::IntoIter`] for [`Bitmap`].
#[derive(Debug, Clone)]
pub struct IntoIter {
    values: Bitmap,
    index: usize,
    end: usize,
}

impl IntoIter {
    /// Creates a new [`IntoIter`] from a [`Bitmap`]
    #[inline]
    pub fn new(values: Bitmap) -> Self {
        let end = values.len();
        Self {
            values,
            index: 0,
            end,
        }
    }
}

impl Iterator for IntoIter {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            return None;
        }
        let old = self.index;
        self.index += 1;
        Some(unsafe { self.values.get_bit_unchecked(old) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.end - self.index, Some(self.end - self.index))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let new_index = self.index + n;
        if new_index > self.end {
            self.index = self.end;
            None
        } else {
            self.index = new_index;
            self.next()
        }
    }
}

impl DoubleEndedIterator for IntoIter {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            None
        } else {
            self.end -= 1;
            Some(unsafe { self.values.get_bit_unchecked(self.end) })
        }
    }
}

unsafe impl TrustedLen for IntoIter {}
