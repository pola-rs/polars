use polars_utils::slice::load_padded_le_u64;

use super::get_bit_unchecked;
use crate::bitmap::MutableBitmap;
use crate::trusted_len::TrustedLen;

/// An iterator over bits according to the [LSB](https://en.wikipedia.org/wiki/Bit_numbering#Least_significant_bit),
/// i.e. the bytes `[4u8, 128u8]` correspond to `[false, false, true, false, ..., true]`.
#[derive(Debug, Clone)]
pub struct BitmapIter<'a> {
    bytes: &'a [u8],
    word: u64,
    word_len: usize,
    rest_len: usize,
}

impl<'a> BitmapIter<'a> {
    /// Creates a new [`BitmapIter`].
    pub fn new(bytes: &'a [u8], offset: usize, len: usize) -> Self {
        if len == 0 {
            return Self {
                bytes,
                word: 0,
                word_len: 0,
                rest_len: 0,
            };
        }

        assert!(bytes.len() * 8 >= offset + len);
        let first_byte_idx = offset / 8;
        let bytes = &bytes[first_byte_idx..];
        let offset = offset % 8;

        // Make sure during our hot loop all our loads are full 8-byte loads
        // by loading the remainder now if it exists.
        let word = load_padded_le_u64(bytes) >> offset;
        let mod8 = bytes.len() % 8;
        let first_word_bytes = if mod8 > 0 { mod8 } else { 8 };
        let bytes = &bytes[first_word_bytes..];

        let word_len = (first_word_bytes * 8 - offset).min(len);
        let rest_len = len - word_len;
        Self {
            bytes,
            word,
            word_len,
            rest_len,
        }
    }

    /// Consume and returns the numbers of `1` / `true` values at the beginning of the iterator.
    ///
    /// This performs the same operation as `(&mut iter).take_while(|b| b).count()`.
    ///
    /// This is a lot more efficient than consecutively polling the iterator and should therefore
    /// be preferred, if the use-case allows for it.
    pub fn take_leading_ones(&mut self) -> usize {
        let word_ones = usize::min(self.word_len, self.word.trailing_ones() as usize);
        self.word_len -= word_ones;
        self.word = self.word.wrapping_shr(word_ones as u32);

        if self.word_len != 0 {
            return word_ones;
        }

        let mut num_leading_ones = word_ones;

        while self.rest_len != 0 {
            self.word_len = usize::min(self.rest_len, 64);
            self.rest_len -= self.word_len;

            unsafe {
                let chunk = self.bytes.get_unchecked(..8).try_into().unwrap();
                self.word = u64::from_le_bytes(chunk);
                self.bytes = self.bytes.get_unchecked(8..);
            }

            let word_ones = usize::min(self.word_len, self.word.trailing_ones() as usize);
            self.word_len -= word_ones;
            self.word = self.word.wrapping_shr(word_ones as u32);
            num_leading_ones += word_ones;

            if self.word_len != 0 {
                return num_leading_ones;
            }
        }

        num_leading_ones
    }

    /// Consume and returns the numbers of `0` / `false` values that the start of the iterator.
    ///
    /// This performs the same operation as `(&mut iter).take_while(|b| !b).count()`.
    ///
    /// This is a lot more efficient than consecutively polling the iterator and should therefore
    /// be preferred, if the use-case allows for it.
    pub fn take_leading_zeros(&mut self) -> usize {
        let word_zeros = usize::min(self.word_len, self.word.trailing_zeros() as usize);
        self.word_len -= word_zeros;
        self.word = self.word.wrapping_shr(word_zeros as u32);

        if self.word_len != 0 {
            return word_zeros;
        }

        let mut num_leading_zeros = word_zeros;

        while self.rest_len != 0 {
            self.word_len = usize::min(self.rest_len, 64);
            self.rest_len -= self.word_len;
            unsafe {
                let chunk = self.bytes.get_unchecked(..8).try_into().unwrap();
                self.word = u64::from_le_bytes(chunk);
                self.bytes = self.bytes.get_unchecked(8..);
            }

            let word_zeros = usize::min(self.word_len, self.word.trailing_zeros() as usize);
            self.word_len -= word_zeros;
            self.word = self.word.wrapping_shr(word_zeros as u32);
            num_leading_zeros += word_zeros;

            if self.word_len != 0 {
                return num_leading_zeros;
            }
        }

        num_leading_zeros
    }

    /// Returns the number of remaining elements in the iterator
    #[inline]
    pub fn num_remaining(&self) -> usize {
        self.word_len + self.rest_len
    }

    /// Collect at most `n` elements from this iterator into `bitmap`
    pub fn collect_n_into(&mut self, bitmap: &mut MutableBitmap, n: usize) {
        fn collect_word(
            word: &mut u64,
            word_len: &mut usize,
            bitmap: &mut MutableBitmap,
            n: &mut usize,
        ) {
            while *n > 0 && *word_len > 0 {
                {
                    let trailing_ones = u32::min(word.trailing_ones(), *word_len as u32);
                    let shift = u32::min(usize::min(*n, u32::MAX as usize) as u32, trailing_ones);
                    *word = word.wrapping_shr(shift);
                    *word_len -= shift as usize;
                    *n -= shift as usize;

                    bitmap.extend_constant(shift as usize, true);
                }

                {
                    let trailing_zeros = u32::min(word.trailing_zeros(), *word_len as u32);
                    let shift = u32::min(usize::min(*n, u32::MAX as usize) as u32, trailing_zeros);
                    *word = word.wrapping_shr(shift);
                    *word_len -= shift as usize;
                    *n -= shift as usize;

                    bitmap.extend_constant(shift as usize, false);
                }
            }
        }

        let mut n = usize::min(n, self.num_remaining());
        bitmap.reserve(n);

        collect_word(&mut self.word, &mut self.word_len, bitmap, &mut n);

        if n == 0 {
            return;
        }

        let num_words = n / 64;

        if num_words > 0 {
            assert!(self.bytes.len() >= num_words * size_of::<u64>());

            bitmap.extend_from_slice(self.bytes, 0, num_words * u64::BITS as usize);

            self.bytes = unsafe { self.bytes.get_unchecked(num_words * 8..) };
            self.rest_len -= num_words * u64::BITS as usize;
            n -= num_words * u64::BITS as usize;
        }

        if n == 0 {
            return;
        }

        assert!(self.bytes.len() >= size_of::<u64>());

        self.word_len = usize::min(self.rest_len, 64);
        self.rest_len -= self.word_len;
        unsafe {
            let chunk = self.bytes.get_unchecked(..8).try_into().unwrap();
            self.word = u64::from_le_bytes(chunk);
            self.bytes = self.bytes.get_unchecked(8..);
        }

        collect_word(&mut self.word, &mut self.word_len, bitmap, &mut n);

        debug_assert!(self.num_remaining() == 0 || n == 0);
    }
}

impl Iterator for BitmapIter<'_> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.word_len == 0 {
            if self.rest_len == 0 {
                return None;
            }

            self.word_len = self.rest_len.min(64);
            self.rest_len -= self.word_len;

            unsafe {
                let chunk = self.bytes.get_unchecked(..8).try_into().unwrap();
                self.word = u64::from_le_bytes(chunk);
                self.bytes = self.bytes.get_unchecked(8..);
            }
        }

        let ret = self.word & 1 != 0;
        self.word >>= 1;
        self.word_len -= 1;
        Some(ret)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let num_remaining = self.num_remaining();
        (num_remaining, Some(num_remaining))
    }

    #[inline]
    fn nth(&mut self, mut n: usize) -> Option<Self::Item> {
        if n >= self.word_len + self.rest_len {
            self.word = 0;
            self.word_len = 0;
            self.rest_len = 0;
            return None;
        }

        // Advance words in buffer, skip words as needed
        if n >= self.word_len {
            n -= self.word_len;

            let word_offset = n / 64;
            n -= word_offset * 64;
            self.rest_len -= word_offset * 64;

            self.word_len = self.rest_len.min(64);
            self.rest_len -= self.word_len;

            let byte_offset = 8 * word_offset;

            // Safety: bytes is large enough at construction time.
            debug_assert!(byte_offset + 8 <= self.bytes.len());
            unsafe {
                let chunk = self
                    .bytes
                    .get_unchecked(byte_offset..byte_offset + 8)
                    .try_into()
                    .unwrap();
                self.word = u64::from_le_bytes(chunk);
                self.bytes = self.bytes.get_unchecked(byte_offset + 8..);
            }
        }

        // At this point, n < self.word_len
        debug_assert!(self.word_len > n);

        // Advance index by n and take value at final index
        self.word >>= n;
        self.word_len -= n;

        let ret = self.word & 1 != 0;
        self.word >>= 1;
        self.word_len -= 1;
        Some(ret)
    }
}

impl DoubleEndedIterator for BitmapIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<bool> {
        if self.rest_len > 0 {
            self.rest_len -= 1;
            Some(unsafe { get_bit_unchecked(self.bytes, self.rest_len) })
        } else if self.word_len > 0 {
            self.word_len -= 1;
            Some(self.word & (1 << self.word_len) != 0)
        } else {
            None
        }
    }
}

unsafe impl TrustedLen for BitmapIter<'_> {}
impl ExactSizeIterator for BitmapIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_into_17579() {
        let mut bitmap = MutableBitmap::with_capacity(64);
        BitmapIter::new(&[0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0], 0, 128)
            .collect_n_into(&mut bitmap, 129);

        let bitmap = bitmap.freeze();

        assert_eq!(bitmap.set_bits(), 4);
    }

    #[test]
    #[ignore = "Fuzz test. Too slow"]
    fn test_fuzz_collect_into() {
        for _ in 0..10_000 {
            let mut set_bits = 0;
            let mut unset_bits = 0;

            let mut length = 0;
            let mut pattern = Vec::new();
            for _ in 0..rand::random::<u64>() % 1024 {
                let bs = rand::random::<u8>() % 4;

                let word = match bs {
                    0 => u64::MIN,
                    1 => u64::MAX,
                    2 | 3 => rand::random(),
                    _ => unreachable!(),
                };

                pattern.extend_from_slice(&word.to_le_bytes());
                set_bits += word.count_ones();
                unset_bits += word.count_zeros();
                length += 64;
            }

            for _ in 0..rand::random::<u64>() % 7 {
                let b = rand::random::<u8>();
                pattern.push(b);
                set_bits += b.count_ones();
                unset_bits += b.count_zeros();
                length += 8;
            }

            let last_length = rand::random::<u64>() % 8;
            if last_length != 0 {
                let b = rand::random::<u8>();
                pattern.push(b);
                let ones = (b & ((1 << last_length) - 1)).count_ones();
                set_bits += ones;
                unset_bits += last_length as u32 - ones;
                length += last_length;
            }

            let mut iter = BitmapIter::new(&pattern, 0, length as usize);
            let mut bitmap = MutableBitmap::with_capacity(length as usize);

            while iter.num_remaining() > 0 {
                let len_before = bitmap.len();
                let n = rand::random::<u64>() as usize % iter.num_remaining();
                iter.collect_n_into(&mut bitmap, n);

                // Ensure we are booking the progress we expect
                assert_eq!(bitmap.len(), len_before + n);
            }

            let bitmap = bitmap.freeze();

            assert_eq!(bitmap.set_bits(), set_bits as usize);
            assert_eq!(bitmap.unset_bits(), unset_bits as usize);
        }
    }

    #[test]
    #[ignore = "Fuzz test. Too slow"]
    fn test_fuzz_leading_ops() {
        for _ in 0..10_000 {
            let mut length = 0;
            let mut pattern = Vec::new();
            for _ in 0..rand::random::<u64>() % 1024 {
                let bs = rand::random::<u8>() % 4;

                let word = match bs {
                    0 => u64::MIN,
                    1 => u64::MAX,
                    2 | 3 => rand::random(),
                    _ => unreachable!(),
                };

                pattern.extend_from_slice(&word.to_le_bytes());
                length += 64;
            }

            for _ in 0..rand::random::<u64>() % 7 {
                pattern.push(rand::random::<u8>());
                length += 8;
            }

            let last_length = rand::random::<u64>() % 8;
            if last_length != 0 {
                pattern.push(rand::random::<u8>());
                length += last_length;
            }

            let mut iter = BitmapIter::new(&pattern, 0, length as usize);

            let mut prev_remaining = iter.num_remaining();
            while iter.num_remaining() != 0 {
                let num_ones = iter.clone().take_leading_ones();
                assert_eq!(num_ones, (&mut iter).take_while(|&b| b).count());

                let num_zeros = iter.clone().take_leading_zeros();
                assert_eq!(num_zeros, (&mut iter).take_while(|&b| !b).count());

                // Ensure that we are making progress
                assert!(iter.num_remaining() < prev_remaining);
                prev_remaining = iter.num_remaining();
            }

            assert_eq!(iter.take_leading_zeros(), 0);
            assert_eq!(iter.take_leading_ones(), 0);
        }
    }

    #[test]
    #[allow(clippy::iter_nth_zero)]
    fn test_bitmap_iter_nth() {
        // Calling nth repeatedly advances through the bitmap
        {
            let mut iter = BitmapIter::new(&[0b10110001], 0, 8);
            assert_eq!(iter.nth(0), Some(true));
            assert_eq!(iter.nth(0), Some(false));
            assert_eq!(iter.nth(2), Some(true));
            assert_eq!(iter.nth(3), None);

            assert_eq!(iter.next(), None);
        }

        // Test parity with next()-based implementation on of singular call to nth()
        for len in [0, 1, 2, 63, 64, 65, 127, 128, 129] {
            for offset in [0, 1, 2] {
                // binary '01010101' == 85
                let iter = BitmapIter::new(
                    &[
                        0, 1, 2, 4, 8, 16, 32, 64, 85, 170, 85, 170, 85, 170, 85, 170, 255, 0,
                    ],
                    offset,
                    len,
                );

                for i in 0..=len {
                    let mut iter_expected = iter.clone();
                    let mut iter_test = iter.clone();

                    let prev_rest_len = iter_test.rest_len;
                    let prev_word_len = iter_test.word_len;

                    assert_eq!(len, prev_rest_len + prev_word_len);

                    // Iterate.
                    let out = iter_test.nth(i);
                    for _ in 0..i {
                        iter_expected.next();
                    }
                    let expected = iter_expected.next();

                    // Check value.
                    assert_eq!(out, expected);

                    // Check internal sate.
                    let final_rest_len = iter_test.rest_len;
                    let final_word_len = iter_test.word_len;
                    match out {
                        Some(_) => assert_eq!(
                            prev_rest_len + prev_word_len,
                            i + 1 + final_rest_len + final_word_len
                        ),
                        None => {
                            assert!(i >= prev_rest_len + prev_word_len);
                            assert_eq!(final_rest_len + final_word_len, 0)
                        },
                    };
                }
            }
        }

        // Check internal state on repeat calls to nth().
        {
            for len in [0, 63, 64, 65, 126, 128, 129] {
                let mut iter =
                    BitmapIter::new(&[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0], 0, len);
                for step in [0, 1, 2, 3] {
                    for i in (0..len + step + 1).step_by(step + 1) {
                        let prev_rest_len = iter.rest_len;
                        let prev_word_len = iter.word_len;

                        let out = iter.nth(step);

                        let final_rest_len = iter.rest_len;
                        let final_word_len = iter.word_len;
                        match out {
                            Some(_) => assert_eq!(
                                prev_rest_len + prev_word_len,
                                step + 1 + final_rest_len + final_word_len
                            ),
                            None => {
                                assert!(i >= prev_rest_len + prev_word_len);
                                assert_eq!(final_rest_len + final_word_len, 0)
                            },
                        };
                    }
                }
            }
        }

        // Edge cases
        let mut iter = BitmapIter::new(&[], 0, 0);
        assert_eq!(iter.nth(0), None);
    }
}
