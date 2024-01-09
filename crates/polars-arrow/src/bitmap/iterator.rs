use super::bitmask::BitMask;
use super::Bitmap;
use crate::trusted_len::TrustedLen;

pub struct TrueIdxIter<'a> {
    mask: BitMask<'a>,
    first_unknown: usize,
    i: usize,
    len: usize,
    remaining: usize,
}

impl<'a> TrueIdxIter<'a> {
    #[inline]
    pub fn new(len: usize, validity: Option<&'a Bitmap>) -> Self {
        if let Some(bitmap) = validity {
            assert!(len == bitmap.len());
            Self {
                mask: BitMask::from_bitmap(bitmap),
                first_unknown: 0,
                i: 0,
                remaining: bitmap.len() - bitmap.unset_bits(),
                len,
            }
        } else {
            Self {
                mask: BitMask::default(),
                first_unknown: len,
                i: 0,
                remaining: len,
                len,
            }
        }
    }
}

impl<'a> Iterator for TrueIdxIter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Fast path for many non-nulls in a row.
        if self.i < self.first_unknown {
            let ret = self.i;
            self.i += 1;
            self.remaining -= 1;
            return Some(ret);
        }

        while self.i < self.len {
            let mask = self.mask.get_u32(self.i);
            let num_null = mask.trailing_zeros();
            self.i += num_null as usize;
            if num_null < 32 {
                self.first_unknown = self.i + (mask >> num_null).trailing_ones() as usize;
                let ret = self.i;
                self.i += 1;
                self.remaining -= 1;
                return Some(ret);
            }
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

unsafe impl<'a> TrustedLen for TrueIdxIter<'a> {}

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
