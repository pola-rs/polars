use crate::bit_util::get_bit_raw;
use crate::trusted_len::TrustedLen;
use arrow::bitmap::Bitmap;

pub trait AddIterBitmap {
    /// constructs a new iterator
    fn iter(&self, len: usize) -> BitmapIter<'_>;
}

impl AddIterBitmap for Bitmap {
    fn iter(&self, len: usize) -> BitmapIter<'_> {
        let slice = self.buffer_ref().as_slice();
        BitmapIter::new(slice, 0, len)
    }
}

/// This whole struct is forked from arrow2
/// An iterator over bits according to the [LSB](https://en.wikipedia.org/wiki/Bit_numbering#Least_significant_bit),
/// i.e. the bytes `[4u8, 128u8]` correspond to `[false, false, true, false, ..., true]`.
#[derive(Debug, Clone)]
pub struct BitmapIter<'a> {
    bytes: &'a [u8],
    index: usize,
    end: usize,
}

impl<'a> BitmapIter<'a> {
    #[inline]
    pub fn new(slice: &'a [u8], offset: usize, len: usize) -> Self {
        // example:
        // slice.len() = 4
        // offset = 9
        // len = 23
        // result:
        let bytes = &slice[offset / 8..];
        // bytes.len() = 3
        let index = offset % 8;
        // index = 9 % 8 = 1
        let end = len + index;
        // end = 23 + 1 = 24
        assert!(end <= bytes.len() * 8);
        // maximum read before UB in bits: bytes.len() * 8 = 24
        // the first read from the end is `end - 1`, thus, end = 24 is ok

        Self { bytes, index, end }
    }
}

impl<'a> Iterator for BitmapIter<'a> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            return None;
        }
        let old = self.index;
        self.index += 1;
        // See comment in `new`
        Some(unsafe { get_bit_raw(self.bytes.as_ptr(), old) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.end - self.index;
        (exact, Some(exact))
    }
}

impl<'a> DoubleEndedIterator for BitmapIter<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<bool> {
        if self.index == self.end {
            None
        } else {
            self.end -= 1;
            // See comment in `new`; end was first decreased
            Some(unsafe { get_bit_raw(self.bytes.as_ptr(), self.end) })
        }
    }
}

unsafe impl TrustedLen for BitmapIter<'_> {}
