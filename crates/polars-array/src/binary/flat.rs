//! What a [`PlBinaryArray`] gains from being known to be [`Flat`].

use std::ops::Range;

use arrow::bitmap::Bitmap;
use polars_buffer::Buffer;

use super::PlBinaryArray;
use crate::flat::Flat;

/// The methods a [`PlBinaryArray`] gains from holding the range of every element and one validity
/// bit per element.
///
/// These are the counterparts of the methods on [`BinaryArray`](arrow::array::BinaryArray), whose
/// offsets *are* the ranges of its elements: they hand out the backing buffers as they are and
/// read them without a [`broadcast_index`](crate::broadcast::broadcast_index). Each shadows the
/// broadcast-aware method of the same name on [`PlBinaryArray`], which remains reachable through
/// the deref — including the iterators, which read the elements the same way either way.
impl Flat<PlBinaryArray> {
    /// The backing offsets buffer, holding exactly [`len`](PlBinaryArray::len) `+ 1` offsets.
    ///
    /// Unlike [`PlBinaryArray::flat_offsets`], this needs no [`Option`] to admit scalar offsets:
    /// element `i` covers `offsets[i]..offsets[i + 1]` of [`values`](PlBinaryArray::values). The
    /// offsets are not normalized: the first one is whatever slicing left it, not necessarily
    /// zero.
    #[inline(always)]
    pub const fn offsets(&self) -> &Buffer<u64> {
        &self.0.offsets
    }

    /// The backing values buffer, holding the bytes the offsets cut the elements out of.
    ///
    /// Being flat says nothing about the values: they are not trimmed to what the offsets reach,
    /// so they may hold bytes before the first offset and after the last, exactly as
    /// [`PlBinaryArray::values`] does.
    #[inline(always)]
    pub const fn values(&self) -> &Buffer<u8> {
        &self.0.values
    }

    /// The values as a slice, which the offsets cut the elements out of.
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        self.0.values.as_slice()
    }

    /// The validity mask, if any element may be null, as an ordinary [`Bitmap`] of exactly
    /// [`len`](PlBinaryArray::len) bits.
    ///
    /// Unlike [`PlBinaryArray::validity`], this needs no [`PlBitmapRef`](crate::PlBitmapRef) to
    /// hide a scalar bit: bit `i` is element `i`.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.0.validity.as_ref()
    }

    /// Consumes this array into its internal components, whose ranges and bits are one per
    /// element.
    ///
    /// The length is not part of the result: it is one fewer than the number of offsets.
    #[inline]
    pub fn into_inner(self) -> (Buffer<u8>, Buffer<u64>, Option<Bitmap>) {
        let PlBinaryArray {
            values,
            offsets,
            length: _,
            validity,
        } = self.0;

        (values, offsets, validity)
    }

    /// The range of [`Self::values`] the element at `i` covers.
    ///
    /// The range of a null element is undetermined (it can be any valid range).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value_range(&self, i: usize) -> Range<usize> {
        assert!(i < self.0.length, "index out of bounds");
        unsafe { self.value_range_unchecked(i) }
    }

    /// The range of [`Self::values`] the element at `i` covers.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_range_unchecked(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.0.length);
        // SAFETY: the offsets hold one slot more than the starts, so `i + 1` is in bounds, and
        // every offset is at most the length of the values and therefore fits in a `usize`.
        unsafe {
            let start = *self.0.offsets.get_unchecked(i) as usize;
            let end = *self.0.offsets.get_unchecked(i + 1) as usize;
            start..end
        }
    }

    /// Returns the bytes of the element at `i`.
    ///
    /// The value of a null element is undetermined (it can be any byte string).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value(&self, i: usize) -> &[u8] {
        assert!(i < self.0.length, "index out of bounds");
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the bytes of the element at `i`.
    ///
    /// The value of a null element is undetermined (it can be any byte string).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        let range = unsafe { self.value_range_unchecked(i) };
        // SAFETY: the offsets are ordered and bounded by the length of the values.
        unsafe { self.0.values.get_unchecked(range) }
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        assert!(i < self.0.length, "index out of bounds");
        unsafe { self.is_valid_unchecked(i) }
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn is_valid_unchecked(&self, i: usize) -> bool {
        debug_assert!(i < self.0.length);
        // SAFETY: the mask has one bit per element, so `i` is in bounds of it too.
        self.validity()
            .is_none_or(|validity| unsafe { validity.get_bit_unchecked(i) })
    }

    /// Returns whether the element at `i` is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn is_null(&self, i: usize) -> bool {
        !self.is_valid(i)
    }

    /// Returns whether the element at `i` is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        unsafe { !self.is_valid_unchecked(i) }
    }

    /// Returns the bytes of the element at `i`, or `None` if it is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn get(&self, i: usize) -> Option<&[u8]> {
        assert!(i < self.0.length, "index out of bounds");
        unsafe { self.get_unchecked(i) }
    }

    /// Returns the bytes of the element at `i`, or `None` if it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> Option<&[u8]> {
        unsafe { self.is_valid_unchecked(i).then(|| self.value_unchecked(i)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffers_are_handed_out_as_they_are() {
        let arr = PlBinaryArray::new_scalar(b"ab", 3);
        let flat = arr.to_flat();

        assert_eq!(flat.offsets().as_slice(), [0, 2, 4, 6]);
        assert_eq!(flat.as_slice(), b"ababab");
        assert_eq!(flat.values().len(), 6);
        assert!(flat.validity().is_none());
        assert_eq!(flat, arr);
    }

    #[test]
    fn a_scalar_mask_is_written_out() {
        let arr = PlBinaryArray::new_full_null(3);
        let flat = arr.to_flat();

        assert_eq!(flat.validity().unwrap().len(), 3);
        assert_eq!(flat.null_count(), 3);
    }

    #[test]
    fn elements_are_read_without_a_broadcast() {
        let flat = PlBinaryArray::from_values_iter([b"foo".as_slice(), b"", b"bar"])
            .with_validity(Some(Bitmap::from_iter([true, false, true])))
            .to_flat();

        assert_eq!(flat.value_range(0), 0..3);
        assert_eq!(flat.value(0), b"foo");
        assert_eq!(flat.value(1), b"");
        assert_eq!(flat.get(0), Some(b"foo".as_slice()));
        assert_eq!(flat.get(1), None);
        assert!(flat.is_valid(2));
        assert!(flat.is_null(1));

        assert_eq!(unsafe { flat.value_range_unchecked(2) }, 3..6);
        assert_eq!(unsafe { flat.value_unchecked(2) }, b"bar");
        assert_eq!(unsafe { flat.get_unchecked(1) }, None);
        assert!(unsafe { flat.is_null_unchecked(1) });
        assert!(unsafe { flat.is_valid_unchecked(0) });
    }
}
