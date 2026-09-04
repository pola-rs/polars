//! What a [`PlFixedSizeBinaryArray`] gains from being known to be [`Flat`].

use arrow::bitmap::Bitmap;
use polars_buffer::Buffer;

use super::PlFixedSizeBinaryArray;
use crate::flat::Flat;

/// The methods a [`PlFixedSizeBinaryArray`] gains from holding the bytes of every element and one
/// validity bit per element.
impl Flat<PlFixedSizeBinaryArray> {
    /// The backing values buffer, holding exactly [`len`](PlFixedSizeBinaryArray::len) `*`
    /// [`width`](PlFixedSizeBinaryArray::width) bytes.
    #[inline(always)]
    pub const fn values(&self) -> &Buffer<u8> {
        &self.0.values
    }

    /// The values as a slice of exactly [`len`](PlFixedSizeBinaryArray::len) `*`
    /// [`width`](PlFixedSizeBinaryArray::width) bytes.
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        self.0.values.as_slice()
    }

    /// The validity mask, if any element may be null, as an ordinary [`Bitmap`] of exactly
    /// [`len`](PlFixedSizeBinaryArray::len) bits.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.0.validity.as_ref()
    }

    /// Returns the bytes of the element at `i`.
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
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        debug_assert!(i < self.0.length);
        let start = i * self.0.width;
        // SAFETY: the values hold the width of every element, so the element at `i` is in bounds.
        unsafe { self.0.values.get_unchecked(start..start + self.0.width) }
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
    use crate::bitmap::PlBitmap;

    #[test]
    fn to_flat_writes_the_elements_out() {
        let scalar = PlFixedSizeBinaryArray::new_scalar(b"ab", 3);
        let flat = scalar.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.values().len(), 6);
        assert_eq!(flat.as_slice(), b"ababab");
        assert_eq!(*flat, scalar);

        // A scalar mask is written out alongside them.
        let null_scalar = PlFixedSizeBinaryArray::new_full_null(2, 3);
        let flat = null_scalar.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.validity().unwrap().len(), 3);
        assert_eq!(flat.null_count(), 3);
        assert_eq!(*flat, null_scalar);
    }

    #[test]
    fn as_flat_borrows_an_already_flat_array() {
        let arr = PlFixedSizeBinaryArray::from_vec(vec![1u8, 2, 3, 4], 2).with_validity(Some(
            PlBitmap::from_bitmap(Bitmap::from_iter([true, false])),
        ));
        let flat = arr.as_flat().expect("the array is flat");

        assert_eq!(flat.as_slice(), [1, 2, 3, 4]);
        assert_eq!(flat.validity().unwrap().len(), 2);
        assert_eq!(*flat, arr);

        // Neither scalar values nor a scalar validity mask can be borrowed as flat, however long
        // the array is: rejecting one is `O(1)`.
        assert!(
            PlFixedSizeBinaryArray::new_full_null(2, 1_000_000_000)
                .as_flat()
                .is_none()
        );
        assert!(
            PlFixedSizeBinaryArray::from_vec(vec![1u8, 2, 3, 4], 2)
                .with_validity(Some(PlBitmap::new_scalar(false, 2)))
                .as_flat()
                .is_none()
        );

        // One element is both flat and scalar, so it is borrowed rather than written out.
        assert!(
            PlFixedSizeBinaryArray::new_scalar(b"ab", 1)
                .as_flat()
                .is_some()
        );
    }

    #[test]
    fn elements_are_read_without_a_broadcast() {
        let arr =
            PlFixedSizeBinaryArray::from_vec(vec![1u8, 2, 3, 4, 5, 6], 2).with_validity(Some(
                PlBitmap::from_bitmap(Bitmap::from_iter([true, false, true])),
            ));
        let flat = arr.to_flat();

        assert_eq!(flat.value(0), [1, 2]);
        assert_eq!(flat.value(1), [3, 4]);
        assert_eq!(flat.get(0), Some([1, 2].as_slice()));
        assert_eq!(flat.get(1), None);
        assert!(flat.is_valid(2));
        assert!(flat.is_null(1));

        assert_eq!(unsafe { flat.value_unchecked(2) }, [5, 6]);
        assert_eq!(unsafe { flat.get_unchecked(1) }, None);
        assert!(unsafe { flat.is_null_unchecked(1) });
        assert!(unsafe { flat.is_valid_unchecked(0) });
    }
}
