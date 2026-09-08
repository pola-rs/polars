//! What a [`PlFixedSizeBinaryArray`] gains from being known to be [`Flat`].

use polars_buffer::Buffer;

use super::PlFixedSizeBinaryArray;
use crate::flat::Flat;

/// The methods a [`PlFixedSizeBinaryArray`] gains from holding one slot and one bit per element.
impl Flat<PlFixedSizeBinaryArray> {
    /// The backing values buffer, holding `len * width` bytes.
    #[inline(always)]
    pub const fn values(&self) -> &Buffer<u8> {
        &self.as_array().values
    }

    /// The values as a slice of `len * width` bytes.
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        self.as_array().values.as_slice()
    }

    /// Returns the bytes of the element at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        debug_assert!(i < self.as_array().length);
        let start = i * self.as_array().width;
        // SAFETY: the values hold the width of every element, so the element at `i` is in bounds.
        unsafe {
            self.as_array()
                .values
                .get_unchecked(start..start + self.as_array().width)
        }
    }
}

crate::impl_flat_methods!(PlFixedSizeBinaryArray, &[u8]);

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;

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
