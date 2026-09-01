//! What a [`PlListArray`] gains from being known to be [`Flat`].

use arrow::bitmap::Bitmap;
use polars_buffer::Buffer;

use super::PlListArray;
use crate::array::PlArray;
use crate::flat::Flat;

/// The methods a [`PlListArray`] gains from holding the range of every element and one validity
/// bit per element.
///
/// These are the counterparts of the methods on [`ListArray`](arrow::array::ListArray), whose
/// offsets *are* the ranges of its elements: they hand out the backing buffers as they are and
/// read them without a [`broadcast_index`](crate::broadcast::broadcast_index). Each shadows the
/// broadcast-aware method of the same name on [`PlListArray`], which remains reachable through the
/// deref — including the iterators, which read the elements the same way either way.
impl Flat<PlListArray> {
    /// The backing offsets buffer, holding exactly [`len`](PlListArray::len) `+ 1` offsets.
    ///
    /// Unlike [`PlListArray::flat_offsets`], this needs no [`Option`] to admit scalar offsets:
    /// element `i` covers `offsets[i]..offsets[i + 1]` of [`values`](PlListArray::values). The
    /// offsets are not normalized: the first one is whatever slicing left it, not necessarily
    /// zero.
    #[inline(always)]
    pub const fn offsets(&self) -> &Buffer<u64> {
        &self.0.offsets
    }

    /// The validity mask, if any element may be null, as an ordinary [`Bitmap`] of exactly
    /// [`len`](PlListArray::len) bits.
    ///
    /// Unlike [`PlListArray::validity`], this needs no [`PlBitmapRef`](crate::PlBitmapRef) to hide
    /// a scalar bit: bit `i` is element `i`.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.0.validity.as_ref()
    }

    /// Consumes this array into its internal components, whose ranges and bits are one per
    /// element.
    ///
    /// The length is not part of the result: it is one fewer than the number of offsets.
    #[inline]
    pub fn into_inner(self) -> (Box<dyn PlArray>, Buffer<u64>, Option<Bitmap>) {
        let PlListArray {
            values,
            offsets,
            length: _,
            validity,
        } = self.0;

        (values, offsets, validity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlPrimitiveArray;

    fn values() -> Box<dyn PlArray> {
        Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4]))
    }

    #[test]
    fn buffers_are_handed_out_as_they_are() {
        let arr = PlListArray::new_scalar(Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])), 3);
        let flat = arr.to_flat();

        assert_eq!(flat.offsets().as_slice(), [0, 2, 4, 6]);
        assert!(flat.validity().is_none());
        assert_eq!(flat, arr);
    }

    #[test]
    fn a_scalar_mask_is_written_out() {
        let arr = PlListArray::new_full_null(values(), 3);
        let flat = arr.to_flat();

        assert_eq!(flat.validity().unwrap().len(), 3);
        assert_eq!(flat.null_count(), 3);
    }

    #[test]
    fn into_inner_gives_up_the_length() {
        let arr = PlListArray::from_offsets(values(), Buffer::from(vec![0u64, 2, 4]));
        let (values, offsets, validity) = arr.to_flat().into_inner();

        assert_eq!(values.len(), 4);
        assert_eq!(offsets.as_slice(), [0, 2, 4]);
        assert!(validity.is_none());
    }
}
