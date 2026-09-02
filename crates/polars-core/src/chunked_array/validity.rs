//! Combining the validity masks of chunks.
//!
//! A chunk hands its mask out as a [`PlBitmapRef`], which is either flat — one bit per element —
//! or scalar, a single bit standing for every element. The helpers here combine two of those
//! without writing a scalar mask out when they do not have to: the `and` of two scalar masks is
//! itself one bit, so a `ChunkedArray` that is fully null stays `O(1)` in memory across an
//! operation that only touches validity.

use arrow::bitmap::Bitmap;
use polars_array::PlBitmapRef;

/// The `and` of two masks over the same elements, or `None` if neither has a null.
///
/// The result is scalar exactly when both inputs are, so it is `O(1)` for two masks that are; a
/// flat mask on either side makes it `O(len)`, the other being written out to meet it.
///
/// # Panics
/// Panics if the masks are over a different number of elements.
pub fn combine_validities_and(
    lhs: Option<PlBitmapRef<'_>>,
    rhs: Option<PlBitmapRef<'_>>,
) -> Option<Bitmap> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => {
            assert_eq!(lhs.len(), rhs.len(), "validity masks cover different lengths");
            match (lhs.scalar_value(), rhs.scalar_value()) {
                // Two single bits `and` to a single bit, which covers every element in turn.
                (Some(lhs), Some(rhs)) => Some(Bitmap::new_with_value(lhs && rhs, 1)),
                _ => arrow::compute::utils::combine_validities_and(
                    Some(&lhs.to_flat()),
                    Some(&rhs.to_flat()),
                ),
            }
        },
        (Some(validity), None) | (None, Some(validity)) => Some(validity.to_flat_or_scalar()),
        (None, None) => None,
    }
}

/// The bits of `mask`, inverted: set where an element is null.
///
/// A scalar mask inverts to a single bit, so this is `O(1)` for one and `O(len)` for a flat one.
pub fn invert(mask: PlBitmapRef<'_>) -> Bitmap {
    match mask.scalar_value() {
        Some(value) => Bitmap::new_with_value(!value, 1),
        None => !mask.flat_bitmap().unwrap(),
    }
}

/// An extension of [`PlBitmapRef`] with the conversion the helpers here need.
pub trait PlBitmapRefExt {
    /// This mask as a [`Bitmap`], keeping the scalar representation where it has one.
    ///
    /// This is [`PlBitmapRef::to_flat`] that does not write a scalar mask out: the single bit is
    /// handed back as the one-bit bitmap it is, which the arrays of `polars-array` read as scalar.
    fn to_flat_or_scalar(&self) -> Bitmap;
}

impl PlBitmapRefExt for PlBitmapRef<'_> {
    #[inline]
    fn to_flat_or_scalar(&self) -> Bitmap {
        match self.scalar_value() {
            Some(value) => Bitmap::new_with_value(value, 1),
            None => self.flat_bitmap().unwrap().clone(),
        }
    }
}
