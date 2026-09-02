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
/// A scalar mask on either side is a shortcut rather than something to write out: two of them
/// `and` to a single bit, one that is unset everywhere makes the whole result a single unset bit,
/// and one that is set everywhere hands the other side back in whatever representation it is in.
/// Only two flat masks cost `O(len)`.
///
/// The result is therefore flat *or* scalar, whichever came out cheaper, so it goes on an array
/// through `set_validity_broadcast` rather than `set_validity` — the latter takes a flat mask
/// only.
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
                // A scalar mask that is set everywhere leaves the other one as it is, in whatever
                // representation that one is in; one that is unset everywhere makes the result a
                // single unset bit, whatever the other one holds.
                (Some(true), None) => Some(rhs.to_flat_or_scalar()),
                (None, Some(true)) => Some(lhs.to_flat_or_scalar()),
                (Some(false), None) | (None, Some(false)) => Some(Bitmap::new_with_value(false, 1)),
                (None, None) => arrow::compute::utils::combine_validities_and(
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
/// The result is in the same representation as `mask`, so this is `O(1)` for a scalar mask —
/// whose single bit inverts to a single bit — and `O(len)` for a flat one.
pub fn invert(mask: PlBitmapRef<'_>) -> Bitmap {
    // The backing bitmap is flat or scalar for the mask's length, and inverting it bit for bit
    // leaves it that way; there is nothing to expand first.
    !mask.into_inner().0
}

/// An extension of [`PlBitmapRef`] with the conversion the helpers here need.
pub trait PlBitmapRefExt {
    /// This mask as a [`Bitmap`], keeping the scalar representation where it has one.
    ///
    /// This is [`PlBitmapRef::to_flat`] that does not write a scalar mask out: the single bit is
    /// handed back as the one-bit bitmap it is, which the arrays of `polars-array` read as scalar.
    /// The result goes on an array through `set_validity_broadcast`, which is the setter that
    /// admits both representations.
    fn to_flat_or_scalar(&self) -> Bitmap;
}

impl PlBitmapRefExt for PlBitmapRef<'_> {
    #[inline]
    fn to_flat_or_scalar(&self) -> Bitmap {
        // The backing bitmap is already flat or scalar for the mask's length, which is exactly
        // what an array accepts as its own mask: hand it over as it is.
        self.into_inner().0.clone()
    }
}
