//! Combining the validity masks of chunks.
//!
//! A chunk hands its mask out as a [`PlBitmapRef`], which is either flat — one bit per element —
//! or scalar, a single bit standing for every element. The helpers here combine two of those
//! without writing a scalar mask out when they do not have to: the `and` of two scalar masks is
//! itself one bit, so a `ChunkedArray` that is fully null stays `O(1)` in memory across an
//! operation that only touches validity.

use arrow::bitmap::Bitmap;
use polars_array::{PlBitmap, PlBitmapRef};

/// The `and` of two masks over the same elements, or `None` if neither has a null.
///
/// A scalar mask on either side is a shortcut rather than something to write out: two of them
/// `and` to a single bit, one that is unset everywhere makes the whole result a single unset bit,
/// and one that is set everywhere hands the other side back in whatever representation it is in.
/// Only two flat masks cost `O(len)`.
///
/// The result is therefore flat *or* scalar, whichever came out cheaper, which is why it comes
/// back as a [`PlBitmap`]: the mask carries the number of elements it covers, so a scalar result
/// is not mistaken for a mask over the single element its backing bitmap has bits for. Hand it to
/// an array through `set_validity_broadcast` — which admits both representations — with
/// [`PlBitmap::into_flat_or_scalar`], or materialize a flat mask with [`PlBitmap::into_bitmap`].
///
/// # Panics
/// Panics if the masks are over a different number of elements.
pub fn combine_validities_and(
    lhs: Option<PlBitmapRef<'_>>,
    rhs: Option<PlBitmapRef<'_>>,
) -> Option<PlBitmap> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => {
            assert_eq!(
                lhs.len(),
                rhs.len(),
                "validity masks cover different lengths"
            );
            let length = lhs.len();
            match (lhs.scalar_value(), rhs.scalar_value()) {
                // Two single bits `and` to a single bit, which covers every element in turn.
                (Some(lhs), Some(rhs)) => Some(PlBitmap::new_scalar(lhs && rhs, length)),
                // A scalar mask that is set everywhere leaves the other one as it is, in whatever
                // representation that one is in; one that is unset everywhere makes the result a
                // single unset bit, whatever the other one holds.
                (Some(true), None) => Some(PlBitmap::from(rhs)),
                (None, Some(true)) => Some(PlBitmap::from(lhs)),
                (Some(false), None) | (None, Some(false)) => {
                    Some(PlBitmap::new_scalar(false, length))
                },
                (None, None) => arrow::compute::utils::combine_validities_and(
                    Some(&lhs.to_flat()),
                    Some(&rhs.to_flat()),
                )
                .map(PlBitmap::from_bitmap),
            }
        },
        (Some(validity), None) | (None, Some(validity)) => Some(PlBitmap::from(validity)),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_scalar_masks_combine_to_a_scalar_one_over_the_same_elements() {
        for (lhs, rhs, expected) in [
            (true, true, true),
            (true, false, false),
            (false, true, false),
            (false, false, false),
        ] {
            let lhs = PlBitmap::new_scalar(lhs, 1_000);
            let rhs = PlBitmap::new_scalar(rhs, 1_000);

            let combined = combine_validities_and(Some(lhs.as_ref()), Some(rhs.as_ref())).unwrap();

            // The single bit is not written out, but the mask still covers every element: it is
            // the length of the inputs that comes back, not the one bit the bitmap holds.
            assert_eq!(combined.len(), 1_000);
            assert!(combined.is_scalar());
            assert_eq!(combined.scalar_value(), Some(expected));
            assert_eq!(combined.set_bits(), if expected { 1_000 } else { 0 });
        }
    }

    #[test]
    fn a_scalar_mask_that_is_set_everywhere_leaves_the_other_one_as_it_is() {
        let flat = PlBitmap::from_iter([true, false, true]);
        let scalar = PlBitmap::new_scalar(true, 3);

        for combined in [
            combine_validities_and(Some(scalar.as_ref()), Some(flat.as_ref())).unwrap(),
            combine_validities_and(Some(flat.as_ref()), Some(scalar.as_ref())).unwrap(),
            // An absent mask is all valid, and does the same.
            combine_validities_and(Some(flat.as_ref()), None).unwrap(),
            combine_validities_and(None, Some(flat.as_ref())).unwrap(),
        ] {
            assert_eq!(combined.len(), 3);
            assert_eq!(combined, flat);
        }
    }

    #[test]
    fn a_scalar_mask_that_is_unset_everywhere_nulls_the_result_out() {
        let flat = PlBitmap::from_iter([true, false, true]);
        let scalar = PlBitmap::new_scalar(false, 3);

        for combined in [
            combine_validities_and(Some(scalar.as_ref()), Some(flat.as_ref())).unwrap(),
            combine_validities_and(Some(flat.as_ref()), Some(scalar.as_ref())).unwrap(),
        ] {
            assert_eq!(combined.len(), 3);
            assert!(combined.is_scalar());
            assert_eq!(combined.scalar_value(), Some(false));
        }
    }

    #[test]
    fn two_flat_masks_are_combined_bit_for_bit() {
        let lhs = PlBitmap::from_iter([true, true, false]);
        let rhs = PlBitmap::from_iter([true, false, false]);

        let combined = combine_validities_and(Some(lhs.as_ref()), Some(rhs.as_ref())).unwrap();

        assert_eq!(combined.len(), 3);
        assert!(combined.is_flat());
        assert_eq!(combined, PlBitmap::from_iter([true, false, false]));
    }

    #[test]
    fn two_absent_masks_stay_absent() {
        assert!(combine_validities_and(None, None).is_none());
    }

    #[test]
    #[should_panic(expected = "validity masks cover different lengths")]
    fn combining_masks_over_different_lengths_panics() {
        let lhs = PlBitmap::new_scalar(true, 3);
        let rhs = PlBitmap::new_scalar(true, 4);

        combine_validities_and(Some(lhs.as_ref()), Some(rhs.as_ref()));
    }

    #[test]
    fn inverting_a_scalar_mask_keeps_it_scalar() {
        let mask = PlBitmap::new_scalar(false, 1_000);

        let inverted = invert(mask.as_ref());

        assert_eq!(inverted.len(), 1);
        assert!(inverted.get_bit(0));
    }
}
