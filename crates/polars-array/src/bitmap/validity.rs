//! Combining the validity masks of arrays.

use arrow::bitmap::Bitmap;

use crate::{PlBitmap, PlBitmapRef};

/// The `and` of two masks over the same elements, or `None` if neither has a null.
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
pub fn invert(mask: PlBitmapRef<'_>) -> Bitmap {
    // The backing bitmap is flat or scalar for the mask's length, and inverting it bit for bit
    // leaves it that way; there is nothing to expand first.
    !mask.into_inner().0
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
}
