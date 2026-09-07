//! Combining the validity masks of arrays.

use crate::{PlBitmap, PlBitmapRef};

/// The `and` of two masks over the same elements, or `None` if neither has a null.
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

/// The `and` of three masks over the same elements, or `None` if none of them has a null.
pub fn combine_validities_and3(
    first: Option<PlBitmapRef<'_>>,
    second: Option<PlBitmapRef<'_>>,
    third: Option<PlBitmapRef<'_>>,
) -> Option<PlBitmap> {
    let head = combine_validities_and(first, second);
    combine_validities_and(head.as_ref().map(PlBitmap::as_ref), third)
}

/// The `and` of any number of masks over the same elements, or `None` if none of them has a null.
pub fn combine_validities_and_many(masks: &[Option<PlBitmap>]) -> Option<PlBitmap> {
    masks.iter().fold(None, |combined, mask| {
        // Folding pairwise keeps every shortcut `combine_validities_and` takes: a mask that is
        // unset everywhere settles the result on the spot, and one that is set everywhere leaves
        // the running answer in whatever representation it is in.
        combine_validities_and(
            combined.as_ref().map(PlBitmap::as_ref),
            mask.as_ref().map(PlBitmap::as_ref),
        )
    })
}

/// The bits of `mask`, inverted: set where an element is null.
pub fn invert(mask: PlBitmapRef<'_>) -> PlBitmap {
    // The backing bitmap is flat or scalar for the mask's length, and inverting it bit for bit
    // leaves it that way; there is nothing to expand first.
    let (bitmap, length) = mask.into_inner();
    // SAFETY: inverting a bitmap leaves its length, and so its representation, alone.
    unsafe { PlBitmap::new_broadcast_unchecked(!bitmap, length) }
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
    fn many_masks_fold_pairwise_and_keep_a_repeated_bit_repeated() {
        let flat = PlBitmap::from_iter([true, false, true]);
        let ones = PlBitmap::new_scalar(true, 3);

        // Nothing to combine, and nothing that has a null.
        assert_eq!(combine_validities_and_many(&[]), None);
        assert_eq!(combine_validities_and_many(&[None, None]), None);

        // Masks that are set everywhere leave the one that is not alone, as it is.
        let combined =
            combine_validities_and_many(&[Some(ones.clone()), None, Some(flat.clone())]).unwrap();
        assert!(combined.is_flat());
        assert_eq!(combined, flat);

        // One mask that is unset everywhere settles the answer for all of them.
        let combined = combine_validities_and_many(&[
            Some(flat.clone()),
            Some(PlBitmap::new_scalar(false, 3)),
            Some(ones),
        ])
        .unwrap();
        assert!(combined.is_scalar());
        assert_eq!(combined.scalar_value(), Some(false));

        // Masks that each hold one bit per element are combined bit for bit.
        let combined = combine_validities_and_many(&[
            Some(flat),
            Some(PlBitmap::from_iter([true, true, false])),
        ])
        .unwrap();
        assert_eq!(combined, PlBitmap::from_iter([true, false, false]));
    }
}
