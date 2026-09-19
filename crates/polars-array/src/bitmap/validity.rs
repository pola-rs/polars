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
                (Some(lhs), Some(rhs)) => Some(PlBitmap::new_scalar(lhs && rhs, length)),
                (Some(true), None) => Some(PlBitmap::from(rhs)),
                (None, Some(true)) => Some(PlBitmap::from(lhs)),
                (Some(false), None) | (None, Some(false)) => {
                    Some(PlBitmap::new_scalar(false, length))
                },
                (None, None) => polars_arrow::compute::utils::combine_validities_and(
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
        combine_validities_and(
            combined.as_ref().map(PlBitmap::as_ref),
            mask.as_ref().map(PlBitmap::as_ref),
        )
    })
}

/// The bits of `mask`, inverted: set where an element is null.
pub fn invert(mask: PlBitmapRef<'_>) -> PlBitmap {
    let (bitmap, length) = mask.into_inner();
    // SAFETY: inverting a bitmap leaves its length, and so its representation, alone.
    unsafe { PlBitmap::new_broadcast_unchecked(!bitmap, length) }
}
