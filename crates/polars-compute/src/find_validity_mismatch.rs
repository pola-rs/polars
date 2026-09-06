//! Finding the elements two chunks disagree about being null.
//!
//! This is what names the rows a strict cast failed on: the cast turned a value it could not
//! convert into a null, so the elements the output is null at and the input is not are the ones it
//! could not convert. The answer is given recursively — a disagreement under a nested element is
//! reported at the element above it — and it is the validity masks alone that are read, never a
//! value.
//!
//! Comparing masks is where the representation pays: a mask that says the same of every element is
//! one bit against the other's, so two chunks that are wholly valid, or wholly null, agree or
//! disagree in `O(1)` however many elements they hold. Walking *into* a nested chunk is the other
//! way around — it maps a value back onto the element above it by position, and so needs one slot
//! per element on both sides — so a chunk that is not laid out that way is written out first. That
//! only ever happens once a cast has already failed.
//!
//! This procedure requires that
//! - Nulls are propagated recursively
//! - Lists to be
//!     - trimmed to normalized offsets
//!     - have the same number of child elements below each element (even nulls)

use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;
use polars_array::{
    PlArray, PlArrayType, PlBitmapRef, PlFixedSizeListArray, PlListArray, PlStructArray,
};
use polars_utils::IdxSize;

use crate::cast::CastOptionsImpl;
use crate::nesting::{covered_range, downcast};

/// Appends the indices of the elements `left` and `right` disagree about being null.
///
/// # Panics
/// Panics unless `left` and `right` hold the same number of elements.
pub fn find_validity_mismatch(left: &dyn PlArray, right: &dyn PlArray, idxs: &mut Vec<IdxSize>) {
    assert_eq!(left.len(), right.len());

    // Handle the top-level.
    //
    // NOTE: This is done always, even if left and right have different nestings. This is
    // intentional and needed.
    let original_idxs_length = idxs.len();
    extend_mismatches(idxs, left.len(), left.validity(), right.validity());

    let pre_nesting_length = idxs.len();
    match (left.array_type(), right.array_type()) {
        (PlArrayType::Struct, PlArrayType::Struct) => {
            let left: &PlStructArray = downcast(left);
            let right: &PlStructArray = downcast(right);

            assert_eq!(left.num_fields(), right.num_fields());
            for (left, right) in left.fields().iter().zip(right.fields()) {
                find_validity_mismatch(&**left, &**right, idxs);
            }
        },
        (PlArrayType::List, PlArrayType::List) => {
            find_validity_mismatch_list_list(downcast(left), downcast(right), idxs)
        },
        (PlArrayType::FixedSizeList, PlArrayType::FixedSizeList) => {
            find_validity_mismatch_fsl_fsl(downcast(left), downcast(right), idxs)
        },
        (PlArrayType::List, PlArrayType::FixedSizeList) => {
            find_validity_mismatch_list_fsl(downcast(left), downcast(right), idxs)
        },
        (PlArrayType::FixedSizeList, PlArrayType::List) => {
            find_validity_mismatch_list_fsl(downcast(right), downcast(left), idxs)
        },
        _ => {},
    }

    if pre_nesting_length == idxs.len() {
        return;
    }
    idxs[original_idxs_length..].sort_unstable();
}

/// Appends the indices at which two validity masks over `length` elements disagree, reading an
/// absent mask as one that says every element is valid.
fn extend_mismatches(
    idxs: &mut Vec<IdxSize>,
    length: usize,
    left: Option<PlBitmapRef<'_>>,
    right: Option<PlBitmapRef<'_>>,
) {
    match (left, right) {
        (None, None) => return,
        // One side says every element is valid, and the other holds a mask that says so too.
        (Some(mask), None) | (None, Some(mask)) if mask.unset_bits() == 0 => return,
        _ => {},
    }

    // A mask that says the same of every element — or that is not there at all — is a single bit
    // against the other's. Two of them either agree about every element or disagree about every
    // one, and neither is ever read.
    let scalar =
        |mask: Option<PlBitmapRef<'_>>| mask.map_or(Some(true), |mask| mask.scalar_value());
    if let (Some(left), Some(right)) = (scalar(left), scalar(right)) {
        if left != right {
            idxs.extend(0..length as IdxSize);
        }
        return;
    }

    // At least one of the two holds one bit per element, which is the side the answer is read off.
    // An absent mask is the one that says every element is valid, as is a scalar mask of a set
    // bit: either way the other side says the same thing of every element, so the elements the two
    // disagree about are the ones the flat mask says the opposite of — which is that mask itself,
    // or its inverse, and never a mask written out from a single bit.
    let mismatches = match (left, right) {
        (Some(left), Some(right)) => match (left.flat_bitmap(), right.flat_bitmap()) {
            (Some(left), Some(right)) => arrow::bitmap::xor(left, right),
            (Some(flat), None) => disagreements_with(flat, right.scalar_value().unwrap()),
            (None, Some(flat)) => disagreements_with(flat, left.scalar_value().unwrap()),
            (None, None) => unreachable!("two scalar masks are answered for above"),
        },
        (Some(mask), None) | (None, Some(mask)) => disagreements_with(
            mask.flat_bitmap()
                .expect("a scalar mask against an absent one is answered for above"),
            true,
        ),
        (None, None) => unreachable!("two absent masks are both a single bit"),
    };

    idxs.extend(mismatches.true_idx_iter().map(|i| i as IdxSize));
}

/// The elements `flat` disagrees about with a side that says `valid` of every one of them.
///
/// Where that side says they are all valid the disagreements are the elements `flat` says are
/// null, which has to be written out; where it says they are all null they are the bits `flat`
/// already has set, and the mask is handed back as it stands.
fn disagreements_with(flat: &Bitmap, valid: bool) -> Bitmap {
    if valid { !flat } else { flat.clone() }
}

/// Reports a disagreement under an element of `left` at that element.
fn find_validity_mismatch_list_list(
    left: &PlListArray,
    right: &PlListArray,
    idxs: &mut Vec<IdxSize>,
) {
    // The values are read against each other one slot per value, and the range every element covers
    // is read off `left`; an array whose elements share one range holds neither.
    let left = left.to_flat();
    let left = left.as_array();
    let right = right.to_flat();

    let mut nested_idxs = Vec::new();
    find_validity_mismatch(left.values(), right.as_array().values(), &mut nested_idxs);

    if nested_idxs.is_empty() {
        return;
    }

    assert_eq!(covered_range(left), 0..left.values().len());

    // @TODO: Optimize. This is only used on the error path so it is find, right?
    let mut j = 0;
    for i in 0..left.len() {
        // SAFETY: `i` is an index of `left`.
        let end = unsafe { left.value_range_unchecked(i) }.end;

        if j < nested_idxs.len() && (nested_idxs[j] as usize) < end {
            idxs.push(i as IdxSize);
            j += 1;

            // Loop over remaining items in same element.
            while j < nested_idxs.len() && (nested_idxs[j] as usize) < end {
                j += 1;
            }
        }

        if j == nested_idxs.len() {
            break;
        }
    }
}

/// Reports a disagreement under an element of two arrays of the same width at that element.
fn find_validity_mismatch_fsl_fsl(
    left: &PlFixedSizeListArray,
    right: &PlFixedSizeListArray,
    idxs: &mut Vec<IdxSize>,
) {
    assert_eq!(left.width(), right.width());
    let width = left.width();

    // A value is mapped back onto the element above it by its position, which needs the values of
    // both sides laid out one list per element.
    let left = left.to_flat();
    let right = right.to_flat();

    find_validity_mismatch_nested(
        left.as_array().values(),
        right.as_array().values(),
        width,
        idxs,
    )
}

/// Reports a disagreement between two values arrays of `size` values per element at the element
/// above it, naming each such element once.
fn find_validity_mismatch_nested(
    left: &dyn PlArray,
    right: &dyn PlArray,
    size: usize,
    idxs: &mut Vec<IdxSize>,
) {
    assert_eq!(left.len(), right.len());
    let start_length = idxs.len();
    find_validity_mismatch(left, right, idxs);
    if idxs.len() > start_length {
        let mut offset = 0;
        idxs[start_length] /= size as IdxSize;
        for i in start_length + 1..idxs.len() {
            idxs[i - offset] = idxs[i] / size as IdxSize;

            if idxs[i - offset] == idxs[i - offset - 1] {
                offset += 1;
            }
        }
        idxs.truncate(idxs.len() - offset);
    }
}

/// Reports a disagreement between a list array and a fixed size list array of the same widths at
/// the element it sits under.
fn find_validity_mismatch_list_fsl(
    left: &PlListArray,
    right: &PlFixedSizeListArray,
    idxs: &mut Vec<IdxSize>,
) {
    let right = right.to_flat();
    let right = right.as_array();

    if left.validity().is_none() && right.validity().is_none() {
        let left = left.to_flat();

        find_validity_mismatch_nested(
            left.as_array().values(),
            right.values(),
            right.width(),
            idxs,
        );
        return;
    }

    // The lists of a null element hold no values of their own, so lining the two sides up value for
    // value means filling those in — which is what the cast to a fixed width does. This only runs
    // once a cast has already failed.
    let from_type = crate::cast::pl_array::physical_dtype(left);
    let ArrowDataType::LargeList(field) = &from_type else {
        unreachable!("a list array of this crate reads as a large list");
    };
    let to_type = ArrowDataType::FixedSizeList(field.clone(), right.width());

    let left = crate::cast::cast_chunk_from(left, &from_type, &to_type, CastOptionsImpl::default())
        .unwrap();
    let left: &PlFixedSizeListArray = downcast(&*left);

    find_validity_mismatch_nested(left.values(), right.values(), right.width(), idxs)
}

#[cfg(test)]
mod tests {
    use polars_array::{PlBitmap, PlPrimitiveArray};

    use super::*;

    const LENGTH: usize = 6;
    /// The elements a flat mask marks valid, which is what every crossing below is read against.
    const VALID: [bool; LENGTH] = [true, false, true, true, false, true];

    fn flat() -> PlBitmap {
        PlBitmap::from_bitmap(VALID.into_iter().collect())
    }

    fn mismatches(left: Option<PlBitmap>, right: Option<PlBitmap>) -> Vec<IdxSize> {
        let mut idxs = Vec::new();
        extend_mismatches(
            &mut idxs,
            LENGTH,
            left.as_ref().map(PlBitmap::as_ref),
            right.as_ref().map(PlBitmap::as_ref),
        );
        idxs
    }

    /// A mask in either representation has to name the same elements as the same mask written out
    /// one bit per element, which is what the flat-against-flat path answers.
    #[test]
    fn a_scalar_mask_disagrees_where_it_is_written_out_to() {
        for value in [false, true] {
            let scalar = PlBitmap::new_scalar(value, LENGTH);
            let written_out = PlBitmap::from_bitmap(std::iter::repeat_n(value, LENGTH).collect());

            assert_eq!(
                mismatches(Some(flat()), Some(scalar.clone())),
                mismatches(Some(flat()), Some(written_out.clone())),
            );
            assert_eq!(
                mismatches(Some(scalar), Some(flat())),
                mismatches(Some(written_out), Some(flat())),
            );
        }
    }

    /// An absent mask says every element is valid, and so names the elements a flat mask says are
    /// null — the same ones a scalar mask of a set bit does.
    #[test]
    fn an_absent_mask_disagrees_where_the_other_says_null() {
        let nulls: Vec<IdxSize> = (0..LENGTH as IdxSize)
            .filter(|i| !VALID[*i as usize])
            .collect();

        assert_eq!(mismatches(Some(flat()), None), nulls);
        assert_eq!(mismatches(None, Some(flat())), nulls);
        assert_eq!(
            mismatches(Some(flat()), Some(PlBitmap::new_scalar(true, LENGTH))),
            nulls,
        );
    }

    /// Two masks that each say the same of every element agree or disagree about all of them at
    /// once, whichever way round they are and whether the mask is absent or a single bit.
    #[test]
    fn two_scalar_masks_are_answered_for_at_once() {
        let all: Vec<IdxSize> = (0..LENGTH as IdxSize).collect();
        let valid = || Some(PlBitmap::new_scalar(true, LENGTH));
        let null = || Some(PlBitmap::new_scalar(false, LENGTH));

        assert!(mismatches(valid(), valid()).is_empty());
        assert!(mismatches(None, valid()).is_empty());
        assert!(mismatches(null(), null()).is_empty());
        assert_eq!(mismatches(valid(), null()), all);
        assert_eq!(mismatches(null(), valid()), all);
        assert_eq!(mismatches(None, null()), all);
        assert_eq!(mismatches(null(), None), all);
    }

    /// The whole procedure, not just the mask comparison: a scalar chunk has to report the same
    /// elements as the same chunk laid out one slot per element.
    #[test]
    fn a_scalar_chunk_reports_what_its_written_out_form_does() {
        let scalar = PlPrimitiveArray::new_scalar(7i32, LENGTH)
            .with_validity(Some(PlBitmap::new_scalar(false, LENGTH)));
        let flat = PlPrimitiveArray::from_vec(vec![7i32; LENGTH]).with_validity(Some(flat()));

        let mut from_scalar = Vec::new();
        find_validity_mismatch(&scalar, &flat, &mut from_scalar);

        let mut from_written_out = Vec::new();
        let written_out = scalar.to_flat().into_owned().into_array();
        find_validity_mismatch(&written_out, &flat, &mut from_written_out);

        assert_eq!(from_scalar, from_written_out);
        assert_eq!(from_scalar, vec![0, 2, 3, 5]);
    }
}
