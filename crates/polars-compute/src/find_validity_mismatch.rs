//! Finding the elements two chunks disagree about being null.

use arrow::bitmap::Bitmap;
use polars_array::{
    PlArray, PlArrayType, PlBitmapRef, PlFixedSizeListArray, PlListArray, PlStructArray,
};
use polars_utils::IdxSize;

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

/// Appends the indices at which two validity masks over `length` elements disagree.
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
fn disagreements_with(flat: &Bitmap, valid: bool) -> Bitmap {
    if valid { !flat } else { flat.clone() }
}

/// Reports a disagreement under an element of `left` at that element.
fn find_validity_mismatch_list_list(
    left: &PlListArray,
    right: &PlListArray,
    idxs: &mut Vec<IdxSize>,
) {
    // Both sides repeat the one range every element of them reads, so the two lists are read
    // against each other once: either they agree about every value, and no element is reported, or
    // they disagree somewhere every element reads, and all of them are — neither side is written
    // out one list per element to say so.
    if let (Some(l), Some(r)) = (left.scalar_offsets(), right.scalar_offsets())
        && l.len() == r.len()
    {
        let mut nested_idxs = Vec::new();
        find_validity_mismatch(
            &*left.values().sliced(l.start, l.len()),
            &*right.values().sliced(r.start, r.len()),
            &mut nested_idxs,
        );

        if !nested_idxs.is_empty() {
            idxs.extend(0..left.len() as IdxSize);
        }
        return;
    }

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

    // Both sides hold the one list every element of them reads, so the two lists are read against
    // each other once: either they agree about every value, and no element is reported, or they
    // disagree somewhere every element reads, and all of them are — neither side is written out
    // one list per element to say so.
    if left.values_are_scalar() && right.values_are_scalar() {
        let mut nested_idxs = Vec::new();
        find_validity_mismatch(left.values(), right.values(), &mut nested_idxs);

        if !nested_idxs.is_empty() {
            idxs.extend(0..left.len() as IdxSize);
        }
        return;
    }

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

/// Reports a disagreement between two values arrays of `size` values per element, once each.
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

/// Reports a disagreement between a list array and a fixed size list array of the same widths.
fn find_validity_mismatch_list_fsl(
    left: &PlListArray,
    right: &PlFixedSizeListArray,
    idxs: &mut Vec<IdxSize>,
) {
    // As in the two same-shape pairs above: both sides hold the one list every element of them
    // reads, so the two lists are read against each other once. Either they agree about every
    // value, and no element is reported, or they disagree somewhere every element reads, and all
    // of them are — neither side is written out one list per element to say so.
    if let Some(range) = left.scalar_offsets()
        && right.values_are_scalar()
        && range.len() == right.width()
    {
        let mut nested_idxs = Vec::new();
        find_validity_mismatch(
            &*left.values().sliced(range.start, range.len()),
            right.values(),
            &mut nested_idxs,
        );

        if !nested_idxs.is_empty() {
            idxs.extend(0..left.len() as IdxSize);
        }
        return;
    }

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
    let left =
        crate::cast::list_to_fixed_size_list(left, right.width(), |values| Ok(values.to_boxed()))
            .unwrap();

    find_validity_mismatch_nested(left.values(), right.values(), right.width(), idxs)
}
