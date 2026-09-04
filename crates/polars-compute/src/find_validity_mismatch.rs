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

use arrow::array::Array;
use arrow::datatypes::ArrowDataType;
use polars_array::arrow::bridge::chunk_to_arrow;
use polars_array::arrow::import::from_arrow;
use polars_array::{
    PlArray, PlArrayType, PlBitmapRef, PlFixedSizeListArray, PlListArray, PlStructArray,
};
use polars_utils::IdxSize;

use crate::cast::CastOptionsImpl;
use crate::nesting::{covered_range, downcast, fsl_values};

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

    // At least one of the two holds one bit per element, so the answer is given bit for bit; the
    // other crosses over to one bit per element to be read against it.
    let mismatches = match (left, right) {
        (Some(left), Some(right)) => arrow::bitmap::xor(&left.to_flat(), &right.to_flat()),
        // An absent mask is the one that says every element is valid, so the elements the two
        // disagree about are the ones the mask that is there says are null.
        (Some(mask), None) | (None, Some(mask)) => !&*mask.to_flat(),
        (None, None) => unreachable!("two absent masks are both a single bit"),
    };

    idxs.extend(mismatches.true_idx_iter().map(|i| i as IdxSize));
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
        fsl_values(left.as_array()),
        fsl_values(right.as_array()),
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
            fsl_values(right),
            right.width(),
            idxs,
        );
        return;
    }

    // The lists of a null element hold no values of their own, so lining the two sides up value for
    // value means filling those in — which is what the cast to a fixed width does. That kernel is
    // the Arrow one, so the chunk crosses over; this only runs once a cast has already failed.
    let arrow = chunk_to_arrow::<PlListArray>(left);
    let ArrowDataType::LargeList(field) = arrow.dtype() else {
        unreachable!("a list array of this crate crosses over as a large list");
    };

    let left = crate::cast::cast_list_to_fixed_size_list(
        &arrow,
        field,
        right.width(),
        CastOptionsImpl::default(),
    )
    .unwrap();
    let left = from_arrow(left.values().as_ref());

    find_validity_mismatch_nested(&*left, fsl_values(right), right.width(), idxs)
}
