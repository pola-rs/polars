//! The pieces the kernels that walk a nested chunk recursively share.
//!
//! Every one of them rebuilds a chunk around new children, and the point of doing that natively is
//! that the chunk keeps the representation it came in: a rebuilt array is handed the very buffers
//! the old one was backed by, so an array whose every element is the same list stays one list long.
//! That is what the `*_with_*` functions here are for — each of them branches on the representation
//! of the array it rebuilds, and hands the children to the constructor that keeps it.

use std::ops::Range;

use polars_array::{PlArray, PlBitmap, PlFixedSizeListArray, PlListArray, PlStructArray};

/// Downcasts a chunk whose [`PlArrayType`](polars_array::PlArrayType) says which array it is.
pub(crate) fn downcast<A: PlArray>(array: &dyn PlArray) -> &A {
    array
        .as_any()
        .downcast_ref()
        .expect("the array type of a chunk names the array it is")
}

/// The range of `array.values()` the elements of `array` cover, taken together.
///
/// The offsets are non-decreasing and the range of one element ends where the next one starts, so
/// the elements tile this range: it is the range of the first element joined to that of the last.
pub(crate) fn covered_range(array: &PlListArray) -> Range<usize> {
    let Some(last) = array.len().checked_sub(1) else {
        // An array of no elements covers nothing, and holds the one offset it starts at.
        let start = array
            .flat_offsets()
            .expect("the offsets of an empty array are flat")[0] as usize;
        return start..start;
    };

    array.value_range(0).start..array.value_range(last).end
}

/// Rebuilds `array` around `values`, which stand for the same values the old ones did.
///
/// # Safety
/// `values` must reach at least as far as the offsets of `array` do.
pub(crate) unsafe fn list_with_values(
    array: &PlListArray,
    values: Box<dyn PlArray>,
) -> PlListArray {
    // The offsets and the mask are the ones `array` is backed by, in the representation it is in;
    // cloning them out of a clone of it copies nothing.
    let offsets_are_flat = array.offsets_are_flat();
    let (_, offsets, length, validity) = array.clone().into_inner();

    unsafe {
        if offsets_are_flat {
            PlListArray::new_unchecked(values, offsets, length, validity)
        } else {
            PlListArray::new_broadcast_unchecked(values, offsets, length, validity)
        }
    }
}

/// Rebuilds `array` around `values`, which stand for the same values the old ones did.
///
/// # Safety
/// `values` must hold exactly as many values as the ones of `array`.
pub(crate) unsafe fn fsl_with_values(
    array: &PlFixedSizeListArray,
    values: Box<dyn PlArray>,
) -> PlFixedSizeListArray {
    let values_are_flat = array.values_are_flat();
    let (_, width, length, validity) = array.clone().into_inner();

    unsafe {
        if values_are_flat {
            PlFixedSizeListArray::new_unchecked(values, width, length, validity)
        } else {
            PlFixedSizeListArray::new_broadcast_unchecked(values, width, length, validity)
        }
    }
}

/// Rebuilds `array` around `fields`, which stand for the same elements the old ones did.
///
/// # Safety
/// Every field must hold exactly `array.len()` elements.
pub(crate) unsafe fn struct_with_fields(
    array: &PlStructArray,
    fields: Vec<Box<dyn PlArray>>,
) -> PlStructArray {
    let validity_is_scalar = array.validity_is_scalar();
    let (_, length, validity) = array.clone().into_inner();

    unsafe {
        if validity_is_scalar {
            PlStructArray::new_broadcast_unchecked(fields, length, validity)
        } else {
            PlStructArray::new_unchecked(fields, length, validity)
        }
    }
}

/// Returns `array` with its validity mask replaced by `validity`, which keeps the representation it
/// is in: a mask that stands for a single bit is not written out one bit per element to be set.
///
/// # Panics
/// Panics unless `validity` covers exactly `array.len()` elements.
pub(crate) fn with_pl_validity(array: &dyn PlArray, validity: PlBitmap) -> Box<dyn PlArray> {
    assert_eq!(
        array.len(),
        validity.len(),
        "a validity mask covers exactly the elements of the array it is set on",
    );

    array.with_validity_broadcast(Some(validity.into_flat_or_scalar()))
}
