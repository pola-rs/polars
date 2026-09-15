//! The pieces the kernels that walk a nested chunk recursively share.

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
    // The offsets and the mask are the ones `array` is backed by, in the representation each of
    // them is in; cloning them out of a clone of it copies nothing.
    let offsets_are_flat = array.offsets_are_flat();
    let validity = array.validity().map(PlBitmap::from);
    let (_, offsets, length, _) = array.clone().into_inner();

    // The constructor is picked on the offsets, which is now the only axis it decides: a mask
    // carries its own representation, so it goes back on afterwards whichever one it is in.
    let out = unsafe {
        if offsets_are_flat {
            PlListArray::new_unchecked(values, offsets, length, None)
        } else {
            PlListArray::new_broadcast_unchecked(values, offsets, length, None)
        }
    };

    out.with_validity(validity)
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
    let validity = array.validity().map(PlBitmap::from);
    let (_, width, length, _) = array.clone().into_inner();

    // As in `list_with_values`: the constructor decides the values, the mask decides itself.
    let out = unsafe {
        if values_are_flat {
            PlFixedSizeListArray::new_unchecked(values, width, length, None)
        } else {
            PlFixedSizeListArray::new_broadcast_unchecked(values, width, length, None)
        }
    };

    out.with_validity(validity)
}

/// Rebuilds `array` around `fields`, which stand for the same elements the old ones did.
///
/// # Safety
/// Every field must hold exactly `array.len()` elements.
pub(crate) unsafe fn struct_with_fields(
    array: &PlStructArray,
    fields: Vec<Box<dyn PlArray>>,
) -> PlStructArray {
    let validity = array.validity().map(PlBitmap::from);

    unsafe { PlStructArray::new_unchecked(fields, array.len(), validity) }
}

/// Returns `array` with its validity mask replaced by `validity`, keeping its representation.
pub(crate) fn with_pl_validity(array: &dyn PlArray, validity: PlBitmap) -> Box<dyn PlArray> {
    assert_eq!(
        array.len(),
        validity.len(),
        "a validity mask covers exactly the elements of the array it is set on",
    );

    array.with_validity(Some(validity))
}
