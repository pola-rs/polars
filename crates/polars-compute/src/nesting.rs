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

    // The constructor is picked on the offsets, which is the axis it decides — but it constrains
    // the mask as well, and the two representations are independent: a scalar mask over flat
    // offsets, and a flat mask over scalar ones, are both arrays this crate hands out. So each
    // constructor is handed no mask at all, and the mask goes back on through the setter that
    // takes it in either representation.
    let out = unsafe {
        if offsets_are_flat {
            PlListArray::new_unchecked(values, offsets, length, None)
        } else {
            PlListArray::new_broadcast_unchecked(values, offsets, length, None)
        }
    };

    out.with_validity_broadcast(validity)
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

    // As in `list_with_values`: the values and the mask are in representations of their own, so
    // the mask goes back on after the constructor the values pick.
    let out = unsafe {
        if values_are_flat {
            PlFixedSizeListArray::new_unchecked(values, width, length, None)
        } else {
            PlFixedSizeListArray::new_broadcast_unchecked(values, width, length, None)
        }
    };

    out.with_validity_broadcast(validity)
}

/// Rebuilds `array` around `fields`, which stand for the same elements the old ones did.
///
/// Unlike the two above, this one branches on the mask itself: a struct has no values buffer of its
/// own, so the mask is its only representation, and both constructors ask the same of the fields.
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

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;
    use polars_array::PlPrimitiveArray;

    use super::*;

    fn values(values: Vec<i32>) -> Box<dyn PlArray> {
        Box::new(PlPrimitiveArray::from_vec(values))
    }

    /// The offsets and the mask of a list array are each flat or scalar independently of the
    /// other, so rebuilding one has to keep whichever representation each is in: the constructors
    /// constrain both axes at once, and picking one on the offsets alone gets the mask wrong.
    #[test]
    fn list_with_values_keeps_a_flat_mask_over_scalar_offsets() {
        let array = PlListArray::new_scalar(values(vec![1, 2]), 4)
            .with_validity_broadcast(Some(Bitmap::from_iter([true, false, true, false])));
        assert!(!array.offsets_are_flat());
        assert!(array.validity().unwrap().is_flat());

        let out = unsafe { list_with_values(&array, values(vec![7, 8])) };

        assert!(!out.offsets_are_flat());
        assert!(out.validity().unwrap().is_flat());
        assert_eq!((out.len(), out.null_count()), (4, 2));
    }

    /// The other way around, which is the pairing `propagate_nulls` reaches: a level whose every
    /// element is null hands one unset bit down onto values that are laid out one per element.
    #[test]
    fn list_with_values_keeps_a_scalar_mask_over_flat_offsets() {
        let array = PlListArray::from_offsets(values(vec![1, 2]), vec![0u64, 1, 2].into())
            .with_validity_broadcast(Some(Bitmap::new_zeroed(1)));
        assert!(array.offsets_are_flat());
        assert!(array.validity().unwrap().is_scalar());

        let out = unsafe { list_with_values(&array, values(vec![7, 8])) };

        assert!(out.offsets_are_flat());
        assert!(out.validity().unwrap().is_scalar());
        assert_eq!((out.len(), out.null_count()), (2, 2));
    }

    /// As for a list array: the values and the mask of a fixed size list array are just as
    /// independent of one another.
    #[test]
    fn fsl_with_values_keeps_a_flat_mask_over_scalar_values() {
        let array = PlFixedSizeListArray::new_scalar(values(vec![1, 2]), 4)
            .with_validity_broadcast(Some(Bitmap::from_iter([true, false, true, false])));
        assert!(!array.values_are_flat());
        assert!(array.validity().unwrap().is_flat());

        let out = unsafe { fsl_with_values(&array, values(vec![7, 8])) };

        assert!(!out.values_are_flat());
        assert!(out.validity().unwrap().is_flat());
        assert_eq!((out.len(), out.null_count()), (4, 2));
    }

    /// The pairing a fully null struct hands down to a fixed size list field in `propagate_nulls`,
    /// and the one the proptest there found: flat values under a mask of a single bit.
    #[test]
    fn fsl_with_values_keeps_a_scalar_mask_over_flat_values() {
        let array = PlFixedSizeListArray::from_values(values(vec![1, 2, 3, 4]), 2)
            .with_validity_broadcast(Some(Bitmap::new_zeroed(1)));
        assert!(array.values_are_flat());
        assert!(array.validity().unwrap().is_scalar());

        let out = unsafe { fsl_with_values(&array, values(vec![7, 8, 9, 10])) };

        assert!(out.values_are_flat());
        assert!(out.validity().unwrap().is_scalar());
        assert_eq!((out.len(), out.null_count()), (2, 2));
    }
}
