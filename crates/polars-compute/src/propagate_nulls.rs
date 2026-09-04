//! Pushing the nulls of a nested chunk down onto the values under them.
//!
//! A null element of a list, a fixed size list or a struct still has values under it — the ones its
//! offsets cover, or the ones its fields hold at that index — and nothing says what they are. The
//! kernels that walk those values without walking the level above them, the row encoding among
//! them, need those values to be null too, which is what this module writes.
//!
//! The nulls travel down through the validity masks alone, so a chunk keeps the representation it
//! came in wherever the answer can be given in it. A level whose every element is null pushes a
//! single unset bit down onto its values, in `O(1)`, however many values there are; and a level
//! whose values are already null under every null element is handed back untouched.
//!
//! What cannot be answered in the scalar representation is a level whose elements share one range
//! of the values while only some of them are null: the values under the null elements are the very
//! values under the valid ones, and no mask over them nulls the former alone. Such a level is
//! written out one range per element first, which is the only place here that materializes
//! anything.

use std::ops::Range;

use arrow::bitmap::BitmapBuilder;
use arrow::bitmap::bitmask::BitMask;
use polars_array::bitmap::combine_validities_and;
use polars_array::{
    ArrayRepr, PlArray, PlArrayType, PlBitmap, PlBitmapRef, PlFixedSizeListArray, PlListArray,
    PlStructArray,
};

use crate::nesting::{
    covered_range, downcast, fsl_with_values, list_with_values, struct_with_fields,
    with_pl_validity,
};

/// Pushes the nulls of `array` down onto the values under them, recursively.
///
/// Returns `None` if no level of `array` had a null to push down, in which case `array` is already
/// the answer.
pub fn propagate_nulls(array: &dyn PlArray) -> Option<Box<dyn PlArray>> {
    match array.array_type() {
        PlArrayType::List => {
            propagate_nulls_list(downcast(array)).map(|array| Box::new(array) as _)
        },
        PlArrayType::FixedSizeList => {
            propagate_nulls_fsl(downcast(array)).map(|array| Box::new(array) as _)
        },
        PlArrayType::Struct => {
            propagate_nulls_struct(downcast(array)).map(|array| Box::new(array) as _)
        },
        // A leaf has no values under its elements for a null to reach.
        _ => None,
    }
}

/// Pushes the nulls of `array` down onto the values its null elements cover.
pub fn propagate_nulls_list(array: &PlListArray) -> Option<PlListArray> {
    let values = array.values();

    let Some(validity) = nulls(array.validity()) else {
        // No element is null, so nothing is pushed down here; only a deeper level can still change.
        let values = propagate_nulls(values)?;

        // SAFETY: pushing nulls down leaves the values as many as they were.
        return Some(unsafe { list_with_values(array, values) });
    };

    let child = if validity.unset_bits() == array.len() {
        // Every element is null, so every value they cover sits under a null. The elements tile the
        // range they cover between them, whatever representation the offsets are in, so that one
        // range is every value the nulls reach.
        unset_ranges(values, || std::iter::once(covered_range(array)))
    } else if array.offsets_are_flat() {
        // A mask with some but not every bit unset holds one bit per element.
        let nulls = !validity
            .flat_bitmap()
            .expect("a mask with some but not every bit unset is flat");

        unset_ranges(values, || {
            // SAFETY: the mask holds one bit per element, so every index it names is in bounds.
            nulls
                .true_idx_iter()
                .map(|i| unsafe { array.value_range_unchecked(i) })
        })
    } else if set_bits_in(values.validity(), covered_range(array)) == 0 {
        // The elements share one range, and no value in it is valid: every value under a null
        // element is already null, so the shared range never has to be written out.
        None
    } else {
        // The elements share one range holding a value the nulls have to reach, and the values
        // under the null elements are the very values under the valid ones. Only writing the array
        // out one range per element separates them.
        let flat = array.to_flat();
        let flat = flat.as_array();

        return Some(propagate_nulls_list(flat).unwrap_or_else(|| flat.clone()));
    };

    let values = match child {
        Some(child) => {
            let values = with_pl_validity(values, child);
            propagate_nulls(&*values).unwrap_or(values)
        },
        // Every value under a null element is already null; only a deeper level can still change.
        None => propagate_nulls(values)?,
    };

    // SAFETY: setting a mask leaves the values as many as they were.
    Some(unsafe { list_with_values(array, values) })
}

/// Pushes the nulls of `array` down onto the values its null elements cover.
pub fn propagate_nulls_fsl(array: &PlFixedSizeListArray) -> Option<PlFixedSizeListArray> {
    let values = array.values();

    let Some(validity) = nulls(array.validity()) else {
        let values = propagate_nulls(values)?;

        // SAFETY: pushing nulls down leaves the values as many as they were.
        return Some(unsafe { fsl_with_values(array, values) });
    };

    let width = array.width();
    let child = if validity.unset_bits() == array.len() {
        // Every element is null, so every value sits under a null — whether the values hold the one
        // list every element reads or one list per element, they are all of them covered.
        unset_ranges(values, || std::iter::once(0..values.len()))
    } else if array.values_are_flat() {
        let nulls = !validity
            .flat_bitmap()
            .expect("a mask with some but not every bit unset is flat");

        unset_ranges(values, || {
            nulls.true_idx_iter().map(|i| i * width..(i + 1) * width)
        })
    } else if set_bits_in(values.validity(), 0..values.len()) == 0 {
        // The elements read one list, and no value of it is valid: every value under a null element
        // is already null, so the list never has to be written out.
        None
    } else {
        // The elements read one list holding a value the nulls have to reach, which is the very
        // list the valid elements read. Only writing it out once per element separates them.
        let flat = array.to_flat();
        let flat = flat.as_array();

        return Some(propagate_nulls_fsl(flat).unwrap_or_else(|| flat.clone()));
    };

    let values = match child {
        Some(child) => {
            let values = with_pl_validity(values, child);
            propagate_nulls(&*values).unwrap_or(values)
        },
        None => propagate_nulls(values)?,
    };

    // SAFETY: setting a mask leaves the values as many as they were.
    Some(unsafe { fsl_with_values(array, values) })
}

/// Pushes the nulls of `array` down onto the value every field holds under them.
pub fn propagate_nulls_struct(array: &PlStructArray) -> Option<PlStructArray> {
    let validity = nulls(array.validity());

    let mut changed = false;
    let fields = array
        .fields()
        .iter()
        .map(|field| match propagate_into_field(&**field, validity) {
            Some(field) => {
                changed = true;
                field
            },
            None => field.clone(),
        })
        .collect();

    if !changed {
        return None;
    }

    // SAFETY: taking on a mask leaves a field as long as it was, so each still holds one element
    // per element of the struct.
    Some(unsafe { struct_with_fields(array, fields) })
}

/// Pushes `validity`, the nulls of the struct above `field`, down onto it, and its own nulls down
/// in turn.
fn propagate_into_field(
    field: &dyn PlArray,
    validity: Option<PlBitmapRef<'_>>,
) -> Option<Box<dyn PlArray>> {
    let Some(validity) = validity else {
        return propagate_nulls(field);
    };

    // A field takes the nulls of the struct on top of its own. Combining the two masks keeps the
    // representation they answer in: a struct whose every element is null hands its one unset bit
    // to every field, in `O(1)`.
    let combined = combine_validities_and(field.validity(), Some(validity))
        .expect("two masks always combine into one");

    // The field is already null wherever the struct is, so the mask it holds is the combined one.
    if field.validity().is_some_and(|old| combined == old) {
        return propagate_nulls(field);
    }

    let field = with_pl_validity(field, combined);
    Some(propagate_nulls(&*field).unwrap_or(field))
}

/// The validity mask of a chunk, if any element of it is null.
fn nulls(validity: Option<PlBitmapRef<'_>>) -> Option<PlBitmapRef<'_>> {
    validity.filter(|validity| validity.unset_bits() > 0)
}

/// The mask `values` takes on once a null is pushed down onto every value in `ranges`, or `None` if
/// every one of those values is already null.
///
/// `ranges` hands out ranges within `values` that are ordered and do not overlap, which the ranges
/// the elements of a list array cover are and do not.
fn unset_ranges<I, F>(values: &dyn PlArray, ranges: F) -> Option<PlBitmap>
where
    I: Iterator<Item = Range<usize>>,
    F: Fn() -> I,
{
    let length = values.len();
    let validity = values.validity();

    // A value that is already null is one the nulls above it have already reached, so it is the
    // values that are still valid that are left to unset: a run of ranges without one is nothing to
    // write at all.
    if !ranges().any(|range| set_bits_in(validity, range) > 0) {
        return None;
    }

    // Every value is covered, so the whole mask comes down to the single unset bit they share.
    // Ordered ranges that do not overlap cover every value exactly when their lengths add up to
    // however many there are.
    if ranges().map(|range| range.len()).sum::<usize>() == length {
        return Some(PlBitmap::new_scalar(false, length));
    }

    let mut mask = BitmapBuilder::with_capacity(length);
    for range in ranges() {
        extend_from_validity(&mut mask, validity, range.start);
        mask.extend_constant(range.len(), false);
    }
    extend_from_validity(&mut mask, validity, length);

    Some(PlBitmap::from_bitmap(mask.freeze()))
}

/// Extends `mask` with the bits `validity` holds from where `mask` ends up to `end`.
fn extend_from_validity(mask: &mut BitmapBuilder, validity: Option<PlBitmapRef<'_>>, end: usize) {
    let offset = mask.len();
    let length = end
        .checked_sub(offset)
        .expect("the ranges a null is pushed down onto are ordered and do not overlap");

    match validity.map(|validity| validity.repr()) {
        // Nothing to read: an absent mask says every value is valid, and a single bit says the same
        // of every value in turn.
        None => mask.extend_constant(length, true),
        Some(ArrayRepr::Scalar(value)) => mask.extend_constant(length, value),
        Some(ArrayRepr::Flat(validity)) => {
            mask.subslice_extend_from_bitmap(validity, offset, length)
        },
    }
}

/// The number of values in `range` that `validity` says are not null.
fn set_bits_in(validity: Option<PlBitmapRef<'_>>, range: Range<usize>) -> usize {
    match validity.map(|validity| validity.repr()) {
        None => range.len(),
        // A single bit says the same of every value, so there is nothing to count.
        Some(ArrayRepr::Scalar(value)) => range.len() * usize::from(value),
        Some(ArrayRepr::Flat(validity)) => BitMask::from_bitmap(validity)
            .sliced(range.start, range.len())
            .set_bits(),
    }
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;
    use polars_array::PlPrimitiveArray;
    use polars_array::arrow::import::from_arrow;
    use polars_buffer::Buffer;

    use super::*;

    fn primitives(values: impl IntoIterator<Item = i32>) -> Box<dyn PlArray> {
        Box::new(PlPrimitiveArray::from_vec(values.into_iter().collect()))
    }

    /// An array whose every element is the same null list pushes a single unset bit down onto the
    /// values, however many elements it holds: neither the list nor the mask is written out.
    #[test]
    fn a_fully_null_repeated_list_pushes_one_bit_down() {
        let array = PlListArray::new_scalar(primitives(1..4), 1_000)
            .with_validity_broadcast(Some(Bitmap::new_zeroed(1)));

        let propagated = propagate_nulls_list(&array).unwrap();

        assert!(propagated.offsets_are_scalar());
        assert_eq!(propagated.len(), 1_000);
        assert_eq!(propagated.values().len(), 3);
        assert_eq!(
            propagated
                .values()
                .validity()
                .and_then(|v| v.scalar_value()),
            Some(false),
        );
    }

    /// A struct whose every element is null hands its one unset bit to every field.
    #[test]
    fn a_fully_null_struct_hands_one_bit_to_every_field() {
        let array = PlStructArray::new_full_null(vec![primitives(1..4)], 3);

        let propagated = propagate_nulls_struct(&array).unwrap();

        assert_eq!(
            propagated
                .field(0)
                .validity()
                .and_then(|v| v.scalar_value()),
            Some(false),
        );
    }

    /// A level whose values are already null under every null element is handed back untouched.
    #[test]
    fn values_that_are_already_null_are_left_alone() {
        let values = PlPrimitiveArray::from_iter([None, None, Some(3i32)]);
        let array = PlListArray::from_offsets(Box::new(values), Buffer::from(vec![0, 2, 3]))
            .with_validity(Some([false, true].into_iter().collect()));

        assert!(propagate_nulls_list(&array).is_none());
    }

    /// The elements of a repeated list share one range of the values, so a null element that shares
    /// it with a valid one is only told apart from it once the array is written out.
    #[test]
    fn a_repeated_list_with_some_nulls_is_written_out() {
        let array = PlListArray::new_scalar(primitives(1..3), 3)
            .with_validity(Some([true, false, true].into_iter().collect()));

        let propagated = propagate_nulls_list(&array).unwrap();

        assert!(propagated.offsets_are_flat());
        assert_eq!(propagated.values().len(), 6);
        assert_eq!(
            propagated
                .values()
                .as_any()
                .downcast_ref::<PlPrimitiveArray<i32>>(),
            Some(&PlPrimitiveArray::from_iter([
                Some(1),
                Some(2),
                None,
                None,
                Some(1),
                Some(2),
            ])),
        );

        // The elements themselves are the ones it came in with.
        assert_eq!(propagated, array);
    }

    /// Pushing nulls down only ever changes values that sit under a null, which are no part of an
    /// element: the elements that come back are the ones that went in.
    mod proptests {
        use arrow::array::proptest::array;
        use proptest::proptest;

        use super::*;

        proptest! {
            #[test]
            fn propagating_nulls_leaves_every_element_alone(array in array(0..100)) {
                let array = from_arrow(array.as_ref());

                if let Some(propagated) = propagate_nulls(&*array) {
                    proptest::prop_assert!(array.eq_dyn(&*propagated));
                }
            }
        }
    }
}
