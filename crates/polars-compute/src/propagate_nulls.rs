//! Pushing the nulls of a nested chunk down onto the values under them.

use std::ops::Range;

use arrow::bitmap::BitmapBuilder;
use arrow::bitmap::bitmask::BitMask;
use polars_array::bitmap::combine_validities_and;
use polars_array::{
    PlArray, PlArrayType, PlBitmap, PlBitmapRef, PlFixedSizeListArray, PlListArray, PlStructArray,
};

use crate::nesting::{
    covered_range, downcast, fsl_with_values, list_with_values, struct_with_fields,
    with_pl_validity,
};

/// Pushes the nulls of `array` down onto the values under them, recursively.
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

/// Pushes `validity`, the nulls of the struct above `field`, down onto it and its own in turn.
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

/// The mask `values` takes on once a null is pushed down onto every value in `ranges`.
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

    // Nothing to read: an absent mask says every value is valid, and a single bit says the same of
    // every value in turn.
    let Some(validity) = validity else {
        return mask.extend_constant(length, true);
    };

    match validity.scalar_value() {
        Some(value) => mask.extend_constant(length, value),
        None => mask.subslice_extend_from_bitmap(validity.flat_bitmap().unwrap(), offset, length),
    }
}

/// The number of values in `range` that `validity` says are not null.
fn set_bits_in(validity: Option<PlBitmapRef<'_>>, range: Range<usize>) -> usize {
    let Some(validity) = validity else {
        return range.len();
    };

    match validity.scalar_value() {
        // A single bit says the same of every value, so there is nothing to count.
        Some(value) => range.len() * usize::from(value),
        None => BitMask::from_bitmap(validity.flat_bitmap().unwrap())
            .sliced(range.start, range.len())
            .set_bits(),
    }
}
