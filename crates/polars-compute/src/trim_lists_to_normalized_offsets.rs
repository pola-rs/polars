//! Trimming a nested chunk down to the values its elements actually cover.
//!
//! A list array is free to hold values no element of it covers: the offsets may start past the
//! beginning of the values array and stop before its end, which is what slicing one leaves behind.
//! This module rewrites such an array into the one that holds the covered values alone, so that the
//! offsets start at zero and end at the end of the values — what the row encoding and the kernels
//! that walk the values directly need of it.
//!
//! Trimming reads no value and writes none: the values are handed over as a slice of the buffers
//! they already sit in, and the offsets are shifted by the one start they all sit past. So a chunk
//! comes back in the representation it went in — an array whose every element is the same list
//! keeps its two offsets, and is trimmed in `O(1)`.

use polars_array::{PlArray, PlArrayType, PlFixedSizeListArray, PlListArray, PlStructArray};
use polars_buffer::Buffer;

use crate::nesting::{covered_range, downcast, fsl_values, fsl_with_values, struct_with_fields};

/// Trims the lists of `array` down to the values its elements cover, recursively.
///
/// Returns `None` if no level of `array` had a value to trim, in which case `array` is already the
/// answer.
pub fn trim_lists_to_normalized_offsets(array: &dyn PlArray) -> Option<Box<dyn PlArray>> {
    match array.array_type() {
        PlArrayType::List => {
            trim_lists_to_normalized_offsets_list(downcast(array)).map(|array| Box::new(array) as _)
        },
        PlArrayType::FixedSizeList => {
            trim_lists_to_normalized_offsets_fsl(downcast(array)).map(|array| Box::new(array) as _)
        },
        PlArrayType::Struct => trim_lists_to_normalized_offsets_struct(downcast(array))
            .map(|array| Box::new(array) as _),
        // A leaf holds no offsets, and nothing below it holds any either.
        _ => None,
    }
}

/// Trims `array` down to the values its elements cover, and its values array in turn.
pub fn trim_lists_to_normalized_offsets_list(array: &PlListArray) -> Option<PlListArray> {
    let covered = covered_range(array);

    // The values array holds exactly the values the elements cover, so the offsets already start at
    // its beginning and end at its end. Only a deeper level can still have something to trim.
    if array.values().len() == covered.len() {
        let values = trim_lists_to_normalized_offsets(array.values())?;

        // SAFETY: the trimmed values are as many as the ones they replace, so the offsets still
        // reach no further than they do.
        return Some(unsafe { crate::nesting::list_with_values(array, values) });
    }

    // Slicing hands the buffers over as they are, under a new offset and length: the values the
    // elements do not cover are dropped without the ones they do being read.
    let values = array.values().sliced(covered.start, covered.len());
    let values = trim_lists_to_normalized_offsets(&*values).unwrap_or(values);

    let offsets_are_flat = array.offsets_are_flat();
    let (_, offsets, length, validity) = array.clone().into_inner();

    // Every offset moves back by the one start they all sit past, which leaves the buffer exactly
    // as long as it was: a scalar array keeps its two offsets, and stays scalar.
    let start = covered.start as u64;
    let offsets = Buffer::from(
        offsets
            .iter()
            .map(|offset| offset - start)
            .collect::<Vec<_>>(),
    );

    // SAFETY: the offsets are as many as they were, and so still flat or scalar for `length` as
    // they were; shifting them all by the same start leaves them non-decreasing, and ending at the
    // end of the values they were trimmed to. The mask is untouched, and so still valid for
    // `length`.
    Some(unsafe {
        if offsets_are_flat {
            PlListArray::new_unchecked(values, offsets, length, validity)
        } else {
            PlListArray::new_broadcast_unchecked(values, offsets, length, validity)
        }
    })
}

/// Trims the values of `array`, which holds no offsets of its own to normalize.
pub fn trim_lists_to_normalized_offsets_fsl(
    array: &PlFixedSizeListArray,
) -> Option<PlFixedSizeListArray> {
    let values = trim_lists_to_normalized_offsets(fsl_values(array))?;

    // SAFETY: the trimmed values are as many as the ones they replace, so they are cut into the
    // same elements of the same width.
    Some(unsafe { fsl_with_values(array, values) })
}

/// Trims every field of `array`, which holds no offsets of its own to normalize.
pub fn trim_lists_to_normalized_offsets_struct(array: &PlStructArray) -> Option<PlStructArray> {
    // The fields are walked until one of them has something to trim; a struct whose every field is
    // already trimmed is handed back untouched, and the fields before that one are borrowed rather
    // than walked a second time.
    let first_trimmed = array
        .fields()
        .iter()
        .position(|field| trim_lists_to_normalized_offsets(&**field).is_some())?;

    let fields = array
        .fields()
        .iter()
        .enumerate()
        .map(|(i, field)| match i.cmp(&first_trimmed) {
            std::cmp::Ordering::Less => field.clone(),
            _ => trim_lists_to_normalized_offsets(&**field).unwrap_or_else(|| field.clone()),
        })
        .collect();

    // SAFETY: every trimmed field is as long as the one it replaces, so each still holds one
    // element per element of the struct.
    Some(unsafe { struct_with_fields(array, fields) })
}

#[cfg(test)]
mod tests {
    use polars_array::PlPrimitiveArray;

    use super::*;

    fn values(range: std::ops::Range<i32>) -> Box<dyn PlArray> {
        Box::new(PlPrimitiveArray::from_vec(range.collect()))
    }

    /// An array whose elements cover the whole values array is already trimmed.
    #[test]
    fn a_normalized_array_is_left_alone() {
        let array = PlListArray::from_offsets(values(0..6), Buffer::from(vec![0, 2, 4, 6]));

        assert!(trim_lists_to_normalized_offsets_list(&array).is_none());
    }

    /// The values the elements do not cover are dropped, and the offsets moved back onto the ones
    /// that are left.
    #[test]
    fn the_values_outside_the_covered_range_are_dropped() {
        let array = PlListArray::from_offsets(values(0..10), Buffer::from(vec![2, 4, 7]));

        let trimmed = trim_lists_to_normalized_offsets_list(&array).unwrap();

        assert_eq!(trimmed.values().len(), 5);
        assert_eq!(
            trimmed
                .values()
                .as_any()
                .downcast_ref::<PlPrimitiveArray<i32>>(),
            Some(&PlPrimitiveArray::from_vec(vec![2, 3, 4, 5, 6])),
        );
        assert_eq!(trimmed.value_range(0), 0..2);
        assert_eq!(trimmed.value_range(1), 2..5);
    }

    /// An array whose every element is the same list is trimmed without that list ever being
    /// written out once per element: it keeps the two offsets it came in with.
    #[test]
    fn a_repeated_list_stays_repeated() {
        let array =
            PlListArray::new_broadcast(values(0..10), Buffer::from(vec![3, 6]), 1_000, None);

        let trimmed = trim_lists_to_normalized_offsets_list(&array).unwrap();

        assert!(trimmed.offsets_are_scalar());
        assert_eq!(trimmed.len(), 1_000);
        assert_eq!(trimmed.values().len(), 3);
        assert_eq!(trimmed.scalar_offsets(), Some(0..3));
        assert_eq!(trimmed, array);
    }

    /// Trimming reaches every level: a struct hands back its fields trimmed, and a fixed size list
    /// its values.
    #[test]
    fn trimming_reaches_through_every_level() {
        let inner = PlListArray::from_offsets(values(0..10), Buffer::from(vec![1, 4]));
        let outer = PlFixedSizeListArray::from_values(Box::new(inner), 1);
        let array = PlStructArray::from_fields(vec![Box::new(outer)]);

        let trimmed = trim_lists_to_normalized_offsets_struct(&array).unwrap();

        let field = trimmed.field(0).as_any();
        let field = field.downcast_ref::<PlFixedSizeListArray>().unwrap();
        let values = fsl_values(field).as_any();
        let values = values.downcast_ref::<PlListArray>().unwrap();

        assert_eq!(values.values().len(), 3);
        assert_eq!(values.value_range(0), 0..3);
    }
}
