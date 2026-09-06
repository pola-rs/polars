//! Trimming a nested chunk down to the values its elements actually cover.

use polars_array::{
    PlArray, PlArrayType, PlBitmap, PlFixedSizeListArray, PlListArray, PlStructArray,
};
use polars_buffer::Buffer;

use crate::nesting::{covered_range, downcast, fsl_with_values, struct_with_fields};

/// Trims the lists of `array` down to the values its elements cover, recursively.
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
    let validity = array.validity().map(PlBitmap::from);
    let (_, offsets, length, _) = array.clone().into_inner();

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
    Some(unsafe {
        if offsets_are_flat {
            PlListArray::new_unchecked(values, offsets, length, validity.clone())
        } else {
            PlListArray::new_broadcast_unchecked(values, offsets, length, validity)
        }
    })
}

/// Trims the values of `array`, which holds no offsets of its own to normalize.
pub fn trim_lists_to_normalized_offsets_fsl(
    array: &PlFixedSizeListArray,
) -> Option<PlFixedSizeListArray> {
    let values = trim_lists_to_normalized_offsets(array.values())?;

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
