use arrow::array::{Array, FixedSizeListArray, ListArray, StructArray};
use arrow::types::Offset;

use crate::rebuild_list::rebuild_list_shallow;

/// Trim all lists of unused start and end elements recursively.
pub fn trim_lists_to_normalized_offsets(arr: &dyn Array) -> Option<Box<dyn Array>> {
    let arr = arr.as_any();
    if let Some(arr) = arr.downcast_ref::<ListArray<i32>>() {
        return trim_lists_to_normalized_offsets_list(arr).map(|arr| Box::new(arr) as _);
    }
    if let Some(arr) = arr.downcast_ref::<ListArray<i64>>() {
        return trim_lists_to_normalized_offsets_list(arr).map(|arr| Box::new(arr) as _);
    }
    if let Some(arr) = arr.downcast_ref::<FixedSizeListArray>() {
        return trim_lists_to_normalized_offsets_fsl(arr).map(|arr| Box::new(arr) as _);
    }
    if let Some(arr) = arr.downcast_ref::<StructArray>() {
        return trim_lists_to_normalized_offsets_struct(arr).map(|arr| Box::new(arr) as _);
    }

    None
}

pub fn trim_lists_to_normalized_offsets_list<O: Offset>(
    arr: &ListArray<O>,
) -> Option<ListArray<O>> {
    let offsets = arr.offsets();
    let values = arr.values();

    let len = offsets.range().to_usize();

    let values = if values.len() == len {
        // The offsets already span the whole child, so only a nested list can change.
        trim_lists_to_normalized_offsets(values.as_ref())?
    } else {
        let values = values.sliced(offsets.first().to_usize(), len);
        trim_lists_to_normalized_offsets(values.as_ref()).unwrap_or(values)
    };

    Some(rebuild_list_shallow(arr, arr.dtype().clone(), values))
}

pub fn trim_lists_to_normalized_offsets_fsl(
    arr: &FixedSizeListArray,
) -> Option<FixedSizeListArray> {
    let values = trim_lists_to_normalized_offsets(arr.values().as_ref())?;

    Some(FixedSizeListArray::new(
        arr.dtype().clone(),
        arr.len(),
        values,
        arr.validity().cloned(),
    ))
}

pub fn trim_lists_to_normalized_offsets_struct(arr: &StructArray) -> Option<StructArray> {
    let mut new_values = Vec::new();
    for (i, field_array) in arr.values().iter().enumerate() {
        let Some(field_array) = trim_lists_to_normalized_offsets(field_array.as_ref()) else {
            // Nothing was changed. Return the original array.
            continue;
        };

        new_values.reserve(arr.values().len());
        new_values.extend(arr.values()[..i].iter().cloned());
        new_values.push(field_array);
        break;
    }

    if new_values.is_empty() {
        return None;
    }

    new_values.extend(arr.values()[new_values.len()..].iter().map(|field_array| {
        trim_lists_to_normalized_offsets(field_array.as_ref())
            .unwrap_or_else(|| field_array.clone())
    }));

    Some(StructArray::new(
        arr.dtype().clone(),
        arr.len(),
        new_values,
        arr.validity().cloned(),
    ))
}
