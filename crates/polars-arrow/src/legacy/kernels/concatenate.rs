use polars_error::{polars_bail, PolarsResult};

use crate::array::growable::make_growable;
use crate::array::ArrayRef;

/// Concatenate multiple [`Array`][Array] of the same type into a single [`Array`][Array].
/// This does not check the arrays types.
///
/// [Array]: arrow::array::Array
pub fn concatenate_owned_unchecked(arrays: &[ArrayRef]) -> PolarsResult<ArrayRef> {
    if arrays.is_empty() {
        polars_bail!(InvalidOperation: "concat requires input of at least one array")
    }
    if arrays.len() == 1 {
        return Ok(arrays[0].clone());
    }
    let mut arrays_ref = Vec::with_capacity(arrays.len());
    let mut lengths = Vec::with_capacity(arrays.len());
    let mut capacity = 0;
    for array in arrays {
        arrays_ref.push(&**array);
        lengths.push(array.len());
        capacity += array.len();
    }

    let mut mutable = make_growable(&arrays_ref, false, capacity);

    for (i, len) in lengths.iter().enumerate() {
        // SAFETY:
        // len is within bounds
        unsafe { mutable.extend(i, 0, *len) }
    }

    Ok(mutable.as_box())
}
