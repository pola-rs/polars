use arrow::array::growable::make_growable;
use arrow::error::{Error as ArrowError, Result};

use crate::prelude::*;

/// Concatenate multiple [Array] of the same type into a single [`Array`].
/// This does not check the arrays types.
pub fn concatenate_owned_unchecked(arrays: &[ArrayRef]) -> Result<ArrayRef> {
    if arrays.is_empty() {
        return Err(ArrowError::InvalidArgumentError(
            "concat requires input of at least one array".to_string(),
        ));
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
        mutable.extend(i, 0, *len)
    }

    Ok(mutable.as_box())
}
