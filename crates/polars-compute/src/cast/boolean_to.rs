//! Casting the boolean arrays of `polars-array`.

use polars_array::{PlBinaryViewArray, PlBitmap, PlBooleanArray, PlUtf8ViewArray};

/// Writes every element as the text of what it holds, which is how a boolean reads as a string.
pub fn boolean_to_binaryview(from: &PlBooleanArray) -> PlBinaryViewArray {
    let text = |set: bool| if set { "true" } else { "false" }.as_bytes();

    let values = match from.scalar_value_ignore_validity() {
        Some(value) => PlBinaryViewArray::new_scalar(text(value), from.len()),
        None => PlBinaryViewArray::from_values_iter(from.flat_values().unwrap().iter().map(text)),
    };
    values.with_validity(from.validity().map(PlBitmap::from))
}

/// [`boolean_to_binaryview`], whose text is UTF-8 because `"true"` and `"false"` are.
pub fn boolean_to_utf8view(from: &PlBooleanArray) -> PlUtf8ViewArray {
    // SAFETY: the two words a boolean is written as are both ASCII, which is valid UTF-8.
    unsafe { PlUtf8ViewArray::from_binview_unchecked(boolean_to_binaryview(from)) }
}
