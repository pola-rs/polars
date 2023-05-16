use super::*;

/// Deserializes an iterator of rows into an [`Array`] of [`DataType`].
/// # Implementation
/// This function is CPU-bounded.
/// This function is guaranteed to return an array of length equal to the length
/// # Errors
/// This function errors iff any of the rows is not a valid JSON (i.e. the format is not valid NDJSON).
pub fn deserialize_iter<'a>(
    rows: impl Iterator<Item = &'a str>,
    data_type: DataType,
) -> PolarsResult<ArrayRef> {
    // deserialize strings to `Value`s
    let rows = rows
        .map(|row| {
            simd_json::serde::to_borrowed_value(row)
                .map_err(|e| PolarsError::ComputeError(format!("json parsing error: '{e}'").into()))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    // deserialize &[Value] to Array
    Ok(super::super::json::deserialize::_deserialize(
        &rows, data_type,
    ))
}
