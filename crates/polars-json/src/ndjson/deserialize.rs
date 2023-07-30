use simd_json::BorrowedValue;

use super::*;

/// Deserializes an iterator of rows into an [`Array`][Array] of [`DataType`].
///
/// [Array]: arrow::array::Array
///
/// # Implementation
/// This function is CPU-bounded.
/// This function is guaranteed to return an array of length equal to the length
/// # Errors
/// This function errors iff any of the rows is not a valid JSON (i.e. the format is not valid NDJSON).
pub fn deserialize_iter<'a>(
    rows: impl Iterator<Item = &'a str>,
    data_type: DataType,
    buf_size: usize,
    count: usize,
) -> PolarsResult<ArrayRef> {
    let mut buf = String::with_capacity(buf_size + count + 2);
    buf.push('[');
    for row in rows {
        buf.push_str(row);
        buf.push(',')
    }
    if buf.len() > 1 {
        let _ = buf.pop();
    }
    buf.push(']');
    let slice = unsafe { buf.as_bytes_mut() };
    let out = simd_json::to_borrowed_value(slice)
        .map_err(|e| PolarsError::ComputeError(format!("json parsing error: '{e}'").into()))?;
    if let BorrowedValue::Array(rows) = out {
        Ok(super::super::json::deserialize::_deserialize(
            &rows, data_type,
        ))
    } else {
        unreachable!()
    }
}
