use arrow::array::Array;
use arrow::legacy::kernels::concatenate::concatenate_owned_unchecked;
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
    data_type: ArrowDataType,
    buf_size: usize,
    count: usize,
) -> PolarsResult<ArrayRef> {
    let mut arr: Vec<Box<dyn Array>> = Vec::new();
    let mut buf = String::with_capacity(std::cmp::min(buf_size + count + 2, u32::MAX as usize));
    buf.push('[');

    fn _deserializer(s: &mut str, data_type: ArrowDataType) -> PolarsResult<Box<dyn Array>> {
        let slice = unsafe { s.as_bytes_mut() };
        let out = simd_json::to_borrowed_value(slice)
            .map_err(|e| PolarsError::ComputeError(format!("json parsing error: '{e}'").into()))?;
        Ok(if let BorrowedValue::Array(rows) = out {
            super::super::json::deserialize::_deserialize(&rows, data_type.clone())
        } else {
            unreachable!()
        })
    }
    let mut row_iter = rows.peekable();

    while let Some(row) = row_iter.next() {
        buf.push_str(row);
        buf.push(',');

        let next_row_length = row_iter.peek().map(|row| row.len()).unwrap_or(0);
        if buf.len() + next_row_length >= u32::MAX as usize {
            let _ = buf.pop();
            buf.push(']');
            arr.push(_deserializer(&mut buf, data_type.clone())?);
            buf.clear();
            buf.push('[');
        }
    }
    if buf.len() > 1 {
        let _ = buf.pop();
    }
    buf.push(']');

    if arr.is_empty() {
        _deserializer(&mut buf, data_type.clone())
    } else {
        arr.push(_deserializer(&mut buf, data_type.clone())?);
        concatenate_owned_unchecked(&arr)
    }
}
