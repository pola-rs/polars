use arrow::array::Array;
use arrow::compute::concatenate::concatenate;
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
) -> PolarsResult<ArrayRef> {
    let mut arr: Vec<Box<dyn Array>> = Vec::new();
    let mut buf = String::with_capacity(std::u32::MAX as usize);
    buf.push('[');

    fn _deserializer(s: String, data_type: ArrowDataType) -> PolarsResult<Box<dyn Array>> {
        let mut buf = s.clone();
        let slice = unsafe { buf.as_bytes_mut() };
        let out = simd_json::to_borrowed_value(slice)
            .map_err(|e| PolarsError::ComputeError(format!("json parsing error: '{e}'").into()))?;
        Ok(if let BorrowedValue::Array(rows) = out {
            super::super::json::deserialize::_deserialize(&rows, data_type.clone())
        } else {
            unreachable!()
        })
    }

    for row in rows {
        buf.push_str(row);
        buf.push(',');

        if buf.len() + row.len() > 5000000 {
            if buf.len() > 1 {
                let _ = buf.pop();
            }
            buf.push(']');
            arr.push(_deserializer(buf.clone(), data_type.clone())?);
            buf.clear();
            buf.push('[');
        }
    }
    if buf.len() > 1 {
        let _ = buf.pop();
    }
    buf.push(']');

    arr.push(_deserializer(buf, data_type.clone())?);
    concatenate(&arr.clone().iter().map(|v| v.as_ref()).collect::<Vec<_>>())
}
