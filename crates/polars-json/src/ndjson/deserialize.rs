use polars_arrow::array::Array;
use polars_arrow::compute::concatenate::concatenate_unchecked;
use simd_json::BorrowedValue;

use super::*;
use crate::json::ordered::TapeGuide;

/// Deserializes an iterator of rows into an [`Array`][Array] of [`DataType`].
///
/// If `guide` is given, rows are parsed with [`tape_to_value`] to keep the source order of
/// the `Map`s it contains.
///
/// [Array]: polars_arrow::array::Array
/// [`tape_to_value`]: crate::json::ordered::tape_to_value
///
/// # Implementation
/// This function is CPU-bounded.
/// This function is guaranteed to return an array of length equal to the length
/// # Errors
/// This function errors iff any of the rows is not a valid JSON (i.e. the format is not valid NDJSON).
pub fn deserialize_iter<'a>(
    rows: impl Iterator<Item = &'a str>,
    dtype: ArrowDataType,
    guide: Option<&ArrowDataType>,
    buf_size: usize,
    count: usize,
    allow_extra_fields_in_struct: bool,
) -> PolarsResult<ArrayRef> {
    // The rows are parsed as one JSON array.
    let guide_dtype = guide.map(|guide| guide.clone().to_large_list(true));
    let guide = guide_dtype.as_ref().map(TapeGuide::new);
    let mut arr: Vec<Box<dyn Array>> = Vec::new();
    let mut buf = Vec::with_capacity(std::cmp::min(buf_size + count + 2, u32::MAX as usize));
    buf.push(b'[');

    fn _deserializer(
        s: &mut [u8],
        dtype: ArrowDataType,
        guide: Option<&TapeGuide>,
        allow_extra_fields_in_struct: bool,
    ) -> PolarsResult<Box<dyn Array>> {
        let parse_err = |e| PolarsError::ComputeError(format!("json parsing error: '{e}'").into());
        let out = match guide {
            Some(guide) => {
                let tape = simd_json::to_tape(s).map_err(parse_err)?;
                crate::json::ordered::tape_to_value(&tape, guide, false)?
            },
            None => simd_json::to_borrowed_value(s).map_err(parse_err)?,
        };
        if let BorrowedValue::Array(rows) = out {
            super::super::json::deserialize::_deserialize(
                &rows,
                dtype,
                allow_extra_fields_in_struct,
            )
        } else {
            unreachable!()
        }
    }
    let mut row_iter = rows.peekable();

    while let Some(row) = row_iter.next() {
        buf.extend_from_slice(row.as_bytes());
        buf.push(b',');

        let next_row_length = row_iter.peek().map(|row| row.len()).unwrap_or(0);
        if buf.len() + next_row_length >= u32::MAX as usize {
            let _ = buf.pop();
            buf.push(b']');
            arr.push(_deserializer(
                &mut buf,
                dtype.clone(),
                guide.as_ref(),
                allow_extra_fields_in_struct,
            )?);
            buf.clear();
            buf.push(b'[');
        }
    }
    if buf.len() > 1 {
        let _ = buf.pop();
    }
    buf.push(b']');

    if arr.is_empty() {
        _deserializer(
            &mut buf,
            dtype,
            guide.as_ref(),
            allow_extra_fields_in_struct,
        )
    } else {
        arr.push(_deserializer(
            &mut buf,
            dtype,
            guide.as_ref(),
            allow_extra_fields_in_struct,
        )?);
        concatenate_unchecked(&arr)
    }
}
