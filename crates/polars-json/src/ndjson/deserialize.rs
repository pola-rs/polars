use simd_json::{Buffers, Tape};

use super::*;
use crate::json::ordered::{TapeGuide, tape_to_default_value, tape_to_value};

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
    let guide = guide.map(TapeGuide::new);
    let mut buf = Vec::with_capacity(buf_size);
    let mut row_ends = Vec::with_capacity(count);
    for row in rows {
        buf.extend_from_slice(row.as_bytes());
        row_ends.push(buf.len());
    }

    // Each row is parsed on its own, so that the end of a row is also the end of its value.
    let parse_err = |e| PolarsError::ComputeError(format!("json parsing error: '{e}'").into());
    let mut buffers = Buffers::default();
    let mut tape = Tape(Vec::new());
    let mut values = Vec::with_capacity(row_ends.len());
    let mut rest = buf.as_mut_slice();
    let mut row_start = 0;
    for row_end in row_ends {
        let (row, tail) = std::mem::take(&mut rest).split_at_mut(row_end - row_start);
        rest = tail;
        row_start = row_end;
        simd_json::fill_tape(row, &mut buffers, &mut tape).map_err(parse_err)?;
        let value = match &guide {
            Some(guide) => tape_to_value(&tape, guide, false)?,
            None => tape_to_default_value(&tape),
        };
        values.push(value);
    }

    super::super::json::deserialize::_deserialize(&values, dtype, allow_extra_fields_in_struct)
}
