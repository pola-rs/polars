use std::io::BufRead;
use std::num::NonZeroUsize;

use arrow::datatypes::ArrowDataType;
use fallible_streaming_iterator::FallibleStreamingIterator;
use indexmap::IndexSet;
use polars_error::*;
use polars_utils::aliases::{PlIndexSet, PlRandomState};
use simd_json::BorrowedValue;

/// Reads up to a number of lines from `reader` into `rows` bounded by `limit`.
fn read_rows<R: BufRead>(reader: &mut R, rows: &mut [String], limit: usize) -> PolarsResult<usize> {
    if limit == 0 {
        return Ok(0);
    }
    let mut row_number = 0;
    for row in rows.iter_mut() {
        loop {
            row.clear();
            let _ = reader.read_line(row).map_err(|e| {
                PolarsError::ComputeError(format!("{e} at line {row_number}").into())
            })?;
            if row.is_empty() {
                break;
            }
            if !row.trim().is_empty() {
                break;
            }
        }
        if row.is_empty() {
            break;
        }
        row_number += 1;
        if row_number == limit {
            break;
        }
    }
    Ok(row_number)
}

/// A [`FallibleStreamingIterator`] of NDJSON rows.
///
/// This iterator is used to read chunks of an NDJSON in batches.
/// This iterator is guaranteed to yield at least one row.
/// # Implementation
/// Advancing this iterator is IO-bounded, but does require parsing each byte to find end of lines.
/// # Error
/// Advancing this iterator errors iff the reader errors.
pub struct FileReader<R: BufRead> {
    reader: R,
    rows: Vec<String>,
    number_of_rows: usize,
    remaining: usize,
}

impl<R: BufRead> FileReader<R> {
    /// Creates a new [`FileReader`] from a reader and `rows`.
    ///
    /// The number of items in `rows` denotes the batch size.
    pub fn new(reader: R, rows: Vec<String>, limit: Option<usize>) -> Self {
        Self {
            reader,
            rows,
            remaining: limit.unwrap_or(usize::MAX),
            number_of_rows: 0,
        }
    }
}

impl<R: BufRead> FallibleStreamingIterator for FileReader<R> {
    type Error = PolarsError;
    type Item = [String];

    fn advance(&mut self) -> PolarsResult<()> {
        self.number_of_rows = read_rows(&mut self.reader, &mut self.rows, self.remaining)?;
        self.remaining -= self.number_of_rows;
        Ok(())
    }

    fn get(&self) -> Option<&Self::Item> {
        if self.number_of_rows > 0 {
            Some(&self.rows[..self.number_of_rows])
        } else {
            None
        }
    }
}

fn parse_value<'a>(scratch: &'a mut Vec<u8>, val: &[u8]) -> PolarsResult<BorrowedValue<'a>> {
    scratch.clear();
    scratch.extend_from_slice(val);
    // 0 because it is row by row

    simd_json::to_borrowed_value(scratch)
        .map_err(|e| PolarsError::ComputeError(format!("{e}").into()))
}

/// Infers the [`ArrowDataType`] from an NDJSON file, optionally only using `number_of_rows` rows.
///
/// # Implementation
/// This implementation reads the file line by line and infers the type of each line.
/// It performs both `O(N)` IO and CPU-bounded operations where `N` is the number of rows.
pub fn iter_unique_dtypes<R: std::io::BufRead>(
    reader: &mut R,
    number_of_rows: Option<NonZeroUsize>,
) -> PolarsResult<impl Iterator<Item = ArrowDataType>> {
    if reader.fill_buf().map(|b| b.is_empty())? {
        return Err(PolarsError::ComputeError(
            "Cannot infer NDJSON types on empty reader because empty string is not a valid JSON value".into(),
        ));
    }

    let rows = vec!["".to_string(); 1]; // 1 <=> read row by row
    let mut reader = FileReader::new(reader, rows, number_of_rows.map(|v| v.into()));

    let mut data_types = PlIndexSet::default();
    let mut buf = vec![];
    while let Some(rows) = reader.next()? {
        // 0 because it is row by row
        let value = parse_value(&mut buf, rows[0].as_bytes())?;
        let data_type = crate::json::infer(&value)?;
        data_types.insert(data_type);
    }
    Ok(data_types.into_iter())
}

/// Infers the [`ArrowDataType`] from an iterator of JSON strings. A limited number of
/// rows can be used by passing `rows.take(number_of_rows)` as an input.
///
/// # Implementation
/// This implementation infers each row by going through the entire iterator.
pub fn infer_iter<A: AsRef<str>>(rows: impl Iterator<Item = A>) -> PolarsResult<ArrowDataType> {
    let mut data_types = IndexSet::<_, PlRandomState>::default();

    let mut buf = vec![];
    for row in rows {
        let v = parse_value(&mut buf, row.as_ref().as_bytes())?;
        let data_type = crate::json::infer(&v)?;
        if data_type != ArrowDataType::Null {
            data_types.insert(data_type);
        }
    }

    let v: Vec<&ArrowDataType> = data_types.iter().collect();
    Ok(crate::json::infer_schema::coerce_data_type(&v))
}
