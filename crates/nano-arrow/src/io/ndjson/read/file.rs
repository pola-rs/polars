use std::io::BufRead;

use fallible_streaming_iterator::FallibleStreamingIterator;
use indexmap::set::IndexSet as HashSet;
use json_deserializer::parse;

use super::super::super::json::read::{coerce_data_type, infer as infer_json};
use crate::datatypes::DataType;
use crate::error::{Error, Result};

/// Reads up to a number of lines from `reader` into `rows` bounded by `limit`.
fn read_rows<R: BufRead>(reader: &mut R, rows: &mut [String], limit: usize) -> Result<usize> {
    if limit == 0 {
        return Ok(0);
    }
    let mut row_number = 0;
    for row in rows.iter_mut() {
        loop {
            row.clear();
            let _ = reader
                .read_line(row)
                .map_err(|e| Error::External(format!(" at line {row_number}"), Box::new(e)))?;
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
/// # Implementantion
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

    /// Deconstruct [`FileReader`] into the reader and the internal buffer.
    pub fn into_inner(self) -> (R, Vec<String>) {
        (self.reader, self.rows)
    }
}

impl<R: BufRead> FallibleStreamingIterator for FileReader<R> {
    type Error = Error;
    type Item = [String];

    fn advance(&mut self) -> Result<()> {
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

/// Infers the [`DataType`] from an NDJSON file, optionally only using `number_of_rows` rows.
///
/// # Implementation
/// This implementation reads the file line by line and infers the type of each line.
/// It performs both `O(N)` IO and CPU-bounded operations where `N` is the number of rows.
pub fn infer<R: std::io::BufRead>(
    reader: &mut R,
    number_of_rows: Option<usize>,
) -> Result<DataType> {
    if reader.fill_buf().map(|b| b.is_empty())? {
        return Err(Error::ExternalFormat(
            "Cannot infer NDJSON types on empty reader because empty string is not a valid JSON value".to_string(),
        ));
    }

    let rows = vec!["".to_string(); 1]; // 1 <=> read row by row
    let mut reader = FileReader::new(reader, rows, number_of_rows);

    let mut data_types = HashSet::new();
    while let Some(rows) = reader.next()? {
        let value = parse(rows[0].as_bytes())?; // 0 because it is row by row
        let data_type = infer_json(&value)?;
        if data_type != DataType::Null {
            data_types.insert(data_type);
        }
    }

    let v: Vec<&DataType> = data_types.iter().collect();
    Ok(coerce_data_type(&v))
}

/// Infers the [`DataType`] from an iterator of JSON strings. A limited number of
/// rows can be used by passing `rows.take(number_of_rows)` as an input.
///
/// # Implementation
/// This implementation infers each row by going through the entire iterator.
pub fn infer_iter<A: AsRef<str>>(rows: impl Iterator<Item = A>) -> Result<DataType> {
    let mut data_types = HashSet::new();
    for row in rows {
        let v = parse(row.as_ref().as_bytes())?;
        let data_type = infer_json(&v)?;
        if data_type != DataType::Null {
            data_types.insert(data_type);
        }
    }

    let v: Vec<&DataType> = data_types.iter().collect();
    Ok(coerce_data_type(&v))
}
