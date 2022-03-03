use crate::csv::CsvEncoding;
use crate::csv_core::parser::{
    next_line_position, skip_bom, skip_line_ending, SplitFields, SplitLines,
};
use crate::mmap::{MmapBytesReader, ReaderBytes};
use crate::prelude::NullValues;
use lazy_static::lazy_static;
use polars_core::datatypes::PlHashSet;
use polars_core::prelude::*;
use regex::{Regex, RegexBuilder};
use std::borrow::Cow;
use std::io::Read;

pub(crate) fn get_file_chunks(
    bytes: &[u8],
    n_threads: usize,
    expected_fields: usize,
    delimiter: u8,
    quote_char: Option<u8>,
) -> Vec<(usize, usize)> {
    let mut last_pos = 0;
    let total_len = bytes.len();
    let chunk_size = total_len / n_threads;
    let mut offsets = Vec::with_capacity(n_threads);
    for _ in 0..n_threads {
        let search_pos = last_pos + chunk_size;

        if search_pos >= bytes.len() {
            break;
        }

        let end_pos = match next_line_position(
            &bytes[search_pos..],
            expected_fields,
            delimiter,
            quote_char,
        ) {
            Some(pos) => search_pos + pos,
            None => {
                break;
            }
        };
        offsets.push((last_pos, end_pos));
        last_pos = end_pos;
    }
    offsets.push((last_pos, total_len));
    offsets
}

pub fn get_reader_bytes<R: Read + MmapBytesReader>(reader: &mut R) -> Result<ReaderBytes<'_>> {
    // we have a file so we can mmap
    if let Some(file) = reader.to_file() {
        let mmap = unsafe { memmap::Mmap::map(file)? };
        Ok(ReaderBytes::Mapped(mmap))
    } else {
        // we can get the bytes for free
        if reader.to_bytes().is_some() {
            // duplicate .to_bytes() is necessary to satisfy the borrow checker
            Ok(ReaderBytes::Borrowed(reader.to_bytes().unwrap()))
        } else {
            // we have to read to an owned buffer to get the bytes.
            let mut bytes = Vec::with_capacity(1024 * 128);
            reader.read_to_end(&mut bytes)?;
            if !bytes.is_empty()
                && (bytes[bytes.len() - 1] != b'\n' || bytes[bytes.len() - 1] != b'\r')
            {
                bytes.push(b'\n')
            }
            Ok(ReaderBytes::Owned(bytes))
        }
    }
}

lazy_static! {
    static ref FLOAT_RE: Regex =
        Regex::new(r"^(\s*-?((\d*\.\d+)[eE]?[-\+]?\d*)|[-+]?inf|[-+]?NaN|\d+[eE][-+]\d+)$")
            .unwrap();
    static ref INTEGER_RE: Regex = Regex::new(r"^\s*-?(\d+)$").unwrap();
    static ref BOOLEAN_RE: Regex = RegexBuilder::new(r"^\s*(true)$|^(false)$")
        .case_insensitive(true)
        .build()
        .unwrap();
}

/// Infer the data type of a record
fn infer_field_schema(string: &str) -> DataType {
    // when quoting is enabled in the reader, these quotes aren't escaped, we default to
    // Utf8 for them
    if string.starts_with('"') {
        return DataType::Utf8;
    }
    // match regex in a particular order
    if BOOLEAN_RE.is_match(string) {
        DataType::Boolean
    } else if FLOAT_RE.is_match(string) {
        DataType::Float64
    } else if INTEGER_RE.is_match(string) {
        DataType::Int64
    } else {
        DataType::Utf8
    }
}

#[inline]
pub(crate) fn parse_bytes_with_encoding(bytes: &[u8], encoding: CsvEncoding) -> Result<Cow<str>> {
    let s = match encoding {
        CsvEncoding::Utf8 => simdutf8::basic::from_utf8(bytes)
            .map_err(anyhow::Error::from)?
            .into(),
        CsvEncoding::LossyUtf8 => String::from_utf8_lossy(bytes),
    };
    Ok(s)
}

/// Infer the schema of a CSV file by reading through the first n records of the file,
/// with `max_read_records` controlling the maximum number of records to read.
///
/// If `max_read_records` is not set, the whole file is read to infer its schema.
///
/// Return inferred schema and number of records used for inference.
#[allow(clippy::too_many_arguments)]
pub fn infer_file_schema(
    reader_bytes: &ReaderBytes,
    delimiter: u8,
    max_read_lines: Option<usize>,
    has_header: bool,
    schema_overwrite: Option<&Schema>,
    // we take &mut because we maybe need to skip more rows dependent
    // on the schema inference
    skip_rows: &mut usize,
    comment_char: Option<u8>,
    quote_char: Option<u8>,
    null_values: Option<&NullValues>,
) -> Result<(Schema, usize)> {
    // We use lossy utf8 here because we don't want the schema inference to fail on utf8.
    // It may later.
    let encoding = CsvEncoding::LossyUtf8;

    let bytes = skip_line_ending(skip_bom(reader_bytes));
    if bytes.is_empty() {
        return Err(PolarsError::NoData("empty csv".into()));
    }
    let mut lines = SplitLines::new(bytes, b'\n').skip(*skip_rows);
    // it can be that we have a single line without eol char
    let has_eol = bytes.contains(&b'\n');

    // get or create header names
    // when has_header is false, creates default column names with column_ prefix

    // skip lines that are comments
    let mut first_line = None;
    if let Some(comment_ch) = comment_char {
        for (i, line) in (&mut lines).enumerate() {
            if let Some(ch) = line.get(0) {
                if *ch != comment_ch {
                    first_line = Some(line);
                    *skip_rows += i;
                    break;
                }
            }
        }
    } else {
        first_line = lines.next();
    }
    // edge case where we a single row, no header and no eol char.
    if first_line.is_none() && !has_eol && !has_header {
        first_line = Some(bytes);
    }

    // now that we've found the first non-comment line we parse the headers, or we create a header
    let headers: Vec<String> = if let Some(mut header_line) = first_line {
        let len = header_line.len();
        if len > 1 {
            // remove carriage return
            let trailing_byte = header_line[len - 1];
            if trailing_byte == b'\r' {
                header_line = &header_line[..len - 1];
            }
        }

        let byterecord = SplitFields::new(header_line, delimiter, quote_char);
        if has_header {
            byterecord
                .map(|(slice, needs_escaping)| {
                    let slice_escaped = if needs_escaping && (slice.len() >= 2) {
                        &slice[1..(slice.len() - 1)]
                    } else {
                        slice
                    };
                    let s = parse_bytes_with_encoding(slice_escaped, encoding)?;
                    Ok(s.into())
                })
                .collect::<Result<_>>()?
        } else {
            let mut column_names: Vec<String> = byterecord
                .enumerate()
                .map(|(i, _s)| format!("column_{}", i + 1))
                .collect();
            // needed because SplitLines does not return the \n char, so SplitFields does not catch
            // the latest value if ending with ','
            if header_line.ends_with(b",") {
                column_names.push(format!("column_{}", column_names.len() + 1))
            }
            column_names
        }
    } else {
        return Err(PolarsError::NoData("empty csv".into()));
    };
    if !has_header {
        // re-init lines so that the header is included in type inference.
        lines = SplitLines::new(bytes, b'\n').skip(*skip_rows);
    }

    let header_length = headers.len();
    // keep track of inferred field types
    let mut column_types: Vec<PlHashSet<DataType>> = vec![PlHashSet::new(); header_length];
    // keep track of columns with nulls
    let mut nulls: Vec<bool> = vec![false; header_length];

    let mut rows_count = 0;
    let mut fields = Vec::with_capacity(header_length);

    // needed to prevent ownership going into the iterator loop
    let records_ref = &mut lines;

    for mut line in records_ref.take(max_read_lines.unwrap_or(usize::MAX)) {
        rows_count += 1;

        if let Some(c) = comment_char {
            // line is a comment -> skip
            if line[0] == c {
                continue;
            }
        }

        let len = line.len();
        if len > 1 {
            // remove carriage return
            let trailing_byte = line[len - 1];
            if trailing_byte == b'\r' {
                line = &line[..len - 1];
            }
        }

        let mut record = SplitFields::new(line, delimiter, quote_char);

        for i in 0..header_length {
            if let Some((slice, needs_escaping)) = record.next() {
                if slice.is_empty() {
                    nulls[i] = true;
                } else {
                    let slice_escaped = if needs_escaping && (slice.len() >= 2) {
                        &slice[1..(slice.len() - 1)]
                    } else {
                        slice
                    };
                    let s = parse_bytes_with_encoding(slice_escaped, encoding)?;
                    match &null_values {
                        None => {
                            column_types[i].insert(infer_field_schema(&s));
                        }
                        Some(NullValues::Columns(names)) => {
                            if !names.iter().any(|name| name == s.as_ref()) {
                                column_types[i].insert(infer_field_schema(&s));
                            }
                        }
                        Some(NullValues::AllColumns(name)) => {
                            if s.as_ref() != name {
                                column_types[i].insert(infer_field_schema(&s));
                            }
                        }
                        Some(NullValues::Named(names)) => {
                            let current_name = &headers[i];
                            let null_name = &names.iter().find(|name| &name.0 == current_name);

                            if let Some(null_name) = null_name {
                                if null_name.1 != s.as_ref() {
                                    column_types[i].insert(infer_field_schema(&s));
                                }
                            } else {
                                column_types[i].insert(infer_field_schema(&s));
                            }
                        }
                    }
                }
            }
        }
    }

    // build schema from inference results
    for i in 0..header_length {
        let possibilities = &column_types[i];
        let field_name = &headers[i];

        if let Some(schema_overwrite) = schema_overwrite {
            if let Some((_, name, dtype)) = schema_overwrite.get_full(field_name) {
                fields.push(Field::new(name, dtype.clone()));
                continue;
            }
        }

        // determine data type based on possible types
        // if there are incompatible types, use DataType::Utf8
        match possibilities.len() {
            1 => {
                for dtype in possibilities.iter() {
                    fields.push(Field::new(field_name, dtype.clone()));
                }
            }
            2 => {
                if possibilities.contains(&DataType::Int64)
                    && possibilities.contains(&DataType::Float64)
                {
                    // we have an integer and double, fall down to double
                    fields.push(Field::new(field_name, DataType::Float64));
                } else {
                    // default to Utf8 for conflicting datatypes (e.g bool and int)
                    fields.push(Field::new(field_name, DataType::Utf8));
                }
            }
            _ => fields.push(Field::new(field_name, DataType::Utf8)),
        }
    }
    // if there is a single line after the header without an eol
    // we copy the bytes add an eol and rerun this function
    // so that the inference is consistent with and without eol char
    if rows_count == 0 && reader_bytes[reader_bytes.len() - 1] != b'\n' {
        let mut rb = Vec::with_capacity(reader_bytes.len() + 1);
        rb.extend_from_slice(reader_bytes);
        rb.push(b'\n');
        return infer_file_schema(
            &ReaderBytes::Owned(rb),
            delimiter,
            max_read_lines,
            has_header,
            schema_overwrite,
            skip_rows,
            comment_char,
            quote_char,
            null_values,
        );
    }

    Ok((Schema::from(fields), rows_count))
}

// magic numbers
const GZIP: [u8; 2] = [31, 139];
const ZLIB0: [u8; 2] = [0x78, 0x01];
const ZLIB1: [u8; 2] = [0x78, 0x9C];
const ZLIB2: [u8; 2] = [0x78, 0xDA];

/// check if csv file is compressed
pub fn is_compressed(bytes: &[u8]) -> bool {
    bytes.starts_with(&ZLIB0)
        || bytes.starts_with(&ZLIB1)
        || bytes.starts_with(&ZLIB2)
        || bytes.starts_with(&GZIP)
}

#[cfg(any(feature = "decompress", feature = "decompress-fast"))]
pub(crate) fn decompress(bytes: &[u8]) -> Option<Vec<u8>> {
    if bytes.starts_with(&GZIP) {
        let mut out = Vec::with_capacity(bytes.len());
        let mut decoder = flate2::read::MultiGzDecoder::new(bytes);
        decoder.read_to_end(&mut out).ok()?;
        Some(out)
    } else if bytes.starts_with(&ZLIB0) || bytes.starts_with(&ZLIB1) || bytes.starts_with(&ZLIB2) {
        let mut out = Vec::with_capacity(bytes.len());
        let mut decoder = flate2::read::ZlibDecoder::new(bytes);
        decoder.read_to_end(&mut out).ok()?;
        Some(out)
    } else {
        None
    }
}

/// replace double quotes by single ones
///
/// This function assumes that bytes is wrapped in the quoting character.
///
/// # Safety
///
/// The caller must ensure that:
///     - Output buffer must have enough capacity to hold `bytes.len()`
///     - bytes ends with the quote character e.g.: `"`
pub(super) unsafe fn escape_field(bytes: &[u8], quote: u8, buf: &mut [u8]) -> usize {
    let mut prev_quote = false;

    let mut count = 0;
    for c in bytes.get_unchecked(1..bytes.len() - 1) {
        if *c == quote {
            if prev_quote {
                prev_quote = false;
                *buf.get_unchecked_mut(count) = *c;
                count += 1;
            } else {
                prev_quote = true;
            }
        } else {
            prev_quote = false;
            *buf.get_unchecked_mut(count) = *c;
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_float_parse() {
        assert!(FLOAT_RE.is_match("0.1"));
        assert!(FLOAT_RE.is_match("3.0"));
        assert!(FLOAT_RE.is_match("3.00001"));
        assert!(FLOAT_RE.is_match("-9.9990e-003"));
        assert!(FLOAT_RE.is_match("9.9990e+003"));
        assert!(FLOAT_RE.is_match("9.9990E+003"));
        assert!(FLOAT_RE.is_match("9.9990E+003"));
        assert!(FLOAT_RE.is_match(".5"));
        assert!(FLOAT_RE.is_match("2.5E-10"));
        assert!(FLOAT_RE.is_match("2.5e10"));
        assert!(FLOAT_RE.is_match("NaN"));
        assert!(FLOAT_RE.is_match("-NaN"));
        assert!(FLOAT_RE.is_match("-inf"));
        assert!(FLOAT_RE.is_match("inf"));
    }

    #[test]
    fn test_get_file_chunks() {
        let path = "../../examples/datasets/foods1.csv";
        let s = std::fs::read_to_string(path).unwrap();
        let bytes = s.as_bytes();
        // can be within -1 / +1 bounds.
        assert!((get_file_chunks(bytes, 10, 4, b',', None).len() as i32 - 10).abs() <= 1);
        assert!((get_file_chunks(bytes, 8, 4, b',', None).len() as i32 - 8).abs() <= 1);
    }
}
