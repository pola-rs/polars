use std::borrow::Cow;
#[cfg(any(feature = "decompress", feature = "decompress-fast"))]
use std::io::Read;
use std::mem::MaybeUninit;

use polars_core::datatypes::PlHashSet;
use polars_core::prelude::*;
#[cfg(feature = "polars-time")]
use polars_time::chunkedarray::utf8::infer as date_infer;
#[cfg(feature = "polars-time")]
use polars_time::prelude::utf8::Pattern;

#[cfg(any(feature = "decompress", feature = "decompress-fast"))]
use crate::csv::parser::next_line_position_naive;
use crate::csv::parser::{next_line_position, skip_bom, skip_line_ending, SplitLines};
use crate::csv::splitfields::SplitFields;
use crate::csv::CsvEncoding;
use crate::mmap::ReaderBytes;
use crate::prelude::parser::is_comment_line;
use crate::prelude::{CommentPrefix, NullValues};
use crate::utils::{BOOLEAN_RE, FLOAT_RE, INTEGER_RE};

pub(crate) fn get_file_chunks(
    bytes: &[u8],
    n_chunks: usize,
    expected_fields: usize,
    separator: u8,
    quote_char: Option<u8>,
    eol_char: u8,
) -> Vec<(usize, usize)> {
    let mut last_pos = 0;
    let total_len = bytes.len();
    let chunk_size = total_len / n_chunks;
    let mut offsets = Vec::with_capacity(n_chunks);
    for _ in 0..n_chunks {
        let search_pos = last_pos + chunk_size;

        if search_pos >= bytes.len() {
            break;
        }

        let end_pos = match next_line_position(
            &bytes[search_pos..],
            Some(expected_fields),
            separator,
            quote_char,
            eol_char,
        ) {
            Some(pos) => search_pos + pos,
            None => {
                break;
            },
        };
        offsets.push((last_pos, end_pos));
        last_pos = end_pos;
    }
    offsets.push((last_pos, total_len));
    offsets
}

/// Infer the data type of a record
fn infer_field_schema(string: &str, try_parse_dates: bool) -> DataType {
    // when quoting is enabled in the reader, these quotes aren't escaped, we default to
    // Utf8 for them
    if string.starts_with('"') {
        if try_parse_dates {
            #[cfg(feature = "polars-time")]
            {
                match date_infer::infer_pattern_single(&string[1..string.len() - 1]) {
                    Some(pattern_with_offset) => match pattern_with_offset {
                        Pattern::DatetimeYMD | Pattern::DatetimeDMY => {
                            DataType::Datetime(TimeUnit::Microseconds, None)
                        },
                        Pattern::DateYMD | Pattern::DateDMY => DataType::Date,
                        Pattern::DatetimeYMDZ => {
                            DataType::Datetime(TimeUnit::Microseconds, Some("UTC".to_string()))
                        },
                    },
                    None => DataType::Utf8,
                }
            }
            #[cfg(not(feature = "polars-time"))]
            {
                panic!("activate one of {{'dtype-date', 'dtype-datetime', dtype-time'}} features")
            }
        } else {
            DataType::Utf8
        }
    }
    // match regex in a particular order
    else if BOOLEAN_RE.is_match(string) {
        DataType::Boolean
    } else if FLOAT_RE.is_match(string) {
        DataType::Float64
    } else if INTEGER_RE.is_match(string) {
        DataType::Int64
    } else if try_parse_dates {
        #[cfg(feature = "polars-time")]
        {
            match date_infer::infer_pattern_single(string) {
                Some(pattern_with_offset) => match pattern_with_offset {
                    Pattern::DatetimeYMD | Pattern::DatetimeDMY => {
                        DataType::Datetime(TimeUnit::Microseconds, None)
                    },
                    Pattern::DateYMD | Pattern::DateDMY => DataType::Date,
                    Pattern::DatetimeYMDZ => {
                        DataType::Datetime(TimeUnit::Microseconds, Some("UTC".to_string()))
                    },
                },
                None => DataType::Utf8,
            }
        }
        #[cfg(not(feature = "polars-time"))]
        {
            panic!("activate one of {{'dtype-date', 'dtype-datetime', dtype-time'}} features")
        }
    } else {
        DataType::Utf8
    }
}

#[inline]
pub(crate) fn parse_bytes_with_encoding(
    bytes: &[u8],
    encoding: CsvEncoding,
) -> PolarsResult<Cow<str>> {
    Ok(match encoding {
        CsvEncoding::Utf8 => simdutf8::basic::from_utf8(bytes)
            .map_err(|_| polars_err!(ComputeError: "invalid utf-8 sequence"))?
            .into(),
        CsvEncoding::LossyUtf8 => String::from_utf8_lossy(bytes),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn infer_file_schema_inner(
    reader_bytes: &ReaderBytes,
    separator: u8,
    max_read_rows: Option<usize>,
    has_header: bool,
    schema_overwrite: Option<&Schema>,
    // we take &mut because we maybe need to skip more rows dependent
    // on the schema inference
    skip_rows: &mut usize,
    skip_rows_after_header: usize,
    comment_prefix: Option<&CommentPrefix>,
    quote_char: Option<u8>,
    eol_char: u8,
    null_values: Option<&NullValues>,
    try_parse_dates: bool,
    recursion_count: u8,
    raise_if_empty: bool,
) -> PolarsResult<(Schema, usize, usize)> {
    // keep track so that we can determine the amount of bytes read
    let start_ptr = reader_bytes.as_ptr() as usize;

    // We use lossy utf8 here because we don't want the schema inference to fail on utf8.
    // It may later.
    let encoding = CsvEncoding::LossyUtf8;

    let bytes = skip_line_ending(skip_bom(reader_bytes), eol_char);
    if raise_if_empty {
        polars_ensure!(!bytes.is_empty(), NoData: "empty CSV");
    };
    let mut lines = SplitLines::new(bytes, quote_char.unwrap_or(b'"'), eol_char).skip(*skip_rows);

    // get or create header names
    // when has_header is false, creates default column names with column_ prefix

    // skip lines that are comments
    let mut first_line = None;

    for (i, line) in (&mut lines).enumerate() {
        if !is_comment_line(line, comment_prefix) {
            first_line = Some(line);
            *skip_rows += i;
            break;
        }
    }

    if first_line.is_none() {
        first_line = lines.next();
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

        let byterecord = SplitFields::new(header_line, separator, quote_char, eol_char);
        if has_header {
            let headers = byterecord
                .map(|(slice, needs_escaping)| {
                    let slice_escaped = if needs_escaping && (slice.len() >= 2) {
                        &slice[1..(slice.len() - 1)]
                    } else {
                        slice
                    };
                    let s = parse_bytes_with_encoding(slice_escaped, encoding)?;
                    Ok(s)
                })
                .collect::<PolarsResult<Vec<_>>>()?;

            let mut final_headers = Vec::with_capacity(headers.len());

            let mut header_names = PlHashMap::with_capacity(headers.len());

            for name in &headers {
                let count = header_names.entry(name.as_ref()).or_insert(0usize);
                if *count != 0 {
                    final_headers.push(format!("{}_duplicated_{}", name, *count - 1))
                } else {
                    final_headers.push(name.to_string())
                }
                *count += 1;
            }
            final_headers
        } else {
            let mut column_names: Vec<String> = byterecord
                .enumerate()
                .map(|(i, _s)| format!("column_{}", i + 1))
                .collect();
            // needed because SplitLines does not return the \n char, so SplitFields does not catch
            // the latest value if ending with a separator.
            if header_line.ends_with(&[separator]) {
                column_names.push(format!("column_{}", column_names.len() + 1))
            }
            column_names
        }
    } else if has_header && !bytes.is_empty() && recursion_count == 0 {
        // there was no new line char. So we copy the whole buf and add one
        // this is likely to be cheap as there are no rows.
        let mut buf = Vec::with_capacity(bytes.len() + 2);
        buf.extend_from_slice(bytes);
        buf.push(eol_char);

        return infer_file_schema_inner(
            &ReaderBytes::Owned(buf),
            separator,
            max_read_rows,
            has_header,
            schema_overwrite,
            skip_rows,
            skip_rows_after_header,
            comment_prefix,
            quote_char,
            eol_char,
            null_values,
            try_parse_dates,
            recursion_count + 1,
            raise_if_empty,
        );
    } else if !raise_if_empty {
        return Ok((Schema::new(), 0, 0));
    } else {
        polars_bail!(NoData: "empty CSV");
    };
    if !has_header {
        // re-init lines so that the header is included in type inference.
        lines = SplitLines::new(bytes, quote_char.unwrap_or(b'"'), eol_char).skip(*skip_rows);
    }

    let header_length = headers.len();
    // keep track of inferred field types
    let mut column_types: Vec<PlHashSet<DataType>> =
        vec![PlHashSet::with_capacity(4); header_length];
    // keep track of columns with nulls
    let mut nulls: Vec<bool> = vec![false; header_length];

    let mut rows_count = 0;
    let mut fields = Vec::with_capacity(header_length);

    // needed to prevent ownership going into the iterator loop
    let records_ref = &mut lines;

    let mut end_ptr = start_ptr;
    for mut line in records_ref
        .take(match max_read_rows {
            Some(max_read_rows) => {
                if max_read_rows <= (usize::MAX - skip_rows_after_header) {
                    // read skip_rows_after_header more rows for inferring
                    // the correct schema as the first skip_rows_after_header
                    // rows will be skipped
                    max_read_rows + skip_rows_after_header
                } else {
                    max_read_rows
                }
            },
            None => usize::MAX,
        })
        .skip(skip_rows_after_header)
    {
        rows_count += 1;
        // keep track so that we can determine the amount of bytes read
        end_ptr = line.as_ptr() as usize + line.len();

        if line.is_empty() {
            continue;
        }

        // line is a comment -> skip
        if is_comment_line(line, comment_prefix) {
            continue;
        }

        let len = line.len();
        if len > 1 {
            // remove carriage return
            let trailing_byte = line[len - 1];
            if trailing_byte == b'\r' {
                line = &line[..len - 1];
            }
        }

        let mut record = SplitFields::new(line, separator, quote_char, eol_char);

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
                            column_types[i].insert(infer_field_schema(&s, try_parse_dates));
                        },
                        Some(NullValues::AllColumns(names)) => {
                            if !names.iter().any(|nv| nv == s.as_ref()) {
                                column_types[i].insert(infer_field_schema(&s, try_parse_dates));
                            }
                        },
                        Some(NullValues::AllColumnsSingle(name)) => {
                            if s.as_ref() != name {
                                column_types[i].insert(infer_field_schema(&s, try_parse_dates));
                            }
                        },
                        Some(NullValues::Named(names)) => {
                            let current_name = &headers[i];
                            let null_name = &names.iter().find(|name| &name.0 == current_name);

                            if let Some(null_name) = null_name {
                                if null_name.1 != s.as_ref() {
                                    column_types[i].insert(infer_field_schema(&s, try_parse_dates));
                                }
                            } else {
                                column_types[i].insert(infer_field_schema(&s, try_parse_dates));
                            }
                        },
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

            // column might have been renamed
            // execute only if schema is complete
            if schema_overwrite.len() == header_length {
                if let Some((name, dtype)) = schema_overwrite.get_at_index(i) {
                    fields.push(Field::new(name, dtype.clone()));
                    continue;
                }
            }
        }

        // determine data type based on possible types
        // if there are incompatible types, use DataType::Utf8
        match possibilities.len() {
            1 => {
                for dtype in possibilities.iter() {
                    fields.push(Field::new(field_name, dtype.clone()));
                }
            },
            2 => {
                if possibilities.contains(&DataType::Int64)
                    && possibilities.contains(&DataType::Float64)
                {
                    // we have an integer and double, fall down to double
                    fields.push(Field::new(field_name, DataType::Float64));
                }
                // prefer a datelike parse above a no parse so choose the date type
                else if possibilities.contains(&DataType::Utf8)
                    && possibilities.contains(&DataType::Date)
                {
                    fields.push(Field::new(field_name, DataType::Date));
                }
                // prefer a datelike parse above a no parse so choose the date type
                else if possibilities.contains(&DataType::Utf8)
                    && possibilities.contains(&DataType::Datetime(TimeUnit::Microseconds, None))
                {
                    fields.push(Field::new(
                        field_name,
                        DataType::Datetime(TimeUnit::Microseconds, None),
                    ));
                } else {
                    // default to Utf8 for conflicting datatypes (e.g bool and int)
                    fields.push(Field::new(field_name, DataType::Utf8));
                }
            },
            _ => fields.push(Field::new(field_name, DataType::Utf8)),
        }
    }
    // if there is a single line after the header without an eol
    // we copy the bytes add an eol and rerun this function
    // so that the inference is consistent with and without eol char
    if rows_count == 0
        && !reader_bytes.is_empty()
        && reader_bytes[reader_bytes.len() - 1] != eol_char
        && recursion_count == 0
    {
        let mut rb = Vec::with_capacity(reader_bytes.len() + 1);
        rb.extend_from_slice(reader_bytes);
        rb.push(eol_char);
        return infer_file_schema_inner(
            &ReaderBytes::Owned(rb),
            separator,
            max_read_rows,
            has_header,
            schema_overwrite,
            skip_rows,
            skip_rows_after_header,
            comment_prefix,
            quote_char,
            eol_char,
            null_values,
            try_parse_dates,
            recursion_count + 1,
            raise_if_empty,
        );
    }

    Ok((Schema::from_iter(fields), rows_count, end_ptr - start_ptr))
}

/// Infer the schema of a CSV file by reading through the first n rows of the file,
/// with `max_read_rows` controlling the maximum number of rows to read.
///
/// If `max_read_rows` is not set, the whole file is read to infer its schema.
///
/// Returns
///     - inferred schema
///     - number of rows used for inference.
///     - bytes read
#[allow(clippy::too_many_arguments)]
pub fn infer_file_schema(
    reader_bytes: &ReaderBytes,
    separator: u8,
    max_read_rows: Option<usize>,
    has_header: bool,
    schema_overwrite: Option<&Schema>,
    // we take &mut because we maybe need to skip more rows dependent
    // on the schema inference
    skip_rows: &mut usize,
    skip_rows_after_header: usize,
    comment_prefix: Option<&CommentPrefix>,
    quote_char: Option<u8>,
    eol_char: u8,
    null_values: Option<&NullValues>,
    try_parse_dates: bool,
    raise_if_empty: bool,
) -> PolarsResult<(Schema, usize, usize)> {
    infer_file_schema_inner(
        reader_bytes,
        separator,
        max_read_rows,
        has_header,
        schema_overwrite,
        skip_rows,
        skip_rows_after_header,
        comment_prefix,
        quote_char,
        eol_char,
        null_values,
        try_parse_dates,
        0,
        raise_if_empty,
    )
}

// magic numbers
const GZIP: [u8; 2] = [31, 139];
const ZLIB0: [u8; 2] = [0x78, 0x01];
const ZLIB1: [u8; 2] = [0x78, 0x9C];
const ZLIB2: [u8; 2] = [0x78, 0xDA];
const ZSTD: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

/// check if csv file is compressed
pub fn is_compressed(bytes: &[u8]) -> bool {
    bytes.starts_with(&ZLIB0)
        || bytes.starts_with(&ZLIB1)
        || bytes.starts_with(&ZLIB2)
        || bytes.starts_with(&GZIP)
        || bytes.starts_with(&ZSTD)
}

#[cfg(any(feature = "decompress", feature = "decompress-fast"))]
fn decompress_impl<R: Read>(
    decoder: &mut R,
    n_rows: Option<usize>,
    separator: u8,
    quote_char: Option<u8>,
    eol_char: u8,
) -> Option<Vec<u8>> {
    let chunk_size = 4096;
    Some(match n_rows {
        None => {
            // decompression in a preallocated buffer does not work with zlib-ng
            // and will put the original compressed data in the buffer.
            let mut out = Vec::new();
            decoder.read_to_end(&mut out).ok()?;
            out
        },
        Some(n_rows) => {
            // we take the first rows first '\n\
            let mut out = vec![];
            let mut expected_fields = 0;
            // make sure that we have enough bytes to decode the header (even if it has embedded new line chars)
            // those extra bytes in the buffer don't matter, we don't need to track them
            loop {
                let read = decoder.take(chunk_size).read_to_end(&mut out).ok()?;
                if read == 0 {
                    break;
                }
                if next_line_position_naive(&out, eol_char).is_some() {
                    // an extra shot
                    let read = decoder.take(chunk_size).read_to_end(&mut out).ok()?;
                    if read == 0 {
                        break;
                    }
                    // now that we have enough, we compute the number of fields (also takes embedding into account)
                    expected_fields =
                        SplitFields::new(&out, separator, quote_char, eol_char).count();
                    break;
                }
            }

            let mut line_count = 0;
            let mut buf_pos = 0;
            // keep decoding bytes and count lines
            // keep track of the n_rows we read
            while line_count < n_rows {
                match next_line_position(
                    &out[buf_pos + 1..],
                    Some(expected_fields),
                    separator,
                    quote_char,
                    eol_char,
                ) {
                    Some(pos) => {
                        line_count += 1;
                        buf_pos += pos;
                    },
                    None => {
                        // take more bytes so that we might find a new line the next iteration
                        let read = decoder.take(chunk_size).read_to_end(&mut out).ok()?;
                        // we depleted the reader
                        if read == 0 {
                            break;
                        }
                        continue;
                    },
                };
            }
            out
        },
    })
}

#[cfg(any(feature = "decompress", feature = "decompress-fast"))]
pub(crate) fn decompress(
    bytes: &[u8],
    n_rows: Option<usize>,
    separator: u8,
    quote_char: Option<u8>,
    eol_char: u8,
) -> Option<Vec<u8>> {
    if bytes.starts_with(&GZIP) {
        let mut decoder = flate2::read::MultiGzDecoder::new(bytes);
        decompress_impl(&mut decoder, n_rows, separator, quote_char, eol_char)
    } else if bytes.starts_with(&ZLIB0) || bytes.starts_with(&ZLIB1) || bytes.starts_with(&ZLIB2) {
        let mut decoder = flate2::read::ZlibDecoder::new(bytes);
        decompress_impl(&mut decoder, n_rows, separator, quote_char, eol_char)
    } else if bytes.starts_with(&ZSTD) {
        let mut decoder = zstd::Decoder::new(bytes).ok()?;
        decompress_impl(&mut decoder, n_rows, separator, quote_char, eol_char)
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
///     - bytes length > 1.
pub(super) unsafe fn escape_field(bytes: &[u8], quote: u8, buf: &mut [MaybeUninit<u8>]) -> usize {
    debug_assert!(bytes.len() > 1);
    let mut prev_quote = false;

    let mut count = 0;
    for c in bytes.get_unchecked(1..bytes.len() - 1) {
        if *c == quote {
            if prev_quote {
                prev_quote = false;
                buf.get_unchecked_mut(count).write(*c);
                count += 1;
            } else {
                prev_quote = true;
            }
        } else {
            prev_quote = false;
            buf.get_unchecked_mut(count).write(*c);
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
        assert!(FLOAT_RE.is_match("-7e-05"));
        assert!(FLOAT_RE.is_match("7e-05"));
        assert!(FLOAT_RE.is_match("+7e+05"));
    }

    #[test]
    fn test_get_file_chunks() {
        let path = "../../examples/datasets/foods1.csv";
        let s = std::fs::read_to_string(path).unwrap();
        let bytes = s.as_bytes();
        // can be within -1 / +1 bounds.
        assert!((get_file_chunks(bytes, 10, 4, b',', None, b'\n').len() as i32 - 10).abs() <= 1);
        assert!((get_file_chunks(bytes, 8, 4, b',', None, b'\n').len() as i32 - 8).abs() <= 1);
    }
}
