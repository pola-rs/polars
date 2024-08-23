use std::borrow::Cow;

use polars_core::config::verbose;
use polars_core::prelude::*;
#[cfg(feature = "polars-time")]
use polars_time::chunkedarray::string::infer as date_infer;
#[cfg(feature = "polars-time")]
use polars_time::prelude::string::Pattern;
use polars_utils::slice::GetSaferUnchecked;

use super::options::{CommentPrefix, CsvEncoding, NullValues};
use super::parser::{is_comment_line, skip_bom, skip_line_ending, SplitLines};
use super::splitfields::SplitFields;
use super::CsvReadOptions;
use crate::mmap::ReaderBytes;
use crate::utils::{BOOLEAN_RE, FLOAT_RE, FLOAT_RE_DECIMAL, INTEGER_RE};

#[derive(Clone, Debug, Default)]
pub struct SchemaInferenceResult {
    inferred_schema: SchemaRef,
    rows_read: usize,
    bytes_read: usize,
    bytes_total: usize,
    n_threads: Option<usize>,
}

impl SchemaInferenceResult {
    pub fn try_from_reader_bytes_and_options(
        reader_bytes: &ReaderBytes,
        options: &CsvReadOptions,
    ) -> PolarsResult<Self> {
        let parse_options = options.get_parse_options();

        let separator = parse_options.separator;
        let infer_schema_length = options.infer_schema_length;
        let has_header = options.has_header;
        let schema_overwrite_arc = options.schema_overwrite.clone();
        let schema_overwrite = schema_overwrite_arc.as_ref().map(|x| x.as_ref());
        let skip_rows = options.skip_rows;
        let skip_rows_after_header = options.skip_rows_after_header;
        let comment_prefix = parse_options.comment_prefix.as_ref();
        let quote_char = parse_options.quote_char;
        let eol_char = parse_options.eol_char;
        let null_values = parse_options.null_values.clone();
        let try_parse_dates = parse_options.try_parse_dates;
        let raise_if_empty = options.raise_if_empty;
        let mut n_threads = options.n_threads;
        let decimal_comma = parse_options.decimal_comma;

        let bytes_total = reader_bytes.len();

        let (inferred_schema, rows_read, bytes_read) = infer_file_schema(
            reader_bytes,
            separator,
            infer_schema_length,
            has_header,
            schema_overwrite,
            skip_rows,
            skip_rows_after_header,
            comment_prefix,
            quote_char,
            eol_char,
            null_values.as_ref(),
            try_parse_dates,
            raise_if_empty,
            &mut n_threads,
            decimal_comma,
        )?;

        let this = Self {
            inferred_schema: Arc::new(inferred_schema),
            rows_read,
            bytes_read,
            bytes_total,
            n_threads,
        };

        Ok(this)
    }

    pub fn with_inferred_schema(mut self, inferred_schema: SchemaRef) -> Self {
        self.inferred_schema = inferred_schema;
        self
    }

    pub fn get_inferred_schema(&self) -> SchemaRef {
        self.inferred_schema.clone()
    }

    pub fn get_estimated_n_rows(&self) -> usize {
        (self.rows_read as f64 / self.bytes_read as f64 * self.bytes_total as f64) as usize
    }
}

impl CsvReadOptions {
    /// Note: This does not update the schema from the inference result.
    pub fn update_with_inference_result(&mut self, si_result: &SchemaInferenceResult) {
        self.n_threads = si_result.n_threads;
    }
}

pub fn finish_infer_field_schema(possibilities: &PlHashSet<DataType>) -> DataType {
    // determine data type based on possible types
    // if there are incompatible types, use DataType::String
    match possibilities.len() {
        1 => possibilities.iter().next().unwrap().clone(),
        2 if possibilities.contains(&DataType::Int64)
            && possibilities.contains(&DataType::Float64) =>
        {
            // we have an integer and double, fall down to double
            DataType::Float64
        },
        // default to String for conflicting datatypes (e.g bool and int)
        _ => DataType::String,
    }
}

/// Infer the data type of a record
pub fn infer_field_schema(string: &str, try_parse_dates: bool, decimal_comma: bool) -> DataType {
    // when quoting is enabled in the reader, these quotes aren't escaped, we default to
    // String for them
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
                    None => DataType::String,
                }
            }
            #[cfg(not(feature = "polars-time"))]
            {
                panic!("activate one of {{'dtype-date', 'dtype-datetime', dtype-time'}} features")
            }
        } else {
            DataType::String
        }
    }
    // match regex in a particular order
    else if BOOLEAN_RE.is_match(string) {
        DataType::Boolean
    } else if !decimal_comma && FLOAT_RE.is_match(string)
        || decimal_comma && FLOAT_RE_DECIMAL.is_match(string)
    {
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
                None => DataType::String,
            }
        }
        #[cfg(not(feature = "polars-time"))]
        {
            panic!("activate one of {{'dtype-date', 'dtype-datetime', dtype-time'}} features")
        }
    } else {
        DataType::String
    }
}

#[inline]
fn parse_bytes_with_encoding(bytes: &[u8], encoding: CsvEncoding) -> PolarsResult<Cow<str>> {
    Ok(match encoding {
        CsvEncoding::Utf8 => simdutf8::basic::from_utf8(bytes)
            .map_err(|_| polars_err!(ComputeError: "invalid utf-8 sequence"))?
            .into(),
        CsvEncoding::LossyUtf8 => String::from_utf8_lossy(bytes),
    })
}

#[allow(clippy::too_many_arguments)]
fn infer_file_schema_inner(
    reader_bytes: &ReaderBytes,
    separator: u8,
    max_read_rows: Option<usize>,
    has_header: bool,
    schema_overwrite: Option<&Schema>,
    // we take &mut because we maybe need to skip more rows dependent
    // on the schema inference
    mut skip_rows: usize,
    skip_rows_after_header: usize,
    comment_prefix: Option<&CommentPrefix>,
    quote_char: Option<u8>,
    eol_char: u8,
    null_values: Option<&NullValues>,
    try_parse_dates: bool,
    recursion_count: u8,
    raise_if_empty: bool,
    n_threads: &mut Option<usize>,
    decimal_comma: bool,
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
    let mut lines = SplitLines::new(bytes, quote_char.unwrap_or(b'"'), eol_char).skip(skip_rows);

    // get or create header names
    // when has_header is false, creates default column names with column_ prefix

    // skip lines that are comments
    let mut first_line = None;

    for (i, line) in (&mut lines).enumerate() {
        if !is_comment_line(line, comment_prefix) {
            first_line = Some(line);
            skip_rows += i;
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
            byterecord
                .enumerate()
                .map(|(i, _s)| format!("column_{}", i + 1))
                .collect::<Vec<String>>()
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
            n_threads,
            decimal_comma,
        );
    } else if !raise_if_empty {
        return Ok((Schema::new(), 0, 0));
    } else {
        polars_bail!(NoData: "empty CSV");
    };
    if !has_header {
        // re-init lines so that the header is included in type inference.
        lines = SplitLines::new(bytes, quote_char.unwrap_or(b'"'), eol_char).skip(skip_rows);
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
                    unsafe { *nulls.get_unchecked_release_mut(i) = true };
                } else {
                    let slice_escaped = if needs_escaping && (slice.len() >= 2) {
                        &slice[1..(slice.len() - 1)]
                    } else {
                        slice
                    };
                    let s = parse_bytes_with_encoding(slice_escaped, encoding)?;
                    let dtype = match &null_values {
                        None => Some(infer_field_schema(&s, try_parse_dates, decimal_comma)),
                        Some(NullValues::AllColumns(names)) => {
                            if !names.iter().any(|nv| nv == s.as_ref()) {
                                Some(infer_field_schema(&s, try_parse_dates, decimal_comma))
                            } else {
                                None
                            }
                        },
                        Some(NullValues::AllColumnsSingle(name)) => {
                            if s.as_ref() != name {
                                Some(infer_field_schema(&s, try_parse_dates, decimal_comma))
                            } else {
                                None
                            }
                        },
                        Some(NullValues::Named(names)) => {
                            // SAFETY:
                            // we iterate over headers length.
                            let current_name = unsafe { headers.get_unchecked_release(i) };
                            let null_name = &names.iter().find(|name| &name.0 == current_name);

                            if let Some(null_name) = null_name {
                                if null_name.1 != s.as_ref() {
                                    Some(infer_field_schema(&s, try_parse_dates, decimal_comma))
                                } else {
                                    None
                                }
                            } else {
                                Some(infer_field_schema(&s, try_parse_dates, decimal_comma))
                            }
                        },
                    };
                    if let Some(dtype) = dtype {
                        if matches!(&dtype, DataType::String)
                            && needs_escaping
                            && n_threads.unwrap_or(2) > 1
                        {
                            // The parser will chunk the file.
                            // However this will be increasingly unlikely to be correct if there are many
                            // new line characters in an escaped field. So we set a (somewhat arbitrary)
                            // upper bound to the number of escaped lines we accept.
                            // On the chunking side we also have logic to make this more robust.
                            if slice.iter().filter(|b| **b == eol_char).count() > 8 {
                                if verbose() {
                                    eprintln!("falling back to single core reading because of many escaped new line chars.")
                                }
                                *n_threads = Some(1);
                            }
                        }
                        unsafe { column_types.get_unchecked_release_mut(i).insert(dtype) };
                    }
                }
            }
        }
    }

    // build schema from inference results
    for i in 0..header_length {
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

        let possibilities = &column_types[i];
        let dtype = finish_infer_field_schema(possibilities);
        fields.push(Field::new(field_name, dtype));
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
            n_threads,
            decimal_comma,
        );
    }

    Ok((Schema::from_iter(fields), rows_count, end_ptr - start_ptr))
}

pub(super) fn check_decimal_comma(decimal_comma: bool, separator: u8) -> PolarsResult<()> {
    if decimal_comma {
        polars_ensure!(b',' != separator, InvalidOperation: "'decimal_comma' argument cannot be combined with ',' separator")
    }
    Ok(())
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
    skip_rows: usize,
    skip_rows_after_header: usize,
    comment_prefix: Option<&CommentPrefix>,
    quote_char: Option<u8>,
    eol_char: u8,
    null_values: Option<&NullValues>,
    try_parse_dates: bool,
    raise_if_empty: bool,
    n_threads: &mut Option<usize>,
    decimal_comma: bool,
) -> PolarsResult<(Schema, usize, usize)> {
    check_decimal_comma(decimal_comma, separator)?;
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
        n_threads,
        decimal_comma,
    )
}
