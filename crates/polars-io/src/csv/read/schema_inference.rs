use polars_buffer::Buffer;
use polars_core::prelude::*;
#[cfg(feature = "polars-time")]
use polars_time::chunkedarray::string::infer as date_infer;
#[cfg(feature = "polars-time")]
use polars_time::prelude::string::Pattern;
use polars_utils::format_pl_smallstr;

use super::splitfields::SplitFields;
use super::{CsvParseOptions, NullValues};
use crate::utils::{BOOLEAN_RE, FLOAT_RE, FLOAT_RE_DECIMAL, INTEGER_RE};

/// Low-level CSV schema inference function.
///
/// Use `read_until_start_and_infer_schema` instead.
#[allow(clippy::too_many_arguments)]
pub(super) fn infer_file_schema_impl(
    header_line: &Option<Buffer<u8>>,
    content_lines: &[Buffer<u8>],
    infer_all_as_str: bool,
    parse_options: &CsvParseOptions,
    schema_overwrite: Option<&Schema>,
) -> Schema {
    let mut headers = header_line
        .as_ref()
        .map(|line| infer_headers(line, parse_options))
        .unwrap_or_else(|| Vec::with_capacity(8));

    let extend_header_with_unknown_column = header_line.is_none();

    let mut column_types = vec![PlHashSet::<DataType>::with_capacity(4); headers.len()];
    let mut nulls = vec![false; headers.len()];

    for content_line in content_lines {
        infer_types_from_line(
            content_line,
            infer_all_as_str,
            &mut headers,
            extend_header_with_unknown_column,
            parse_options,
            &mut column_types,
            &mut nulls,
        );
    }

    build_schema(&headers, &column_types, schema_overwrite)
}

fn infer_headers(mut header_line: &[u8], parse_options: &CsvParseOptions) -> Vec<PlSmallStr> {
    let len = header_line.len();

    if header_line.last().copied() == Some(b'\r') {
        header_line = &header_line[..len - 1];
    }

    let byterecord = SplitFields::new(
        header_line,
        parse_options.separator,
        parse_options.quote_char,
        parse_options.eol_char,
    );

    let headers = byterecord
        .map(|(slice, needs_escaping)| {
            let slice_escaped = if needs_escaping && (slice.len() >= 2) {
                &slice[1..(slice.len() - 1)]
            } else {
                slice
            };
            String::from_utf8_lossy(slice_escaped)
        })
        .collect::<Vec<_>>();

    let mut deduplicated_headers = Vec::with_capacity(headers.len());
    let mut header_names = PlHashMap::with_capacity(headers.len());

    for name in &headers {
        let count = header_names.entry(name.as_ref()).or_insert(0usize);
        if *count != 0 {
            deduplicated_headers.push(format_pl_smallstr!("{}_duplicated_{}", name, *count - 1))
        } else {
            deduplicated_headers.push(PlSmallStr::from_str(name))
        }
        *count += 1;
    }

    deduplicated_headers
}

fn infer_types_from_line(
    mut line: &[u8],
    infer_all_as_str: bool,
    headers: &mut Vec<PlSmallStr>,
    extend_header_with_unknown_column: bool,
    parse_options: &CsvParseOptions,
    column_types: &mut Vec<PlHashSet<DataType>>,
    nulls: &mut Vec<bool>,
) {
    let line_len = line.len();
    if line.last().copied() == Some(b'\r') {
        line = &line[..line_len - 1];
    }

    let record = SplitFields::new(
        line,
        parse_options.separator,
        parse_options.quote_char,
        parse_options.eol_char,
    );

    for (i, (slice, needs_escaping)) in record.enumerate() {
        if i >= headers.len() {
            if extend_header_with_unknown_column {
                headers.push(column_name(i));
                column_types.push(Default::default());
                nulls.push(false);
            } else {
                break;
            }
        }

        if infer_all_as_str {
            column_types[i].insert(DataType::String);
            continue;
        }

        if slice.is_empty() {
            nulls[i] = true;
        } else {
            let slice_escaped = if needs_escaping && (slice.len() >= 2) {
                &slice[1..(slice.len() - 1)]
            } else {
                slice
            };
            let s = String::from_utf8_lossy(slice_escaped);
            let dtype = match &parse_options.null_values {
                None => Some(infer_field_schema(
                    &s,
                    parse_options.try_parse_dates,
                    parse_options.decimal_comma,
                )),
                Some(NullValues::AllColumns(names)) => {
                    if !names.iter().any(|nv| nv == s.as_ref()) {
                        Some(infer_field_schema(
                            &s,
                            parse_options.try_parse_dates,
                            parse_options.decimal_comma,
                        ))
                    } else {
                        None
                    }
                },
                Some(NullValues::AllColumnsSingle(name)) => {
                    if s.as_ref() != name.as_str() {
                        Some(infer_field_schema(
                            &s,
                            parse_options.try_parse_dates,
                            parse_options.decimal_comma,
                        ))
                    } else {
                        None
                    }
                },
                Some(NullValues::Named(names)) => {
                    let current_name = &headers[i];
                    let null_name = &names.iter().find(|name| name.0 == current_name);

                    if let Some(null_name) = null_name {
                        if null_name.1.as_str() != s.as_ref() {
                            Some(infer_field_schema(
                                &s,
                                parse_options.try_parse_dates,
                                parse_options.decimal_comma,
                            ))
                        } else {
                            None
                        }
                    } else {
                        Some(infer_field_schema(
                            &s,
                            parse_options.try_parse_dates,
                            parse_options.decimal_comma,
                        ))
                    }
                },
            };
            if let Some(dtype) = dtype {
                column_types[i].insert(dtype);
            }
        }
    }
}

fn build_schema(
    headers: &[PlSmallStr],
    column_types: &[PlHashSet<DataType>],
    schema_overwrite: Option<&Schema>,
) -> Schema {
    assert!(headers.len() == column_types.len());

    let get_schema_overwrite = |field_name| {
        if let Some(schema_overwrite) = schema_overwrite {
            // Apply schema_overwrite by column name only. Positional overrides are handled
            // separately via dtype_overwrite.
            if let Some((_, name, dtype)) = schema_overwrite.get_full(field_name) {
                return Some((name.clone(), dtype.clone()));
            }
        }

        None
    };

    Schema::from_iter(
        headers
            .iter()
            .zip(column_types)
            .map(|(field_name, type_possibilities)| {
                let (name, dtype) = get_schema_overwrite(field_name).unwrap_or_else(|| {
                    (
                        field_name.clone(),
                        finish_infer_field_schema(type_possibilities),
                    )
                });

                Field::new(name, dtype)
            }),
    )
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
    let bytes = string.as_bytes();
    if bytes.len() >= 2 && *bytes.first().unwrap() == b'"' && *bytes.last().unwrap() == b'"' {
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
                            DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC))
                        },
                        Pattern::Time => DataType::Time,
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
                        DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC))
                    },
                    Pattern::Time => DataType::Time,
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

fn column_name(i: usize) -> PlSmallStr {
    format_pl_smallstr!("column_{}", i + 1)
}
