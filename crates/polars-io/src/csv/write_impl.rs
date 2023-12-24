use std::io::Write;

use arrow::legacy::time_zone::Tz;
#[cfg(any(
    feature = "dtype-date",
    feature = "dtype-time",
    feature = "dtype-datetime"
))]
use arrow::temporal_conversions;
#[cfg(feature = "timezones")]
use chrono::TimeZone;
use memchr::{memchr, memchr2};
use polars_core::prelude::*;
use polars_core::series::SeriesIter;
use polars_core::POOL;
use polars_utils::contention_pool::LowContentionPool;
use rayon::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::write::QuoteStyle;

fn fmt_and_escape_str(f: &mut Vec<u8>, v: &str, options: &SerializeOptions) -> std::io::Result<()> {
    if options.quote_style == QuoteStyle::Never {
        return write!(f, "{v}");
    }
    let quote = options.quote_char as char;
    if v.is_empty() {
        return write!(f, "{quote}{quote}");
    }
    let needs_escaping = memchr(options.quote_char, v.as_bytes()).is_some();
    if needs_escaping {
        let replaced = unsafe {
            // Replace from single quote " to double quote "".
            v.replace(
                std::str::from_utf8_unchecked(&[options.quote_char]),
                std::str::from_utf8_unchecked(&[options.quote_char, options.quote_char]),
            )
        };
        return write!(f, "{quote}{replaced}{quote}");
    }
    let surround_with_quotes = match options.quote_style {
        QuoteStyle::Always | QuoteStyle::NonNumeric => true,
        QuoteStyle::Necessary => memchr2(options.separator, b'\n', v.as_bytes()).is_some(),
        QuoteStyle::Never => false,
    };

    if surround_with_quotes {
        write!(f, "{quote}{v}{quote}")
    } else {
        write!(f, "{v}")
    }
}

fn fast_float_write<I: ryu::Float>(f: &mut Vec<u8>, val: I) {
    let mut buffer = ryu::Buffer::new();
    let value = buffer.format(val);
    f.extend_from_slice(value.as_bytes())
}

fn write_integer<I: itoa::Integer>(f: &mut Vec<u8>, val: I) {
    let mut buffer = itoa::Buffer::new();
    let value = buffer.format(val);
    f.extend_from_slice(value.as_bytes())
}

#[allow(unused_variables)]
unsafe fn write_anyvalue(
    f: &mut Vec<u8>,
    value: AnyValue,
    options: &SerializeOptions,
    datetime_formats: &[&str],
    time_zones: &[Option<Tz>],
    i: usize,
) -> PolarsResult<()> {
    match value {
        // First do the string-like types as they know how to deal with quoting.
        AnyValue::Utf8(v) => {
            fmt_and_escape_str(f, v, options)?;
            Ok(())
        },
        #[cfg(feature = "dtype-categorical")]
        AnyValue::Categorical(idx, rev_map, _) => {
            let v = rev_map.get(idx);
            fmt_and_escape_str(f, v, options)?;
            Ok(())
        },
        _ => {
            // Then we deal with the numeric types
            let quote = options.quote_char as char;

            let mut end_with_quote = matches!(options.quote_style, QuoteStyle::Always);
            if end_with_quote {
                // start the quote
                write!(f, "{quote}")?
            }

            match value {
                AnyValue::Null => write!(f, "{}", &options.null),
                AnyValue::Int8(v) => {
                    write_integer(f, v);
                    Ok(())
                },
                AnyValue::Int16(v) => {
                    write_integer(f, v);
                    Ok(())
                },
                AnyValue::Int32(v) => {
                    write_integer(f, v);
                    Ok(())
                },
                AnyValue::Int64(v) => {
                    write_integer(f, v);
                    Ok(())
                },
                AnyValue::UInt8(v) => {
                    write_integer(f, v);
                    Ok(())
                },
                AnyValue::UInt16(v) => {
                    write_integer(f, v);
                    Ok(())
                },
                AnyValue::UInt32(v) => {
                    write_integer(f, v);
                    Ok(())
                },
                AnyValue::UInt64(v) => {
                    write_integer(f, v);
                    Ok(())
                },
                AnyValue::Float32(v) => match &options.float_precision {
                    None => {
                        fast_float_write(f, v);
                        Ok(())
                    },
                    Some(precision) => write!(f, "{v:.precision$}"),
                },
                AnyValue::Float64(v) => match &options.float_precision {
                    None => {
                        fast_float_write(f, v);
                        Ok(())
                    },
                    Some(precision) => write!(f, "{v:.precision$}"),
                },
                _ => {
                    // And here we deal with the non-numeric types (excluding strings)
                    if !end_with_quote && matches!(options.quote_style, QuoteStyle::NonNumeric) {
                        // start the quote
                        write!(f, "{quote}")?;
                        end_with_quote = true
                    }

                    match value {
                        AnyValue::Boolean(v) => write!(f, "{v}"),
                        #[cfg(feature = "dtype-date")]
                        AnyValue::Date(v) => {
                            let date = temporal_conversions::date32_to_date(v);
                            match &options.date_format {
                                None => write!(f, "{date}"),
                                Some(fmt) => write!(f, "{}", date.format(fmt)),
                            }
                        },
                        #[cfg(feature = "dtype-datetime")]
                        AnyValue::Datetime(v, tu, tz) => {
                            let datetime_format = { *datetime_formats.get_unchecked(i) };
                            let time_zone = { time_zones.get_unchecked(i) };
                            let ndt = match tu {
                                TimeUnit::Nanoseconds => {
                                    temporal_conversions::timestamp_ns_to_datetime(v)
                                },
                                TimeUnit::Microseconds => {
                                    temporal_conversions::timestamp_us_to_datetime(v)
                                },
                                TimeUnit::Milliseconds => {
                                    temporal_conversions::timestamp_ms_to_datetime(v)
                                },
                            };
                            let formatted = match time_zone {
                                #[cfg(feature = "timezones")]
                                Some(time_zone) => {
                                    time_zone.from_utc_datetime(&ndt).format(datetime_format)
                                },
                                #[cfg(not(feature = "timezones"))]
                                Some(_) => {
                                    panic!("activate 'timezones' feature");
                                },
                                _ => ndt.format(datetime_format),
                            };
                            let str_result = write!(f, "{formatted}");
                            if str_result.is_err() {
                                let datetime_format = unsafe { *datetime_formats.get_unchecked(i) };
                                let type_name = if tz.is_some() {
                                    "DateTime"
                                } else {
                                    "NaiveDateTime"
                                };
                                polars_bail!(
                                    ComputeError: "cannot format {} with format '{}'", type_name, datetime_format,
                                )
                            };
                            str_result
                        },
                        #[cfg(feature = "dtype-time")]
                        AnyValue::Time(v) => {
                            let date = temporal_conversions::time64ns_to_time(v);
                            match &options.time_format {
                                None => write!(f, "{date}"),
                                Some(fmt) => write!(f, "{}", date.format(fmt)),
                            }
                        },
                        ref dt => {
                            polars_bail!(ComputeError: "datatype {} cannot be written to csv", dt)
                        },
                    }
                },
            }
            .map_err(|err| polars_err!(ComputeError: "error writing value {}: {}", value, err))?;

            if end_with_quote {
                write!(f, "{quote}")?
            }
            Ok(())
        },
    }
}

/// Options to serialize logical types to CSV.
///
/// The default is to format times and dates as `chrono` crate formats them.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SerializeOptions {
    /// Used for [`DataType::Date`].
    pub date_format: Option<String>,
    /// Used for [`DataType::Time`].
    pub time_format: Option<String>,
    /// Used for [`DataType::Datetime`].
    pub datetime_format: Option<String>,
    /// Used for [`DataType::Float64`] and [`DataType::Float32`].
    pub float_precision: Option<usize>,
    /// Used as separator.
    pub separator: u8,
    /// Quoting character.
    pub quote_char: u8,
    /// Null value representation.
    pub null: String,
    /// String appended after every row.
    pub line_terminator: String,
    pub quote_style: QuoteStyle,
}

impl Default for SerializeOptions {
    fn default() -> Self {
        SerializeOptions {
            date_format: None,
            time_format: None,
            datetime_format: None,
            float_precision: None,
            separator: b',',
            quote_char: b'"',
            null: String::new(),
            line_terminator: "\n".into(),
            quote_style: Default::default(),
        }
    }
}

/// Utility to write to `&mut Vec<u8>` buffer.
struct StringWrap<'a>(pub &'a mut Vec<u8>);

impl<'a> std::fmt::Write for StringWrap<'a> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

pub(crate) fn write<W: Write>(
    writer: &mut W,
    df: &DataFrame,
    chunk_size: usize,
    options: &SerializeOptions,
    n_threads: usize,
) -> PolarsResult<()> {
    for s in df.get_columns() {
        let nested = match s.dtype() {
            DataType::List(_) => true,
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => true,
            #[cfg(feature = "object")]
            DataType::Object(_, _) => {
                return Err(PolarsError::ComputeError(
                    "csv writer does not support object dtype".into(),
                ))
            },
            _ => false,
        };
        polars_ensure!(
            !nested,
            ComputeError: "CSV format does not support nested data",
        );
    }

    // Check that the double quote is valid UTF-8.
    polars_ensure!(
        std::str::from_utf8(&[options.quote_char, options.quote_char]).is_ok(),
        ComputeError: "quote char results in invalid utf-8",
    );
    let separator = char::from(options.separator);

    let (datetime_formats, time_zones): (Vec<&str>, Vec<Option<Tz>>) = df
        .get_columns()
        .iter()
        .map(|column| match column.dtype() {
            DataType::Datetime(TimeUnit::Milliseconds, tz) => {
                let (format, tz_parsed) = match tz {
                    #[cfg(feature = "timezones")]
                    Some(tz) => (
                        options
                            .datetime_format
                            .as_deref()
                            .unwrap_or("%FT%H:%M:%S.%3f%z"),
                        tz.parse::<Tz>().ok(),
                    ),
                    _ => (
                        options
                            .datetime_format
                            .as_deref()
                            .unwrap_or("%FT%H:%M:%S.%3f"),
                        None,
                    ),
                };
                (format, tz_parsed)
            },
            DataType::Datetime(TimeUnit::Microseconds, tz) => {
                let (format, tz_parsed) = match tz {
                    #[cfg(feature = "timezones")]
                    Some(tz) => (
                        options
                            .datetime_format
                            .as_deref()
                            .unwrap_or("%FT%H:%M:%S.%6f%z"),
                        tz.parse::<Tz>().ok(),
                    ),
                    _ => (
                        options
                            .datetime_format
                            .as_deref()
                            .unwrap_or("%FT%H:%M:%S.%6f"),
                        None,
                    ),
                };
                (format, tz_parsed)
            },
            DataType::Datetime(TimeUnit::Nanoseconds, tz) => {
                let (format, tz_parsed) = match tz {
                    #[cfg(feature = "timezones")]
                    Some(tz) => (
                        options
                            .datetime_format
                            .as_deref()
                            .unwrap_or("%FT%H:%M:%S.%9f%z"),
                        tz.parse::<Tz>().ok(),
                    ),
                    _ => (
                        options
                            .datetime_format
                            .as_deref()
                            .unwrap_or("%FT%H:%M:%S.%9f"),
                        None,
                    ),
                };
                (format, tz_parsed)
            },
            _ => ("", None),
        })
        .unzip();
    let datetime_formats = datetime_formats.into_iter().collect::<Vec<_>>();
    let time_zones = time_zones.into_iter().collect::<Vec<_>>();

    let len = df.height();
    let total_rows_per_pool_iter = n_threads * chunk_size;
    let any_value_iter_pool = LowContentionPool::<Vec<_>>::new(n_threads);
    let write_buffer_pool = LowContentionPool::<Vec<_>>::new(n_threads);

    let mut n_rows_finished = 0;

    // holds the buffers that will be written
    let mut result_buf: Vec<PolarsResult<Vec<u8>>> = Vec::with_capacity(n_threads);

    while n_rows_finished < len {
        let buf_writer = |thread_no| {
            let thread_offset = thread_no * chunk_size;
            let total_offset = n_rows_finished + thread_offset;
            let mut df = df.slice(total_offset as i64, chunk_size);
            // the `series.iter` needs rechunked series.
            // we don't do this on the whole as this probably needs much less rechunking
            // so will be faster.
            // and allows writing `pl.concat([df] * 100, rechunk=False).write_csv()` as the rechunk
            // would go OOM
            df.as_single_chunk();
            let cols = df.get_columns();

            // Safety:
            // the bck thinks the lifetime is bounded to write_buffer_pool, but at the time we return
            // the vectors the buffer pool, the series have already been removed from the buffers
            // in other words, the lifetime does not leave this scope
            let cols = unsafe { std::mem::transmute::<&[Series], &[Series]>(cols) };
            let mut write_buffer = write_buffer_pool.get();

            // don't use df.empty, won't work if there are columns.
            if df.height() == 0 {
                return Ok(write_buffer);
            }

            let any_value_iters = cols.iter().map(|s| s.iter());
            let mut col_iters = any_value_iter_pool.get();
            col_iters.extend(any_value_iters);

            let last_ptr = &col_iters[col_iters.len() - 1] as *const SeriesIter;
            let mut finished = false;
            // loop rows
            while !finished {
                for (i, col) in &mut col_iters.iter_mut().enumerate() {
                    match col.next() {
                        Some(value) => unsafe {
                            write_anyvalue(
                                &mut write_buffer,
                                value,
                                options,
                                &datetime_formats,
                                &time_zones,
                                i,
                            )?;
                        },
                        None => {
                            finished = true;
                            break;
                        },
                    }
                    let current_ptr = col as *const SeriesIter;
                    if current_ptr != last_ptr {
                        write!(&mut write_buffer, "{separator}").unwrap()
                    }
                }
                if !finished {
                    write!(&mut write_buffer, "{}", options.line_terminator).unwrap();
                }
            }

            // return buffers to the pool
            col_iters.clear();
            any_value_iter_pool.set(col_iters);

            Ok(write_buffer)
        };

        if n_threads > 1 {
            let par_iter = (0..n_threads).into_par_iter().map(buf_writer);
            // rayon will ensure the right order
            POOL.install(|| result_buf.par_extend(par_iter));
        } else {
            result_buf.push(buf_writer(0));
        }

        for buf in result_buf.drain(..) {
            let mut buf = buf?;
            let _ = writer.write(&buf)?;
            buf.clear();
            write_buffer_pool.set(buf);
        }

        n_rows_finished += total_rows_per_pool_iter;
    }
    Ok(())
}

/// Writes a CSV header to `writer`.
pub(crate) fn write_header<W: Write>(
    writer: &mut W,
    names: &[&str],
    options: &SerializeOptions,
) -> PolarsResult<()> {
    let mut escaped_names: Vec<String> = Vec::with_capacity(names.len());
    let mut nm: Vec<u8> = vec![];

    for name in names {
        fmt_and_escape_str(&mut nm, name, options)?;
        unsafe {
            // SAFETY: we know headers will be valid UTF-8 at this point
            escaped_names.push(std::str::from_utf8_unchecked(&nm).to_string());
        }
        nm.clear();
    }
    writer.write_all(
        escaped_names
            .join(std::str::from_utf8(&[options.separator]).unwrap())
            .as_bytes(),
    )?;
    writer.write_all(options.line_terminator.as_bytes())?;
    Ok(())
}

/// Writes a UTF-8 BOM to `writer`.
pub(crate) fn write_bom<W: Write>(writer: &mut W) -> PolarsResult<()> {
    const BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];
    writer.write_all(&BOM)?;
    Ok(())
}
