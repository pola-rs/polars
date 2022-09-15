use std::io::Write;

use arrow::temporal_conversions;
use lexical_core::{FormattedSize, ToLexical};
use memchr::{memchr, memchr2};
use polars_core::prelude::*;
use polars_core::series::SeriesIter;
use polars_core::POOL;
use polars_utils::contention_pool::LowContentionPool;
use rayon::prelude::*;

fn fmt_and_escape_str(f: &mut Vec<u8>, v: &str, options: &SerializeOptions) -> std::io::Result<()> {
    if v.is_empty() {
        write!(f, "\"\"")
    } else {
        let needs_escaping = memchr(options.quote, v.as_bytes()).is_some();

        if needs_escaping {
            let replaced = unsafe {
                // replace from single quote "
                // to double quote ""
                v.replace(
                    std::str::from_utf8_unchecked(&[options.quote]),
                    std::str::from_utf8_unchecked(&[options.quote, options.quote]),
                )
            };
            return write!(f, "\"{}\"", replaced);
        }
        let surround_with_quotes = memchr2(options.delimiter, b'\n', v.as_bytes()).is_some();

        if surround_with_quotes {
            write!(f, "\"{}\"", v)
        } else {
            write!(f, "{}", v)
        }
    }
}

fn fast_float_write<N: ToLexical>(f: &mut Vec<u8>, n: N, write_size: usize) -> std::io::Result<()> {
    let len = f.len();
    f.reserve(write_size);
    unsafe {
        let buffer = std::slice::from_raw_parts_mut(f.as_mut_ptr().add(len), write_size);
        let written_n = n.to_lexical(buffer).len();
        f.set_len(len + written_n);
    }
    Ok(())
}

fn write_anyvalue(f: &mut Vec<u8>, value: AnyValue, options: &SerializeOptions) {
    match value {
        AnyValue::Null => write!(f, "{}", &options.null),
        AnyValue::Int8(v) => write!(f, "{}", v),
        AnyValue::Int16(v) => write!(f, "{}", v),
        AnyValue::Int32(v) => write!(f, "{}", v),
        AnyValue::Int64(v) => write!(f, "{}", v),
        AnyValue::UInt8(v) => write!(f, "{}", v),
        AnyValue::UInt16(v) => write!(f, "{}", v),
        AnyValue::UInt32(v) => write!(f, "{}", v),
        AnyValue::UInt64(v) => write!(f, "{}", v),
        AnyValue::Float32(v) => match &options.float_precision {
            None => fast_float_write(f, v, f32::FORMATTED_SIZE_DECIMAL),
            Some(precision) => write!(f, "{v:.precision$}", v = v, precision = precision),
        },
        AnyValue::Float64(v) => match &options.float_precision {
            None => fast_float_write(f, v, f64::FORMATTED_SIZE_DECIMAL),
            Some(precision) => write!(f, "{v:.precision$}", v = v, precision = precision),
        },
        AnyValue::Boolean(v) => write!(f, "{}", v),
        AnyValue::Utf8(v) => fmt_and_escape_str(f, v, options),
        #[cfg(feature = "dtype-categorical")]
        AnyValue::Categorical(idx, rev_map) => {
            let v = rev_map.get(idx);
            fmt_and_escape_str(f, v, options)
        }
        #[cfg(feature = "dtype-date")]
        AnyValue::Date(v) => {
            let date = temporal_conversions::date32_to_date(v);
            match &options.date_format {
                None => write!(f, "{}", date),
                Some(fmt) => write!(f, "{}", date.format(fmt)),
            }
        }
        #[cfg(feature = "dtype-datetime")]
        AnyValue::Datetime(v, tu, tz) => match tz {
            None => {
                let dt = match tu {
                    TimeUnit::Nanoseconds => temporal_conversions::timestamp_ns_to_datetime(v),
                    TimeUnit::Microseconds => temporal_conversions::timestamp_us_to_datetime(v),
                    TimeUnit::Milliseconds => temporal_conversions::timestamp_ms_to_datetime(v),
                };
                match &options.datetime_format {
                    None => write!(f, "{}", dt),
                    Some(fmt) => write!(f, "{}", dt.format(fmt)),
                }
            }
            Some(tz) => {
                let tz = temporal_conversions::parse_offset(tz).unwrap();

                let dt = temporal_conversions::timestamp_to_datetime(v, tu.to_arrow(), &tz);
                match &options.datetime_format {
                    None => write!(f, "{}", dt),
                    Some(fmt) => write!(f, "{}", dt.format(fmt)),
                }
            }
        },
        #[cfg(feature = "dtype-time")]
        AnyValue::Time(v) => {
            let date = temporal_conversions::time64ns_to_time(v);
            match &options.time_format {
                None => write!(f, "{}", date),
                Some(fmt) => write!(f, "{}", date.format(fmt)),
            }
        }
        dt => panic!("DataType: {} not supported in writing to csv", dt),
    }
    .unwrap();
}

/// Options to serialize logical types to CSV
/// The default is to format times and dates as `chrono` crate formats them.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct SerializeOptions {
    /// used for [`DataType::Date`]
    pub date_format: Option<String>,
    /// used for [`DataType::Time`]
    pub time_format: Option<String>,
    /// used for [`DataType::Datetime]
    pub datetime_format: Option<String>,
    /// used for [`DataType::Float64`] and [`DataType::Float32`]
    pub float_precision: Option<usize>,
    /// used as separator/delimiter
    pub delimiter: u8,
    /// quoting character
    pub quote: u8,
    /// null value representation
    pub null: String,
}

impl Default for SerializeOptions {
    fn default() -> Self {
        SerializeOptions {
            date_format: None,
            time_format: None,
            datetime_format: None,
            float_precision: None,
            delimiter: b',',
            quote: b'"',
            null: String::new(),
        }
    }
}

/// Utility to write to `&mut Vec<u8>` buffer
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
    options: &mut SerializeOptions,
) -> PolarsResult<()> {
    // check that the double quote is valid utf8
    std::str::from_utf8(&[options.quote, options.quote])
        .map_err(|_| PolarsError::ComputeError("quote char leads invalid utf8".into()))?;
    let delimiter = char::from(options.delimiter);

    // if datetime format not specified, infer the maximum required precision
    if options.datetime_format.is_none() {
        for col in df.get_columns() {
            match col.dtype() {
                DataType::Datetime(TimeUnit::Microseconds, _)
                    if options.datetime_format.is_none() =>
                {
                    options.datetime_format = Some("%FT%H:%M:%S.%6f".to_string());
                }
                DataType::Datetime(TimeUnit::Nanoseconds, _) => {
                    options.datetime_format = Some("%FT%H:%M:%S.%9f".to_string());
                    break; // highest precision; no need to check further
                }
                _ => {}
            }
        }
        // if still not set, no cols require higher precision than "ms" (or no datetime cols)
        if options.datetime_format.is_none() {
            options.datetime_format = Some("%FT%H:%M:%S.%3f".to_string());
        }
    }

    let len = df.height();
    let n_threads = POOL.current_num_threads();
    let total_rows_per_pool_iter = n_threads * chunk_size;
    let any_value_iter_pool = LowContentionPool::<Vec<_>>::new(n_threads);
    let write_buffer_pool = LowContentionPool::<Vec<_>>::new(n_threads);

    let mut n_rows_finished = 0;

    // holds the buffers that will be written
    let mut result_buf = Vec::with_capacity(n_threads);
    while n_rows_finished < len {
        let par_iter = (0..n_threads).into_par_iter().map(|thread_no| {
            let thread_offset = thread_no * chunk_size;
            let total_offset = n_rows_finished + thread_offset;
            let df = df.slice(total_offset as i64, chunk_size);
            let cols = df.get_columns();

            // Safety:
            // the bck thinks the lifetime is bounded to write_buffer_pool, but at the time we return
            // the vectors the buffer pool, the series have already been removed from the buffers
            // in other words, the lifetime does not leave this scope
            let cols = unsafe { std::mem::transmute::<&Vec<Series>, &Vec<Series>>(cols) };
            let mut write_buffer = write_buffer_pool.get();

            // don't use df.empty, won't work if there are columns.
            if df.height() == 0 {
                return write_buffer;
            }

            let any_value_iters = cols.iter().map(|s| s.iter());
            let mut col_iters = any_value_iter_pool.get();
            col_iters.extend(any_value_iters);

            let last_ptr = &col_iters[col_iters.len() - 1] as *const SeriesIter;
            let mut finished = false;
            // loop rows
            while !finished {
                for col in &mut col_iters {
                    match col.next() {
                        Some(value) => {
                            write_anyvalue(&mut write_buffer, value, options);
                        }
                        None => {
                            finished = true;
                            break;
                        }
                    }
                    let current_ptr = col as *const SeriesIter;
                    if current_ptr != last_ptr {
                        write!(&mut write_buffer, "{}", delimiter).unwrap()
                    }
                }
                if !finished {
                    writeln!(&mut write_buffer).unwrap();
                }
            }

            // return buffers to the pool
            col_iters.clear();
            any_value_iter_pool.set(col_iters);

            write_buffer
        });

        // rayon will ensure the right order
        result_buf.par_extend(par_iter);

        for mut buf in result_buf.drain(..) {
            let _ = writer.write(&buf)?;
            buf.clear();
            write_buffer_pool.set(buf);
        }

        n_rows_finished += total_rows_per_pool_iter;
    }

    Ok(())
}
/// Writes a CSV header to `writer`
pub(crate) fn write_header<W: Write>(
    writer: &mut W,
    names: &[&str],
    options: &SerializeOptions,
) -> PolarsResult<()> {
    writer.write_all(
        names
            .join(std::str::from_utf8(&[options.delimiter]).unwrap())
            .as_bytes(),
    )?;
    writer.write_all(&[b'\n'])?;
    Ok(())
}
