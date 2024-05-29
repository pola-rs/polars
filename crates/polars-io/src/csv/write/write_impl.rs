mod serializer;

use std::io::Write;

use arrow::array::NullArray;
use arrow::legacy::time_zone::Tz;
use polars_core::prelude::*;
use polars_core::POOL;
use polars_error::polars_ensure;
use polars_utils::contention_pool::LowContentionPool;
use rayon::prelude::*;
use serializer::{serializer_for, string_serializer};

use crate::csv::write::SerializeOptions;

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

    let len = df.height();
    let total_rows_per_pool_iter = n_threads * chunk_size;
    let serializer_pool = LowContentionPool::<Vec<_>>::new(n_threads);
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

            // SAFETY:
            // the bck thinks the lifetime is bounded to write_buffer_pool, but at the time we return
            // the vectors the buffer pool, the series have already been removed from the buffers
            // in other words, the lifetime does not leave this scope
            let cols = unsafe { std::mem::transmute::<&[Series], &[Series]>(cols) };
            let mut write_buffer = write_buffer_pool.get();

            if df.is_empty() {
                return Ok(write_buffer);
            }

            let mut serializers_vec = serializer_pool.get();
            if serializers_vec.is_empty() {
                serializers_vec = cols
                    .iter()
                    .enumerate()
                    .map(|(i, col)| {
                        serializer_for(
                            &*col.chunks()[0],
                            options,
                            col.dtype(),
                            datetime_formats[i],
                            time_zones[i],
                        )
                    })
                    .collect::<Result<_, _>>()?;
            } else {
                debug_assert_eq!(serializers_vec.len(), cols.len());
                for (col_iter, col) in std::iter::zip(&mut serializers_vec, cols) {
                    col_iter.update_array(&*col.chunks()[0]);
                }
            }

            let serializers = serializers_vec.as_mut_slice();

            let len = std::cmp::min(cols[0].len(), chunk_size);

            for _ in 0..len {
                serializers[0].serialize(&mut write_buffer, options);
                for serializer in &mut serializers[1..] {
                    write_buffer.push(options.separator);
                    serializer.serialize(&mut write_buffer, options);
                }

                write_buffer.extend_from_slice(options.line_terminator.as_bytes());
            }

            serializer_pool.set(serializers_vec);

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
            writer.write_all(&buf)?;
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
    let mut header = Vec::new();

    // A hack, but it works for this case.
    let fake_arr = NullArray::new(ArrowDataType::Null, 0);
    let mut names_serializer = string_serializer(
        |iter: &mut std::slice::Iter<&str>| iter.next().copied(),
        options,
        |_| names.iter(),
        &fake_arr,
    );
    for i in 0..names.len() {
        names_serializer.serialize(&mut header, options);
        if i != names.len() - 1 {
            header.push(options.separator);
        }
    }
    header.extend_from_slice(options.line_terminator.as_bytes());
    writer.write_all(&header)?;
    Ok(())
}

/// Writes a UTF-8 BOM to `writer`.
pub(crate) fn write_bom<W: Write>(writer: &mut W) -> PolarsResult<()> {
    const BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];
    writer.write_all(&BOM)?;
    Ok(())
}
