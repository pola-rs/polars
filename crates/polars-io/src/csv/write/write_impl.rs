mod serializer;

use std::io::Write;

use arrow::array::NullArray;
use arrow::legacy::time_zone::Tz;
use polars_core::POOL;
use polars_core::prelude::*;
use polars_error::{feature_gated, polars_ensure};
use rayon::prelude::*;
use serializer::{serializer_for, string_serializer};

use crate::csv::write::{CsvCompression, SerializeOptions};

pub(crate) fn write<W: Write>(
    writer: &mut W,
    df: &DataFrame,
    chunk_size: usize,
    options: &SerializeOptions,
    compression: Option<CsvCompression>,
    n_threads: usize,
) -> PolarsResult<()> {
    if let Some(method) = compression {
        feature_gated!(
            "compress",
            match method {
                CsvCompression::Gzip(level) => {
                    let level = level.unwrap_or_default();
                    let mut encoder = flate2::write::GzEncoder::new(
                        writer,
                        flate2::Compression::new(level.compression_level() as _),
                    );
                    write_impl(&mut encoder, df, chunk_size, options, n_threads)?;
                    encoder.try_finish().map_err(|e| e.into())
                },
                CsvCompression::Zstd(level) => {
                    let level = level.unwrap_or_default();
                    let mut encoder = zstd::Encoder::new(writer, level.compression_level())?;
                    write_impl(&mut encoder, df, chunk_size, options, n_threads)?;
                    encoder.do_finish().map_err(|e| e.into())
                },
            }
        )
    } else {
        write_impl(writer, df, chunk_size, options, n_threads)
    }
}

pub(crate) fn write_impl<W: Write>(
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
            DataType::Object(_) => {
                return Err(PolarsError::ComputeError(
                    "csv writer does not support object dtype".into(),
                ));
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

    let mut n_rows_finished = 0;

    // To comply with the safety requirements for the buf_writer closure, we need to make sure
    // the column dtype references have a lifetime that exceeds the scope of the serializer, i.e.
    // the full dataframe. If not, we can run into use-after-free memory issues for types that
    // allocate, such as Enum or Categorical dtype (see GH issue #23939).
    let col_dtypes: Vec<_> = df.get_columns().iter().map(|c| c.dtype()).collect();

    let mut buffers: Vec<_> = (0..n_threads).map(|_| (Vec::new(), Vec::new())).collect();
    while n_rows_finished < len {
        let buf_writer = |thread_no, write_buffer: &mut Vec<_>, serializers_vec: &mut Vec<_>| {
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
            let cols = unsafe { std::mem::transmute::<&[Column], &[Column]>(cols) };

            if df.is_empty() {
                return Ok(());
            }

            if serializers_vec.is_empty() {
                debug_assert_eq!(cols.len(), col_dtypes.len());
                *serializers_vec = std::iter::zip(cols, &col_dtypes)
                    .enumerate()
                    .map(|(i, (col, &col_dtype))| {
                        serializer_for(
                            &*col.as_materialized_series().chunks()[0],
                            options,
                            col_dtype,
                            datetime_formats[i],
                            time_zones[i],
                        )
                    })
                    .collect::<Result<_, _>>()?;
            } else {
                debug_assert_eq!(serializers_vec.len(), cols.len());
                for (col_iter, col) in std::iter::zip(serializers_vec.iter_mut(), cols) {
                    col_iter.update_array(&*col.as_materialized_series().chunks()[0]);
                }
            }

            let serializers = serializers_vec.as_mut_slice();

            let len = std::cmp::min(cols[0].len(), chunk_size);

            for _ in 0..len {
                serializers[0].serialize(write_buffer, options);
                for serializer in &mut serializers[1..] {
                    write_buffer.push(options.separator);
                    serializer.serialize(write_buffer, options);
                }

                write_buffer.extend_from_slice(options.line_terminator.as_bytes());
            }

            Ok(())
        };

        if n_threads > 1 {
            POOL.install(|| {
                buffers
                    .par_iter_mut()
                    .enumerate()
                    .map(|(i, (w, s))| buf_writer(i, w, s))
                    .collect::<PolarsResult<()>>()
            })?;
        } else {
            let (w, s) = &mut buffers[0];
            buf_writer(0, w, s)?;
        }

        for (write_buffer, _) in &mut buffers {
            writer.write_all(write_buffer)?;
            write_buffer.clear();
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
    compression: Option<CsvCompression>,
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
    write_preamble(writer, &header, compression)?;
    Ok(())
}

/// Writes a UTF-8 BOM to `writer`.
pub(crate) fn write_bom<W: Write>(
    writer: &mut W,
    compression: Option<CsvCompression>,
) -> PolarsResult<()> {
    const BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];
    write_preamble(writer, &BOM, compression)?;
    Ok(())
}

pub(crate) fn write_preamble<W: Write>(
    writer: &mut W,
    input_buf: &[u8],
    compression: Option<CsvCompression>,
) -> PolarsResult<()> {
    if let Some(method) = compression {
        feature_gated!(
            "compress",
            match method {
                CsvCompression::Gzip(level) => {
                    let level = level.unwrap_or_default();
                    let mut encoder = flate2::write::GzEncoder::new(
                        writer,
                        flate2::Compression::new(level.compression_level() as _),
                    );
                    encoder.write_all(input_buf)?;
                    encoder.try_finish().map_err(|e| e.into())
                },
                CsvCompression::Zstd(level) => {
                    let level = level.unwrap_or_default();
                    let mut encoder = zstd::Encoder::new(writer, level.compression_level())?;
                    encoder.write_all(input_buf)?;
                    encoder.do_finish().map_err(|e| e.into())
                },
            }
        )
    } else {
        writer.write_all(input_buf)?;
        Ok(())
    }
}
