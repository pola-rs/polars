mod serializer;

use arrow::array::NullArray;
use arrow::legacy::time_zone::Tz;
use polars_core::POOL;
use polars_core::prelude::*;
use polars_error::polars_ensure;
use polars_utils::reuse_vec::reuse_vec;
use rayon::prelude::*;
use serializer::{serializer_for, string_serializer};

use crate::csv::write::SerializeOptions;

type ColumnSerializer<'a> =
    dyn crate::csv::write::write_impl::serializer::Serializer<'a> + Send + 'a;

/// Writes CSV from DataFrames.
pub struct CsvSerializer {
    serializers: Vec<Box<ColumnSerializer<'static>>>,
    options: Arc<SerializeOptions>,
    datetime_formats: Arc<[PlSmallStr]>,
    time_zones: Arc<[Option<Tz>]>,
}

impl Clone for CsvSerializer {
    fn clone(&self) -> Self {
        Self {
            serializers: vec![],
            options: self.options.clone(),
            datetime_formats: self.datetime_formats.clone(),
            time_zones: self.time_zones.clone(),
        }
    }
}

impl CsvSerializer {
    pub fn new(schema: SchemaRef, options: Arc<SerializeOptions>) -> PolarsResult<Self> {
        for dtype in schema.iter_values() {
            let nested = match dtype {
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

        let (datetime_formats, time_zones): (Vec<PlSmallStr>, Vec<Option<Tz>>) = schema
            .iter_values()
            .map(|dtype| {
                let (datetime_format_str, time_zone) = match dtype {
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
                };

                (datetime_format_str.into(), time_zone)
            })
            .collect();

        Ok(Self {
            serializers: vec![],
            options,
            datetime_formats: Arc::from_iter(datetime_formats),
            time_zones: Arc::from_iter(time_zones),
        })
    }

    /// # Panics
    /// Panics if a column has >1 chunk.
    pub fn serialize_to_csv<'a>(
        &'a mut self,
        df: &'a DataFrame,
        buffer: &mut Vec<u8>,
    ) -> PolarsResult<()> {
        if df.height() == 0 || df.width() == 0 {
            return Ok(());
        }

        let options = Arc::clone(&self.options);
        let options = options.as_ref();

        let mut serializers_vec = reuse_vec(std::mem::take(&mut self.serializers));
        let serializers = self.build_serializers(df.columns(), &mut serializers_vec)?;

        for _ in 0..df.height() {
            serializers[0].serialize(buffer, options);
            for serializer in &mut serializers[1..] {
                buffer.push(options.separator);
                serializer.serialize(buffer, options);
            }

            buffer.extend_from_slice(options.line_terminator.as_bytes());
        }

        self.serializers = reuse_vec(serializers_vec);

        Ok(())
    }

    /// # Panics
    /// Panics if a column has >1 chunk.
    fn build_serializers<'a, 'b>(
        &'a mut self,
        columns: &'a [Column],
        serializers: &'b mut Vec<Box<ColumnSerializer<'a>>>,
    ) -> PolarsResult<&'b mut [Box<ColumnSerializer<'a>>]> {
        serializers.clear();
        serializers.reserve(columns.len());

        for (i, c) in columns.iter().enumerate() {
            assert_eq!(c.n_chunks(), 1);

            serializers.push(serializer_for(
                c.as_materialized_series().chunks()[0].as_ref(),
                Arc::as_ref(&self.options),
                c.dtype(),
                self.datetime_formats[i].as_str(),
                self.time_zones[i],
            )?)
        }

        Ok(serializers)
    }
}

pub(crate) fn write(
    mut writer: impl std::io::Write,
    df: &DataFrame,
    chunk_size: usize,
    options: Arc<SerializeOptions>,
    n_threads: usize,
) -> PolarsResult<()> {
    let len = df.height();
    let total_rows_per_pool_iter = n_threads * chunk_size;

    let mut n_rows_finished = 0;

    let csv_serializer = CsvSerializer::new(Arc::clone(df.schema()), options)?;

    let mut buffers: Vec<(Vec<u8>, CsvSerializer)> = (0..n_threads)
        .map(|_| (Vec::new(), csv_serializer.clone()))
        .collect();
    while n_rows_finished < len {
        let buf_writer =
            |thread_no, write_buffer: &mut Vec<_>, csv_serializer: &mut CsvSerializer| {
                let thread_offset = thread_no * chunk_size;
                let total_offset = n_rows_finished + thread_offset;
                let mut df = df.slice(total_offset as i64, chunk_size);
                // the `series.iter` needs rechunked series.
                // we don't do this on the whole as this probably needs much less rechunking
                // so will be faster.
                // and allows writing `pl.concat([df] * 100, rechunk=False).write_csv()` as the rechunk
                // would go OOM
                df.rechunk_mut();

                csv_serializer.serialize_to_csv(&df, write_buffer)?;

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
pub fn csv_header(names: &[&str], options: &SerializeOptions) -> PolarsResult<Vec<u8>> {
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
    Ok(header)
}

pub const UTF8_BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];
