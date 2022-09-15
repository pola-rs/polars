use std::io::Write;

use arrow::array::Array;
use arrow::chunk::Chunk;
use arrow::datatypes::{DataType as ArrowDataType, PhysicalType};
use arrow::error::Error as ArrowError;
use arrow::io::parquet::read::ParquetError;
use arrow::io::parquet::write::{self, DynIter, DynStreamingIterator, Encoding, FileWriter, *};
use polars_core::prelude::*;
use polars_core::utils::{accumulate_dataframes_vertical_unchecked, split_df};
use rayon::prelude::*;
pub use write::{BrotliLevel, CompressionOptions as ParquetCompression, GzipLevel, ZstdLevel};

/// Write a DataFrame to parquet format
///
#[must_use]
pub struct ParquetWriter<W> {
    writer: W,
    compression: write::CompressionOptions,
    statistics: bool,
    row_group_size: Option<usize>,
}

impl<W> ParquetWriter<W>
where
    W: Write,
{
    /// Create a new writer
    pub fn new(writer: W) -> Self
    where
        W: Write,
    {
        ParquetWriter {
            writer,
            compression: write::CompressionOptions::Lz4Raw,
            statistics: false,
            row_group_size: None,
        }
    }

    /// Set the compression used. Defaults to `Lz4Raw`.
    ///
    /// The default compression `Lz4Raw` has very good performance, but may not yet been supported
    /// by older readers. If you want more compatability guarantees, consider using `Snappy`.
    pub fn with_compression(mut self, compression: write::CompressionOptions) -> Self {
        self.compression = compression;
        self
    }

    /// Compute and write statistic
    pub fn with_statistics(mut self, statistics: bool) -> Self {
        self.statistics = statistics;
        self
    }

    /// Set the row group size (in number of rows) during writing. This can reduce memory pressure and improve
    /// writing performance.
    pub fn with_row_group_size(mut self, size: Option<usize>) -> Self {
        self.row_group_size = size;
        self
    }

    /// Write the given DataFrame in the the writer `W`.
    pub fn finish(mut self, df: &mut DataFrame) -> PolarsResult<()> {
        // ensures all chunks are aligned.
        df.rechunk();

        if let Some(n) = self.row_group_size {
            let n_splits = df.height() / n;
            if n_splits > 0 {
                *df = accumulate_dataframes_vertical_unchecked(split_df(df, n_splits)?);
            }
        };

        let fields = df.schema().to_arrow().fields;
        let rb_iter = df.iter_chunks();

        let options = write::WriteOptions {
            write_statistics: self.statistics,
            compression: self.compression,
            version: write::Version::V2,
        };
        let schema = ArrowSchema::from(fields);
        let parquet_schema = write::to_parquet_schema(&schema)?;
        // declare encodings
        let encoding_map = |data_type: &ArrowDataType| {
            match data_type.to_physical_type() {
                PhysicalType::Dictionary(_) => Encoding::RleDictionary,
                // remaining is plain
                _ => Encoding::Plain,
            }
        };

        let encodings = schema
            .fields
            .iter()
            .map(|f| transverse(&f.data_type, encoding_map))
            .collect::<Vec<_>>();

        let row_group_iter = rb_iter.filter_map(|batch| match batch.len() {
            0 => None,
            _ => {
                let row_group =
                    create_serializer(batch, parquet_schema.fields().to_vec(), &encodings, options);

                Some(row_group)
            }
        });

        let mut writer = FileWriter::try_new(&mut self.writer, schema, options)?;
        for group in row_group_iter {
            writer.write(group?)?;
        }
        let _ = writer.end(None)?;

        Ok(())
    }
}

fn create_serializer(
    batch: Chunk<Box<dyn Array>>,
    fields: Vec<ParquetType>,
    encodings: &[Vec<Encoding>],
    options: WriteOptions,
) -> std::result::Result<RowGroupIter<'static, ArrowError>, ArrowError> {
    let columns = batch
        .columns()
        .par_iter()
        .zip(fields)
        .zip(encodings)
        .map(move |((array, type_), encoding)| {
            let encoded_columns = array_to_columns(array, type_, options, encoding).unwrap();

            encoded_columns
                .into_iter()
                .map(|encoded_pages| {
                    // iterator over pages
                    let pages = DynStreamingIterator::new(
                        Compressor::new_from_vec(
                            encoded_pages.map(|result| {
                                result.map_err(|e| {
                                    ParquetError::FeatureNotSupported(format!(
                                        "reraised in polars: {}",
                                        e
                                    ))
                                })
                            }),
                            options.compression,
                            vec![],
                        )
                        .map_err(|e| ArrowError::External(format!("{}", e), Box::new(e))),
                    );

                    Ok(pages)
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();

    let row_group = DynIter::new(columns.into_iter());

    Ok(row_group)
}
