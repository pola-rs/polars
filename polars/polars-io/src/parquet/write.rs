use std::io::Write;

use arrow::array::Array;
use arrow::chunk::Chunk;
use arrow::datatypes::{DataType as ArrowDataType, PhysicalType};
use arrow::error::Error as ArrowError;
use arrow::io::parquet::read::ParquetError;
use arrow::io::parquet::write::{self, DynIter, DynStreamingIterator, Encoding, FileWriter, *};
use polars_core::prelude::*;
use polars_core::utils::{accumulate_dataframes_vertical_unchecked, split_df};
use polars_core::POOL;
use rayon::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use write::{
    BrotliLevel as BrotliLevelParquet, CompressionOptions, GzipLevel as GzipLevelParquet,
    ZstdLevel as ZstdLevelParquet,
};

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GzipLevel(u8);

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BrotliLevel(u32);

/// Represents a valid zstd compression level.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ZstdLevel(i32);

impl ZstdLevel {
    pub fn try_new(level: i32) -> PolarsResult<Self> {
        ZstdLevelParquet::try_new(level).map_err(ArrowError::from)?;
        Ok(ZstdLevel(level))
    }
}

impl BrotliLevel {
    pub fn try_new(level: u32) -> PolarsResult<Self> {
        BrotliLevelParquet::try_new(level).map_err(ArrowError::from)?;
        Ok(BrotliLevel(level))
    }
}

impl GzipLevel {
    pub fn try_new(level: u8) -> PolarsResult<Self> {
        GzipLevelParquet::try_new(level).map_err(ArrowError::from)?;
        Ok(GzipLevel(level))
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ParquetCompression {
    Uncompressed,
    Snappy,
    Gzip(Option<GzipLevel>),
    Lzo,
    Brotli(Option<BrotliLevel>),
    Zstd(Option<ZstdLevel>),
    #[default]
    Lz4Raw,
}

impl From<ParquetCompression> for CompressionOptions {
    fn from(value: ParquetCompression) -> Self {
        use ParquetCompression::*;
        match value {
            Uncompressed => CompressionOptions::Uncompressed,
            Snappy => CompressionOptions::Snappy,
            Gzip(level) => {
                CompressionOptions::Gzip(level.map(|v| GzipLevelParquet::try_new(v.0).unwrap()))
            }
            Lzo => CompressionOptions::Lzo,
            Brotli(level) => {
                CompressionOptions::Brotli(level.map(|v| BrotliLevelParquet::try_new(v.0).unwrap()))
            }
            Lz4Raw => CompressionOptions::Lz4Raw,
            Zstd(level) => {
                CompressionOptions::Zstd(level.map(|v| ZstdLevelParquet::try_new(v.0).unwrap()))
            }
        }
    }
}

/// Write a DataFrame to parquet format
///
#[must_use]
pub struct ParquetWriter<W> {
    writer: W,
    /// Data page compression
    compression: CompressionOptions,
    /// Compute and write column statistics.
    statistics: bool,
    /// If `None` will be all written to a single row group.
    row_group_size: Option<usize>,
    /// if `None` will be 1024^2 bytes
    data_pagesize_limit: Option<usize>,
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
            compression: CompressionOptions::Zstd(None),
            statistics: false,
            row_group_size: None,
            data_pagesize_limit: None,
        }
    }

    /// Set the compression used. Defaults to `Lz4Raw`.
    ///
    /// The default compression `Lz4Raw` has very good performance, but may not yet been supported
    /// by older readers. If you want more compatability guarantees, consider using `Snappy`.
    pub fn with_compression(mut self, compression: ParquetCompression) -> Self {
        self.compression = compression.into();
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

    /// Sets the maximum bytes size of a data page. If `None` will be 1024^2 bytes.
    pub fn with_data_pagesize_limit(mut self, limit: Option<usize>) -> Self {
        self.data_pagesize_limit = limit;
        self
    }

    fn materialize_options(&self) -> WriteOptions {
        WriteOptions {
            write_statistics: self.statistics,
            compression: self.compression,
            version: Version::V2,
            data_pagesize_limit: self.data_pagesize_limit,
        }
    }

    pub fn batched(self, schema: &Schema) -> PolarsResult<BatchedWriter<W>> {
        let fields = schema.to_arrow().fields;
        let schema = ArrowSchema::from(fields);

        let parquet_schema = to_parquet_schema(&schema)?;
        let encodings = get_encodings(&schema);
        let options = self.materialize_options();
        let writer = FileWriter::try_new(self.writer, schema, options)?;

        Ok(BatchedWriter {
            writer,
            parquet_schema,
            encodings,
            options,
        })
    }

    /// Write the given DataFrame in the the writer `W`. Returns the total size of the file.
    pub fn finish(self, df: &mut DataFrame) -> PolarsResult<u64> {
        // ensures all chunks are aligned.
        df.rechunk();

        if let Some(n) = self.row_group_size {
            let n_splits = df.height() / n;
            if n_splits > 0 {
                *df = accumulate_dataframes_vertical_unchecked(split_df(df, n_splits)?);
            }
        };
        let mut batched = self.batched(&df.schema())?;
        batched.write_batch(df)?;
        batched.finish()
    }
}

// Note that the df should be rechunked
fn prepare_rg_iter<'a>(
    df: &'a DataFrame,
    parquet_schema: &'a SchemaDescriptor,
    encodings: &'a [Vec<Encoding>],
    options: WriteOptions,
) -> impl Iterator<Item = Result<RowGroupIter<'a, ArrowError>, ArrowError>> + 'a {
    let rb_iter = df.iter_chunks();
    rb_iter.filter_map(move |batch| match batch.len() {
        0 => None,
        _ => {
            let row_group = create_serializer(batch, parquet_schema.fields(), encodings, options);

            Some(row_group)
        }
    })
}

fn get_encodings(schema: &ArrowSchema) -> Vec<Vec<Encoding>> {
    schema
        .fields
        .iter()
        .map(|f| transverse(&f.data_type, encoding_map))
        .collect()
}

/// Declare encodings
fn encoding_map(data_type: &ArrowDataType) -> Encoding {
    match data_type.to_physical_type() {
        PhysicalType::Dictionary(_) => Encoding::RleDictionary,
        // remaining is plain
        _ => Encoding::Plain,
    }
}

pub struct BatchedWriter<W: Write> {
    writer: FileWriter<W>,
    parquet_schema: SchemaDescriptor,
    encodings: Vec<Vec<Encoding>>,
    options: WriteOptions,
}

impl<W: Write> BatchedWriter<W> {
    /// Write a batch to the parquet writer.
    ///
    /// # Panics
    /// The caller must ensure the chunks in the given [`DataFrame`] are aligned.
    pub fn write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        let row_group_iter =
            prepare_rg_iter(df, &self.parquet_schema, &self.encodings, self.options);
        for group in row_group_iter {
            self.writer.write(group?)?;
        }
        Ok(())
    }

    /// Writes the footer of the parquet file. Returns the total size of the file.
    pub fn finish(&mut self) -> PolarsResult<u64> {
        let size = self.writer.end(None)?;
        Ok(size)
    }
}

fn create_serializer<'a>(
    batch: Chunk<Box<dyn Array>>,
    fields: &[ParquetType],
    encodings: &[Vec<Encoding>],
    options: WriteOptions,
) -> Result<RowGroupIter<'a, ArrowError>, ArrowError> {
    let columns = POOL.install(|| {
        batch
            .columns()
            .par_iter()
            .zip(fields)
            .zip(encodings)
            .flat_map(move |((array, type_), encoding)| {
                let encoded_columns =
                    array_to_columns(array, type_.clone(), options, encoding).unwrap();

                encoded_columns
                    .into_iter()
                    .map(|encoded_pages| {
                        // iterator over pages
                        let pages = DynStreamingIterator::new(
                            Compressor::new_from_vec(
                                encoded_pages.map(|result| {
                                    result.map_err(|e| {
                                        ParquetError::FeatureNotSupported(format!(
                                            "reraised in polars: {e}",
                                        ))
                                    })
                                }),
                                options.compression,
                                vec![],
                            )
                            .map_err(|e| ArrowError::External(format!("{e}"), Box::new(e))),
                        );

                        Ok(pages)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });

    let row_group = DynIter::new(columns.into_iter());

    Ok(row_group)
}
