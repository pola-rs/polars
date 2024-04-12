use std::collections::VecDeque;
use std::io::Write;
use std::sync::Mutex;

use arrow::array::Array;
use arrow::datatypes::PhysicalType;
use arrow::record_batch::RecordBatch;
use polars_core::prelude::*;
use polars_core::POOL;
use polars_parquet::read::ParquetError;
pub use polars_parquet::write::RowGroupIter;
use polars_parquet::write::{self, *};
use rayon::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use write::{
    BrotliLevel as BrotliLevelParquet, GzipLevel as GzipLevelParquet, ZstdLevel as ZstdLevelParquet,
};

use crate::prelude::chunk_df_for_writing;

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
        ZstdLevelParquet::try_new(level)?;
        Ok(ZstdLevel(level))
    }
}

impl BrotliLevel {
    pub fn try_new(level: u32) -> PolarsResult<Self> {
        BrotliLevelParquet::try_new(level)?;
        Ok(BrotliLevel(level))
    }
}

impl GzipLevel {
    pub fn try_new(level: u8) -> PolarsResult<Self> {
        GzipLevelParquet::try_new(level)?;
        Ok(GzipLevel(level))
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ParquetCompression {
    Uncompressed,
    Snappy,
    Gzip(Option<GzipLevel>),
    Lzo,
    Brotli(Option<BrotliLevel>),
    Zstd(Option<ZstdLevel>),
    Lz4Raw,
}

impl Default for ParquetCompression {
    fn default() -> Self {
        Self::Zstd(None)
    }
}

impl From<ParquetCompression> for CompressionOptions {
    fn from(value: ParquetCompression) -> Self {
        use ParquetCompression::*;
        match value {
            Uncompressed => CompressionOptions::Uncompressed,
            Snappy => CompressionOptions::Snappy,
            Gzip(level) => {
                CompressionOptions::Gzip(level.map(|v| GzipLevelParquet::try_new(v.0).unwrap()))
            },
            Lzo => CompressionOptions::Lzo,
            Brotli(level) => {
                CompressionOptions::Brotli(level.map(|v| BrotliLevelParquet::try_new(v.0).unwrap()))
            },
            Lz4Raw => CompressionOptions::Lz4Raw,
            Zstd(level) => {
                CompressionOptions::Zstd(level.map(|v| ZstdLevelParquet::try_new(v.0).unwrap()))
            },
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
    /// if `None` will be 512^2 rows
    row_group_size: Option<usize>,
    /// if `None` will be 1024^2 bytes
    data_page_size: Option<usize>,
    /// Serialize columns in parallel
    parallel: bool,
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
            compression: ParquetCompression::default().into(),
            statistics: true,
            row_group_size: None,
            data_page_size: None,
            parallel: true,
        }
    }

    /// Set the compression used. Defaults to `Zstd`.
    ///
    /// The default compression `Zstd` has very good performance, but may not yet been supported
    /// by older readers. If you want more compatibility guarantees, consider using `Snappy`.
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
    pub fn with_data_page_size(mut self, limit: Option<usize>) -> Self {
        self.data_page_size = limit;
        self
    }

    /// Serialize columns in parallel
    pub fn set_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    fn materialize_options(&self) -> WriteOptions {
        WriteOptions {
            write_statistics: self.statistics,
            compression: self.compression,
            version: Version::V2,
            data_pagesize_limit: self.data_page_size,
        }
    }

    pub fn batched(self, schema: &Schema) -> PolarsResult<BatchedWriter<W>> {
        let fields = schema.to_arrow(true).fields;
        let schema = ArrowSchema::from(fields);

        let parquet_schema = to_parquet_schema(&schema)?;
        let encodings = get_encodings(&schema);
        let options = self.materialize_options();
        let writer = Mutex::new(FileWriter::try_new(self.writer, schema, options)?);

        Ok(BatchedWriter {
            writer,
            parquet_schema,
            encodings,
            options,
            parallel: self.parallel,
        })
    }

    /// Write the given DataFrame in the writer `W`. Returns the total size of the file.
    pub fn finish(self, df: &mut DataFrame) -> PolarsResult<u64> {
        let chunked_df = chunk_df_for_writing(df, self.row_group_size.unwrap_or(512 * 512))?;
        let mut batched = self.batched(&chunked_df.schema())?;
        batched.write_batch(&chunked_df)?;
        batched.finish()
    }
}

// Note that the df should be rechunked
fn prepare_rg_iter<'a>(
    df: &'a DataFrame,
    parquet_schema: &'a SchemaDescriptor,
    encodings: &'a [Vec<Encoding>],
    options: WriteOptions,
    parallel: bool,
) -> impl Iterator<Item = PolarsResult<RowGroupIter<'static, PolarsError>>> + 'a {
    let rb_iter = df.iter_chunks(true);
    rb_iter.filter_map(move |batch| match batch.len() {
        0 => None,
        _ => {
            let row_group =
                create_serializer(batch, parquet_schema.fields(), encodings, options, parallel);

            Some(row_group)
        },
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
        PhysicalType::Dictionary(_)
        | PhysicalType::LargeBinary
        | PhysicalType::LargeUtf8
        | PhysicalType::Utf8View
        | PhysicalType::BinaryView => Encoding::RleDictionary,
        PhysicalType::Primitive(dt) => {
            use arrow::types::PrimitiveType::*;
            match dt {
                Float32 | Float64 | Float16 => Encoding::Plain,
                _ => Encoding::RleDictionary,
            }
        },
        // remaining is plain
        _ => Encoding::Plain,
    }
}

pub struct BatchedWriter<W: Write> {
    // A mutex so that streaming engine can get concurrent read access to
    // compress pages.
    writer: Mutex<FileWriter<W>>,
    parquet_schema: SchemaDescriptor,
    encodings: Vec<Vec<Encoding>>,
    options: WriteOptions,
    parallel: bool,
}

impl<W: Write> BatchedWriter<W> {
    pub fn encode_and_compress<'a>(
        &'a self,
        df: &'a DataFrame,
    ) -> impl Iterator<Item = PolarsResult<RowGroupIter<'static, PolarsError>>> + 'a {
        let rb_iter = df.iter_chunks(true);
        rb_iter.filter_map(move |batch| match batch.len() {
            0 => None,
            _ => {
                let row_group = create_eager_serializer(
                    batch,
                    self.parquet_schema.fields(),
                    self.encodings.as_ref(),
                    self.options,
                );

                Some(row_group)
            },
        })
    }

    /// Write a batch to the parquet writer.
    ///
    /// # Panics
    /// The caller must ensure the chunks in the given [`DataFrame`] are aligned.
    pub fn write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        let row_group_iter = prepare_rg_iter(
            df,
            &self.parquet_schema,
            &self.encodings,
            self.options,
            self.parallel,
        );
        // Lock before looping so that order is maintained under contention.
        let mut writer = self.writer.lock().unwrap();
        for group in row_group_iter {
            writer.write(group?)?;
        }
        Ok(())
    }

    pub fn get_writer(&self) -> &Mutex<FileWriter<W>> {
        &self.writer
    }

    pub fn write_row_groups(
        &self,
        rgs: Vec<RowGroupIter<'static, PolarsError>>,
    ) -> PolarsResult<()> {
        // Lock before looping so that order is maintained.
        let mut writer = self.writer.lock().unwrap();
        for group in rgs {
            writer.write(group)?;
        }
        Ok(())
    }

    /// Writes the footer of the parquet file. Returns the total size of the file.
    pub fn finish(&self) -> PolarsResult<u64> {
        let mut writer = self.writer.lock().unwrap();
        let size = writer.end(None)?;
        Ok(size)
    }
}

fn create_serializer(
    batch: RecordBatch<Box<dyn Array>>,
    fields: &[ParquetType],
    encodings: &[Vec<Encoding>],
    options: WriteOptions,
    parallel: bool,
) -> PolarsResult<RowGroupIter<'static, PolarsError>> {
    let func = move |((array, type_), encoding): ((&ArrayRef, &ParquetType), &Vec<Encoding>)| {
        let encoded_columns = array_to_columns(array, type_.clone(), options, encoding).unwrap();

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
                    .map_err(PolarsError::from),
                );

                Ok(pages)
            })
            .collect::<Vec<_>>()
    };

    let columns = if parallel {
        POOL.install(|| {
            batch
                .columns()
                .par_iter()
                .zip(fields)
                .zip(encodings)
                .flat_map(func)
                .collect::<Vec<_>>()
        })
    } else {
        batch
            .columns()
            .iter()
            .zip(fields)
            .zip(encodings)
            .flat_map(func)
            .collect::<Vec<_>>()
    };

    let row_group = DynIter::new(columns.into_iter());

    Ok(row_group)
}

struct CompressedPages {
    pages: VecDeque<PolarsResult<CompressedPage>>,
    current: Option<CompressedPage>,
}

impl CompressedPages {
    fn new(pages: VecDeque<PolarsResult<CompressedPage>>) -> Self {
        Self {
            pages,
            current: None,
        }
    }
}

impl FallibleStreamingIterator for CompressedPages {
    type Item = CompressedPage;
    type Error = PolarsError;

    fn advance(&mut self) -> Result<(), Self::Error> {
        self.current = self.pages.pop_front().transpose()?;
        Ok(())
    }

    fn get(&self) -> Option<&Self::Item> {
        self.current.as_ref()
    }
}

/// This serializer encodes and compresses all eagerly in memory.
/// Used for separating compute from IO.
fn create_eager_serializer(
    batch: RecordBatch<Box<dyn Array>>,
    fields: &[ParquetType],
    encodings: &[Vec<Encoding>],
    options: WriteOptions,
) -> PolarsResult<RowGroupIter<'static, PolarsError>> {
    let func = move |((array, type_), encoding): ((&ArrayRef, &ParquetType), &Vec<Encoding>)| {
        let encoded_columns = array_to_columns(array, type_.clone(), options, encoding).unwrap();

        encoded_columns
            .into_iter()
            .map(|encoded_pages| {
                let compressed_pages = encoded_pages
                    .into_iter()
                    .map(|page| {
                        let page = page?;
                        let page = compress(page, vec![], options.compression)?;
                        Ok(Ok(page))
                    })
                    .collect::<PolarsResult<VecDeque<_>>>()?;

                Ok(DynStreamingIterator::new(CompressedPages::new(
                    compressed_pages,
                )))
            })
            .collect::<Vec<_>>()
    };

    let columns = batch
        .columns()
        .iter()
        .zip(fields)
        .zip(encodings)
        .flat_map(func)
        .collect::<Vec<_>>();

    let row_group = DynIter::new(columns.into_iter());

    Ok(row_group)
}
