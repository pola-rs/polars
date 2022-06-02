use super::ArrowResult;
use arrow::array::Array;
use arrow::chunk::Chunk;
use arrow::datatypes::DataType as ArrowDataType;
use arrow::datatypes::PhysicalType;
use arrow::error::Error as ArrowError;
use arrow::io::parquet::read::ParquetError;
use arrow::io::parquet::write::{self, FileWriter, *};
use arrow::io::parquet::write::{DynIter, DynStreamingIterator, Encoding};
use polars_core::prelude::*;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::io::Write;

pub use write::{BrotliLevel, CompressionOptions as ParquetCompression, GzipLevel, ZstdLevel};

struct Bla {
    columns: VecDeque<CompressedPage>,
    current: Option<CompressedPage>,
}

impl Bla {
    pub fn new(columns: VecDeque<CompressedPage>) -> Self {
        Self {
            columns,
            current: None,
        }
    }
}

impl FallibleStreamingIterator for Bla {
    type Item = CompressedPage;
    type Error = ArrowError;

    fn advance(&mut self) -> ArrowResult<()> {
        self.current = self.columns.pop_front();
        Ok(())
    }

    fn get(&self) -> Option<&Self::Item> {
        self.current.as_ref()
    }
}

/// Write a DataFrame to parquet format
///
/// # Example
///
///
#[must_use]
pub struct ParquetWriter<W> {
    writer: W,
    compression: write::CompressionOptions,
    statistics: bool,
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
        }
    }

    /// Set the compression used. Defaults to `Lz4Raw`.
    pub fn with_compression(mut self, compression: write::CompressionOptions) -> Self {
        self.compression = compression;
        self
    }

    pub fn with_statistics(mut self, statistics: bool) -> Self {
        self.statistics = statistics;
        self
    }

    /// Write the given DataFrame in the the writer `W`.
    pub fn finish(mut self, df: &mut DataFrame) -> Result<()> {
        df.rechunk();
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

        let encodings = (&schema.fields)
            .iter()
            .map(|f| transverse(&f.data_type, encoding_map))
            .collect::<Vec<_>>();

        let row_group_iter = rb_iter.filter_map(|batch| match batch.len() {
            0 => None,
            _ => {
                let row_group = create_serializer(
                    batch,
                    parquet_schema.fields().to_vec(),
                    encodings.clone(),
                    options,
                );

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
    batch: Chunk<Arc<dyn Array>>,
    fields: Vec<ParquetType>,
    encodings: Vec<Vec<Encoding>>,
    options: WriteOptions,
) -> std::result::Result<RowGroupIter<'static, ArrowError>, ArrowError> {
    let columns = batch
        .columns()
        .par_iter()
        .zip(fields)
        .zip(encodings)
        .flat_map(move |((array, type_), encoding)| {
            let encoded_columns = array_to_columns(array, type_, options, encoding).unwrap();
            encoded_columns
                .into_iter()
                .map(|encoded_pages| {
                    let encoded_pages = DynIter::new(
                        encoded_pages
                            .into_iter()
                            .map(|x| x.map_err(|e| ParquetError::General(e.to_string()))),
                    );
                    encoded_pages
                        .map(|page| {
                            compress(page?, vec![], options.compression).map_err(|x| x.into())
                        })
                        .collect::<ArrowResult<VecDeque<_>>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<ArrowResult<Vec<VecDeque<CompressedPage>>>>()?;

    let row_group = DynIter::new(
        columns
            .into_iter()
            .map(|column| Ok(DynStreamingIterator::new(Bla::new(column)))),
    );
    Ok(row_group)
}
