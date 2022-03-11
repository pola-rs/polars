use super::ArrowResult;
use arrow::datatypes::PhysicalType;
use arrow::error::ArrowError;
use arrow::io::parquet::write::{self, FileWriter, *};
use arrow::io::parquet::write::{array_to_pages, DynIter, DynStreamingIterator, Encoding};
use polars_core::prelude::*;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::io::Write;

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
    compression: write::Compression,
    statistics: bool,
}

pub use write::Compression as ParquetCompression;

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
            compression: write::Compression::Snappy,
            statistics: false,
        }
    }

    /// Set the compression used. Defaults to `Snappy`.
    pub fn with_compression(mut self, compression: write::Compression) -> Self {
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
        let encodings = schema
            .fields
            .iter()
            .map(|field| match field.data_type().to_physical_type() {
                // delta encoding
                // Not yet supported by pyarrow
                // PhysicalType::LargeUtf8 => Encoding::DeltaLengthByteArray,
                // dictionaries are kept dict-encoded
                PhysicalType::Dictionary(_) => Encoding::RleDictionary,
                // remaining is plain
                _ => Encoding::Plain,
            })
            .collect::<Vec<_>>();

        let row_group_iter = rb_iter.map(|batch| {
            let columns = batch
                .columns()
                .par_iter()
                .zip(parquet_schema.columns().par_iter())
                .zip(encodings.par_iter())
                .map(|((array, descriptor), encoding)| {
                    let encoded_pages =
                        array_to_pages(array.as_ref(), descriptor.clone(), options, *encoding)?;
                    encoded_pages
                        .map(|page| {
                            compress(page?, vec![], options.compression).map_err(|x| x.into())
                        })
                        .collect::<ArrowResult<VecDeque<_>>>()
                })
                .collect::<ArrowResult<Vec<VecDeque<CompressedPage>>>>()?;

            let row_group = DynIter::new(
                columns
                    .into_iter()
                    .map(|column| Ok(DynStreamingIterator::new(Bla::new(column)))),
            );
            ArrowResult::Ok((row_group, batch.columns()[0].len()))
        });

        let mut writer = FileWriter::try_new(&mut self.writer, schema, options)?;
        // write the headers
        writer.start()?;
        for group in row_group_iter {
            let (group, len) = group?;
            writer.write(group, len)?;
        }
        let _ = writer.end(None)?;

        Ok(())
    }
}
