use std::collections::VecDeque;
use std::io::Write;
use std::sync::Mutex;

use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use polars_core::prelude::*;
use polars_core::POOL;
use polars_parquet::read::ParquetError;
use polars_parquet::write::{
    array_to_columns, compress, CompressedPage, Compressor, DynIter, DynStreamingIterator,
    Encoding, FallibleStreamingIterator, FileWriter, ParquetType, RowGroupIter, SchemaDescriptor,
    WriteOptions,
};
use rayon::prelude::*;

pub struct BatchedWriter<W: Write> {
    // A mutex so that streaming engine can get concurrent read access to
    // compress pages.
    pub(super) writer: Mutex<FileWriter<W>>,
    pub(super) parquet_schema: SchemaDescriptor,
    pub(super) encodings: Vec<Vec<Encoding>>,
    pub(super) options: WriteOptions,
    pub(super) parallel: bool,
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
