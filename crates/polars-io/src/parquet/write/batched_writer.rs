use std::io::Write;
use std::sync::Mutex;

use arrow::record_batch::RecordBatch;
use polars_buffer::Buffer;
use polars_core::POOL;
use polars_core::prelude::*;
use polars_parquet::read::{ParquetError, fallible_streaming_iterator};
use polars_parquet::write::{
    CompressedPage, Compressor, DynIter, DynStreamingIterator, Encoding, FallibleStreamingIterator,
    FileWriter, Page, ParquetType, RowGroupIterColumns, SchemaDescriptor, WriteOptions,
    array_to_columns, schema_to_metadata_key,
};
use rayon::prelude::*;

use super::{KeyValueMetadata, ParquetMetadataContext};

pub struct BatchedWriter<W: Write> {
    // A mutex so that streaming engine can get concurrent read access to
    // compress pages.
    //
    // @TODO: Remove mutex when old streaming engine is removed
    pub(super) writer: Mutex<FileWriter<W>>,
    // @TODO: Remove when old streaming engine is removed
    pub(super) parquet_schema: SchemaDescriptor,
    pub(super) encodings: Buffer<Vec<Encoding>>,
    pub(super) options: WriteOptions,
    pub(super) parallel: bool,
    pub(super) key_value_metadata: Option<KeyValueMetadata>,
}

impl<W: Write> BatchedWriter<W> {
    pub fn new(
        writer: Mutex<FileWriter<W>>,
        encodings: Buffer<Vec<Encoding>>,
        options: WriteOptions,
        parallel: bool,
        key_value_metadata: Option<KeyValueMetadata>,
    ) -> Self {
        Self {
            writer,
            parquet_schema: SchemaDescriptor::new(PlSmallStr::EMPTY, vec![]),
            encodings,
            options,
            parallel,
            key_value_metadata,
        }
    }

    pub fn encode_and_compress<'a>(
        &'a self,
        df: &'a DataFrame,
    ) -> impl Iterator<Item = PolarsResult<RowGroupIterColumns<'static, PolarsError>>> + 'a {
        let rb_iter = df.iter_chunks(CompatLevel::newest(), false);
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
        for (num_rows, group) in row_group_iter {
            writer.write(num_rows as u64, group?)?;
        }
        Ok(())
    }

    pub fn parquet_schema(&mut self) -> &SchemaDescriptor {
        let writer = self.writer.get_mut().unwrap();
        writer.parquet_schema()
    }

    /// Note: `num_rows` can be passed as `u64::MAX` to infer `num_rows` from the encoded data.
    pub fn write_row_group(
        &mut self,
        num_rows: u64,
        rg: &[Vec<CompressedPage>],
    ) -> PolarsResult<()> {
        let writer = self.writer.get_mut().unwrap();
        let rg = DynIter::new(rg.iter().map(|col_pages| {
            Ok(DynStreamingIterator::new(
                fallible_streaming_iterator::convert(col_pages.iter().map(PolarsResult::Ok)),
            ))
        }));
        writer.write(num_rows, rg)?;
        Ok(())
    }

    pub fn get_writer(&self) -> &Mutex<FileWriter<W>> {
        &self.writer
    }

    pub fn write_row_groups(
        &self,
        rgs: Vec<RowGroupIterColumns<'static, PolarsError>>,
    ) -> PolarsResult<()> {
        // Lock before looping so that order is maintained.
        let mut writer = self.writer.lock().unwrap();
        for group in rgs {
            writer.write(u64::MAX, group)?;
        }
        Ok(())
    }

    /// Writes the footer of the parquet file. Returns the total size of the file.
    pub fn finish(&self) -> PolarsResult<u64> {
        let mut writer = self.writer.lock().unwrap();

        let key_value_metadata = self
            .key_value_metadata
            .as_ref()
            .map(|meta| {
                let arrow_schema = schema_to_metadata_key(writer.schema());
                let ctx = ParquetMetadataContext {
                    arrow_schema: arrow_schema.value.as_ref().unwrap(),
                };
                let mut out = meta.collect(ctx)?;
                if !out.iter().any(|kv| kv.key == arrow_schema.key) {
                    out.insert(0, arrow_schema);
                }
                PolarsResult::Ok(out)
            })
            .transpose()?;

        let size = writer.end(key_value_metadata)?;
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
) -> impl Iterator<
    Item = (
        usize,
        PolarsResult<RowGroupIterColumns<'static, PolarsError>>,
    ),
> + 'a {
    let rb_iter = df.iter_chunks(CompatLevel::newest(), false);
    rb_iter.filter_map(move |batch| match batch.len() {
        0 => None,
        num_rows => {
            let row_group =
                create_serializer(batch, parquet_schema.fields(), encodings, options, parallel);

            Some((num_rows, row_group))
        },
    })
}

fn pages_iter_to_compressor(
    encoded_columns: Vec<DynIter<'static, PolarsResult<Page>>>,
    options: WriteOptions,
) -> Vec<PolarsResult<DynStreamingIterator<'static, CompressedPage, PolarsError>>> {
    encoded_columns
        .into_iter()
        .map(|encoded_pages| {
            // iterator over pages
            let pages = DynStreamingIterator::new(
                Compressor::new_from_vec(
                    encoded_pages.map(|result| {
                        result.map_err(|e| {
                            ParquetError::FeatureNotSupported(format!("reraised in polars: {e}",))
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
}

fn array_to_pages_iter(
    array: &ArrayRef,
    type_: &ParquetType,
    encoding: &[Encoding],
    options: WriteOptions,
) -> Vec<PolarsResult<DynStreamingIterator<'static, CompressedPage, PolarsError>>> {
    let encoded_columns = array_to_columns(array, type_.clone(), options, encoding).unwrap();
    pages_iter_to_compressor(encoded_columns, options)
}

fn create_serializer(
    batch: RecordBatch,
    fields: &[ParquetType],
    encodings: &[Vec<Encoding>],
    options: WriteOptions,
    parallel: bool,
) -> PolarsResult<RowGroupIterColumns<'static, PolarsError>> {
    let func = move |((array, type_), encoding): ((&ArrayRef, &ParquetType), &Vec<Encoding>)| {
        array_to_pages_iter(array, type_, encoding, options)
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

/// This serializer encodes and compresses all eagerly in memory.
/// Used for separating compute from IO.
fn create_eager_serializer(
    batch: RecordBatch,
    fields: &[ParquetType],
    encodings: &[Vec<Encoding>],
    options: WriteOptions,
) -> PolarsResult<RowGroupIterColumns<'static, PolarsError>> {
    let func = move |((array, type_), encoding): ((&ArrayRef, &ParquetType), &Vec<Encoding>)| {
        array_to_pages_iter(array, type_, encoding, options)
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
