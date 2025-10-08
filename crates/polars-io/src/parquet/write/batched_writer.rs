use std::io::Write;
use std::sync::Mutex;

use arrow::record_batch::RecordBatch;
use polars_core::POOL;
use polars_core::prelude::*;
use polars_parquet::read::{ParquetError, fallible_streaming_iterator};
use polars_parquet::write::{
    ColumnWriteOptions, CompressedPage, Compressor, DynIter, DynStreamingIterator,
    FallibleStreamingIterator, FileWriter, Page, ParquetType, RowGroupIterColumns,
    SchemaDescriptor, WriteOptions, array_to_columns, schema_to_metadata_key,
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
    pub(super) column_options: Vec<ColumnWriteOptions>,
    pub(super) options: WriteOptions,
    pub(super) parallel: bool,
    pub(super) key_value_metadata: Option<KeyValueMetadata>,
}

impl<W: Write> BatchedWriter<W> {
    pub fn new(
        writer: Mutex<FileWriter<W>>,
        column_options: Vec<ColumnWriteOptions>,
        options: WriteOptions,
        parallel: bool,
        key_value_metadata: Option<KeyValueMetadata>,
    ) -> Self {
        Self {
            writer,
            parquet_schema: SchemaDescriptor::new(PlSmallStr::EMPTY, vec![]),
            column_options,
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
                    self.column_options.as_ref(),
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
            &self.column_options,
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

    pub fn parquet_schema(&mut self) -> &SchemaDescriptor {
        let writer = self.writer.get_mut().unwrap();
        writer.parquet_schema()
    }

    pub fn write_row_group(&mut self, rg: &[Vec<CompressedPage>]) -> PolarsResult<()> {
        let writer = self.writer.get_mut().unwrap();
        let rg = DynIter::new(rg.iter().map(|col_pages| {
            Ok(DynStreamingIterator::new(
                fallible_streaming_iterator::convert(col_pages.iter().map(PolarsResult::Ok)),
            ))
        }));
        writer.write(rg)?;
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
            writer.write(group)?;
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
                let arrow_schema = schema_to_metadata_key(writer.schema(), &self.column_options);
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

        let size = writer.end(key_value_metadata, &self.column_options)?;
        Ok(size)
    }
}

// Note that the df should be rechunked
fn prepare_rg_iter<'a>(
    df: &'a DataFrame,
    parquet_schema: &'a SchemaDescriptor,
    column_options: &'a [ColumnWriteOptions],
    options: WriteOptions,
    parallel: bool,
) -> impl Iterator<Item = PolarsResult<RowGroupIterColumns<'static, PolarsError>>> + 'a {
    let rb_iter = df.iter_chunks(CompatLevel::newest(), false);
    rb_iter.filter_map(move |batch| match batch.len() {
        0 => None,
        _ => {
            let row_group = create_serializer(
                batch,
                parquet_schema.fields(),
                column_options,
                options,
                parallel,
            );

            Some(row_group)
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
    column_options: &ColumnWriteOptions,
    options: WriteOptions,
) -> Vec<PolarsResult<DynStreamingIterator<'static, CompressedPage, PolarsError>>> {
    let encoded_columns = array_to_columns(array, type_.clone(), column_options, options).unwrap();
    pages_iter_to_compressor(encoded_columns, options)
}

fn create_serializer(
    batch: RecordBatch,
    fields: &[ParquetType],
    column_options: &[ColumnWriteOptions],
    options: WriteOptions,
    parallel: bool,
) -> PolarsResult<RowGroupIterColumns<'static, PolarsError>> {
    let func = move |((array, type_), column_options): (
        (&ArrayRef, &ParquetType),
        &ColumnWriteOptions,
    )| { array_to_pages_iter(array, type_, column_options, options) };

    let columns = if parallel {
        POOL.install(|| {
            batch
                .columns()
                .par_iter()
                .zip(fields)
                .zip(column_options)
                .flat_map(func)
                .collect::<Vec<_>>()
        })
    } else {
        batch
            .columns()
            .iter()
            .zip(fields)
            .zip(column_options)
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
    column_options: &[ColumnWriteOptions],
    options: WriteOptions,
) -> PolarsResult<RowGroupIterColumns<'static, PolarsError>> {
    let func = move |((array, type_), column_options): (
        (&ArrayRef, &ParquetType),
        &ColumnWriteOptions,
    )| { array_to_pages_iter(array, type_, column_options, options) };

    let columns = batch
        .columns()
        .iter()
        .zip(fields)
        .zip(column_options)
        .flat_map(func)
        .collect::<Vec<_>>();

    let row_group = DynIter::new(columns.into_iter());

    Ok(row_group)
}
