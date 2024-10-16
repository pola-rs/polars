use std::io::Write;
use std::sync::Mutex;

use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::POOL;
use polars_parquet::read::ParquetError;
use polars_parquet::write::{
    array_to_columns, CompressedPage, Compressor, DynIter, DynStreamingIterator, Encoding,
    FallibleStreamingIterator, FileWriter, Page, ParquetType, RowGroupIterColumns,
    RowGroupWriteOptions, SchemaDescriptor, SortingColumn, WriteOptions,
};
use rayon::prelude::*;

use super::options::{MaterializedSortingColumns, MetadataOptions, SortingColumnBehavior};

pub struct BatchedWriter<W: Write> {
    // A mutex so that streaming engine can get concurrent read access to
    // compress pages.
    pub(super) writer: Mutex<FileWriter<W>>,
    pub(super) parquet_schema: SchemaDescriptor,
    pub(super) encodings: Vec<Vec<Encoding>>,
    pub(super) options: WriteOptions,
    pub(super) metadata_options: MetadataOptions,
    pub(super) parallel: bool,
}

impl<W: Write> BatchedWriter<W> {
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
            &self.metadata_options,
            self.parallel,
        );
        // Lock before looping so that order is maintained under contention.
        let mut writer = self.writer.lock().unwrap();
        for item in row_group_iter {
            let (group, rg_options) = item?;
            writer.write(group, rg_options)?;
        }
        Ok(())
    }

    pub fn get_writer(&self) -> &Mutex<FileWriter<W>> {
        &self.writer
    }

    pub fn write_row_groups_default_options(
        &self,
        rgs: Vec<RowGroupIterColumns<'static, PolarsError>>,
    ) -> PolarsResult<()> {
        // Lock before looping so that order is maintained.
        let mut writer = self.writer.lock().unwrap();
        for group in rgs {
            writer.write(group, RowGroupWriteOptions::default())?;
        }
        Ok(())
    }

    pub fn write_row_groups(
        &self,
        rgs: Vec<(
            RowGroupIterColumns<'static, PolarsError>,
            RowGroupWriteOptions,
        )>,
    ) -> PolarsResult<()> {
        // Lock before looping so that order is maintained.
        let mut writer = self.writer.lock().unwrap();
        for (group, rg_options) in rgs {
            writer.write(group, rg_options)?;
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
    md_options: &'a MetadataOptions,
    parallel: bool,
) -> impl Iterator<
    Item = PolarsResult<(
        RowGroupIterColumns<'static, PolarsError>,
        RowGroupWriteOptions,
    )>,
> + 'a {
    // @TODO: This does not work for nested columns.
    let sortedness = df
        .get_columns()
        .iter()
        .map(|c| c.is_sorted_flag())
        .collect::<Vec<_>>();
    // @TODO: This does not work for nested columns.
    let dtypes = df
        .get_columns()
        .iter()
        .map(|c| c.dtype())
        .collect::<Vec<_>>();

    let rb_iter = df.iter_chunks(CompatLevel::newest(), false);
    rb_iter.filter_map(move |batch| match batch.len() {
        0 => None,
        _ => Some(create_serializer(
            batch,
            parquet_schema.fields(),
            encodings,
            &sortedness,
            &dtypes,
            options,
            md_options,
            parallel,
        )),
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
    sortedness: &[IsSorted],
    dtypes: &[&DataType],
    options: WriteOptions,
    md_options: &MetadataOptions,
    parallel: bool,
) -> PolarsResult<(
    RowGroupIterColumns<'static, PolarsError>,
    RowGroupWriteOptions,
)> {
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

    let mut rg_options = RowGroupWriteOptions::default();

    match &md_options.sorting_columns {
        MaterializedSortingColumns::All(behavior) => {
            // @TODO: properly handle nested columns.
            rg_options.sorting_columns = (0..batch.columns().len())
                .filter_map(|leaf_idx| {
                    to_sorting_column(&batch, dtypes, sortedness, leaf_idx, *behavior)
                })
                .collect();
        },
        MaterializedSortingColumns::PerLeaf(sorting_columns) => {
            rg_options.sorting_columns = sorting_columns
                .iter()
                .filter_map(|(leaf_idx, behavior)| {
                    to_sorting_column(&batch, dtypes, sortedness, *leaf_idx as usize, *behavior)
                })
                .collect();
        },
    }

    Ok((row_group, rg_options))
}

fn has_compatible_sortedness(dtype: &DataType, _array: &dyn Array) -> bool {
    use DataType as DT;

    matches!(
        dtype,
        DT::UInt8
            | DT::UInt16
            | DT::UInt32
            | DT::UInt64
            | DT::Int8
            | DT::Int16
            | DT::Int32
            | DT::Int64
    )
}

fn to_sorting_column(
    batch: &RecordBatch,
    dtypes: &[&DataType],
    sortedness: &[IsSorted],
    leaf_idx: usize,
    behavior: SortingColumnBehavior,
) -> Option<SortingColumn> {
    use SortingColumnBehavior as B;

    // @TODO: This does not work for nested structures.
    let col_idx = leaf_idx;
    let array = &batch.columns()[col_idx as usize];
    let dtype = dtypes[leaf_idx as usize];

    if matches!(
        behavior,
        B::Preserve { force: false } | B::Evaluate { force: false }
    ) {
        if !has_compatible_sortedness(dtype, array.as_ref()) {
            return None;
        }
    }

    match (behavior, sortedness[leaf_idx as usize]) {
        (B::NoPreserve, _) => None,
        (
            B::Force {
                descending,
                nulls_first,
            },
            _,
        ) => Some(SortingColumn {
            column_idx: leaf_idx as i32,
            descending,
            nulls_first,
        }),
        (B::Preserve { .. }, IsSorted::Not) => None,
        (B::Preserve { .. } | B::Evaluate { .. }, IsSorted::Ascending) => {
            let nulls_first = !array.is_empty() && unsafe { array.get_unchecked(0) }.is_null();
            Some(SortingColumn {
                column_idx: leaf_idx as i32,
                descending: false,
                nulls_first,
            })
        },
        (B::Preserve { .. } | B::Evaluate { .. }, IsSorted::Descending) => {
            let nulls_first = !array.is_empty() && unsafe { array.get_unchecked(0) }.is_null();
            Some(SortingColumn {
                column_idx: leaf_idx as i32,
                descending: true,
                nulls_first,
            })
        },
        (B::Evaluate { .. }, IsSorted::Not) => todo!(),
    }
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
