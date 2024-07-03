use arrow::array::Array;
use arrow::datatypes::ArrowSchema;
use arrow::record_batch::RecordBatchT;
use polars_error::{polars_bail, to_compute_err, PolarsError, PolarsResult};

use super::{
    array_to_columns, to_parquet_schema, DynIter, DynStreamingIterator, Encoding,
    RowGroupIterColumns, SchemaDescriptor, WriteOptions,
};
use crate::parquet::error::ParquetError;
use crate::parquet::schema::types::ParquetType;
use crate::parquet::write::Compressor;
use crate::parquet::FallibleStreamingIterator;

/// Maps a [`RecordBatchT`] and parquet-specific options to an [`RowGroupIterColumns`] used to
/// write to parquet
/// # Panics
/// Iff
/// * `encodings.len() != fields.len()` or
/// * `encodings.len() != chunk.arrays().len()`
pub fn row_group_iter<A: AsRef<dyn Array> + 'static + Send + Sync>(
    chunk: RecordBatchT<A>,
    encodings: Vec<Vec<Encoding>>,
    fields: Vec<ParquetType>,
    options: WriteOptions,
) -> RowGroupIterColumns<'static, PolarsError> {
    assert_eq!(encodings.len(), fields.len());
    assert_eq!(encodings.len(), chunk.arrays().len());
    DynIter::new(
        chunk
            .into_arrays()
            .into_iter()
            .zip(fields)
            .zip(encodings)
            .flat_map(move |((array, type_), encoding)| {
                let encoded_columns = array_to_columns(array, type_, options, &encoding).unwrap();
                encoded_columns
                    .into_iter()
                    .map(|encoded_pages| {
                        let pages = encoded_pages;

                        let pages = DynIter::new(
                            pages
                                .into_iter()
                                .map(|x| x.map_err(|e| ParquetError::oos(e.to_string()))),
                        );

                        let compressed_pages = Compressor::new(pages, options.compression, vec![])
                            .map_err(to_compute_err);
                        Ok(DynStreamingIterator::new(compressed_pages))
                    })
                    .collect::<Vec<_>>()
            }),
    )
}

/// An iterator adapter that converts an iterator over [`RecordBatchT`] into an iterator
/// of row groups.
/// Use it to create an iterator consumable by the parquet's API.
pub struct RowGroupIterator<
    A: AsRef<dyn Array> + 'static,
    I: Iterator<Item = PolarsResult<RecordBatchT<A>>>,
> {
    iter: I,
    options: WriteOptions,
    parquet_schema: SchemaDescriptor,
    encodings: Vec<Vec<Encoding>>,
}

impl<A: AsRef<dyn Array> + 'static, I: Iterator<Item = PolarsResult<RecordBatchT<A>>>>
    RowGroupIterator<A, I>
{
    /// Creates a new [`RowGroupIterator`] from an iterator over [`RecordBatchT`].
    ///
    /// # Errors
    /// Iff
    /// * the Arrow schema can't be converted to a valid Parquet schema.
    /// * the length of the encodings is different from the number of fields in schema
    pub fn try_new(
        iter: I,
        schema: &ArrowSchema,
        options: WriteOptions,
        encodings: Vec<Vec<Encoding>>,
    ) -> PolarsResult<Self> {
        if encodings.len() != schema.fields.len() {
            polars_bail!(InvalidOperation:
                "The number of encodings must equal the number of fields".to_string(),
            )
        }
        let parquet_schema = to_parquet_schema(schema)?;

        Ok(Self {
            iter,
            options,
            parquet_schema,
            encodings,
        })
    }

    /// Returns the [`SchemaDescriptor`] of the [`RowGroupIterator`].
    pub fn parquet_schema(&self) -> &SchemaDescriptor {
        &self.parquet_schema
    }
}

impl<
        A: AsRef<dyn Array> + 'static + Send + Sync,
        I: Iterator<Item = PolarsResult<RecordBatchT<A>>>,
    > Iterator for RowGroupIterator<A, I>
{
    type Item = PolarsResult<RowGroupIterColumns<'static, PolarsError>>;

    fn next(&mut self) -> Option<Self::Item> {
        let options = self.options;

        self.iter.next().map(|maybe_chunk| {
            let chunk = maybe_chunk?;
            if self.encodings.len() != chunk.arrays().len() {
                polars_bail!(InvalidOperation:
                    "The number of arrays in the chunk must equal the number of fields in the schema"
                )
            };
            let encodings = self.encodings.clone();
            Ok(row_group_iter(
                chunk,
                encodings,
                self.parquet_schema.fields().to_vec(),
                options,
            ))
        })
    }
}
