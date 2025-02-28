use std::io::{Read, Seek};

use arrow::array::Array;
use arrow::datatypes::{ArrowSchemaRef, Field};
use arrow::record_batch::RecordBatchT;
use polars::prelude::ArrowSchema;
use polars_error::PolarsResult;
use polars_parquet::arrow::read::{column_iter_to_arrays, Filter};
use polars_parquet::parquet::metadata::ColumnChunkMetadata;
use polars_parquet::parquet::read::{BasicDecompressor, PageReader};
use polars_parquet::read::RowGroupMetadata;
use polars_utils::mmap::MemReader;

/// An [`Iterator`] of [`RecordBatchT`] that (dynamically) adapts a vector of iterators of [`Array`] into
/// an iterator of [`RecordBatchT`].
///
/// This struct tracks advances each of the iterators individually and combines the
/// result in a single [`RecordBatchT`].
///
/// # Implementation
/// This iterator is single-threaded and advancing it is CPU-bounded.
pub struct RowGroupDeserializer {
    num_rows: usize,
    remaining_rows: usize,
    column_schema: ArrowSchemaRef,
    column_chunks: Vec<Box<dyn Array>>,
}

impl RowGroupDeserializer {
    /// Creates a new [`RowGroupDeserializer`].
    ///
    /// # Panic
    /// This function panics iff any of the `column_chunks`
    /// do not return an array with an equal length.
    pub fn new(
        column_schema: ArrowSchemaRef,
        column_chunks: Vec<Box<dyn Array>>,
        num_rows: usize,
        limit: Option<usize>,
    ) -> Self {
        Self {
            num_rows,
            remaining_rows: limit.unwrap_or(usize::MAX).min(num_rows),
            column_schema,
            column_chunks,
        }
    }

    /// Returns the number of rows on this row group
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }
}

impl Iterator for RowGroupDeserializer {
    type Item = PolarsResult<RecordBatchT<Box<dyn Array>>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_rows == 0 {
            return None;
        }
        let length = self.column_chunks.first().map_or(0, |chunk| chunk.len());
        let chunk = RecordBatchT::try_new(
            length,
            self.column_schema.clone(),
            std::mem::take(&mut self.column_chunks),
        );
        self.remaining_rows = self.remaining_rows.saturating_sub(
            chunk
                .as_ref()
                .map(|x| x.len())
                .unwrap_or(self.remaining_rows),
        );

        Some(chunk)
    }
}

/// Reads all columns that are part of the parquet field `field_name`
/// # Implementation
/// This operation is IO-bounded `O(C)` where C is the number of columns associated to
/// the field (one for non-nested types)
pub fn read_columns<'a, R: Read + Seek>(
    reader: &mut R,
    row_group_metadata: &'a RowGroupMetadata,
    field_name: &'a str,
) -> PolarsResult<Vec<(&'a ColumnChunkMetadata, Vec<u8>)>> {
    row_group_metadata
        .columns_under_root_iter(field_name)
        .unwrap()
        .map(|meta| _read_single_column(reader, meta))
        .collect()
}

fn _read_single_column<'a, R>(
    reader: &mut R,
    meta: &'a ColumnChunkMetadata,
) -> PolarsResult<(&'a ColumnChunkMetadata, Vec<u8>)>
where
    R: Read + Seek,
{
    let byte_range = meta.byte_range();
    let length = byte_range.end - byte_range.start;
    reader.seek(std::io::SeekFrom::Start(byte_range.start))?;

    let mut chunk = vec![];
    chunk.try_reserve(length as usize)?;
    reader.by_ref().take(length).read_to_end(&mut chunk)?;
    Ok((meta, chunk))
}

/// Converts a vector of columns associated with the parquet field whose name is [`Field`]
/// to an iterator of [`Array`], [`ArrayIter`] of chunk size `chunk_size`.
pub fn to_deserializer(
    columns: Vec<(&ColumnChunkMetadata, Vec<u8>)>,
    field: Field,
    filter: Option<Filter>,
) -> PolarsResult<Box<dyn Array>> {
    let (columns, types): (Vec<_>, Vec<_>) = columns
        .into_iter()
        .map(|(column_meta, chunk)| {
            let len = chunk.len();
            let pages = PageReader::new(
                MemReader::from_vec(chunk),
                column_meta,
                vec![],
                len * 2 + 1024,
            );
            (
                BasicDecompressor::new(pages, vec![]),
                &column_meta.descriptor().descriptor.primitive_type,
            )
        })
        .unzip();

    column_iter_to_arrays(columns, types, field, filter).map(|v| v.0)
}

/// Returns a vector of iterators of [`Array`] ([`ArrayIter`]) corresponding to the top
/// level parquet fields whose name matches `fields`'s names.
///
/// # Implementation
/// This operation is IO-bounded `O(C)` where C is the number of columns in the row group -
/// it reads all the columns to memory from the row group associated to the requested fields.
///
/// This operation is single-threaded. For readers with stronger invariants
/// (e.g. implement [`Clone`]) you can use [`read_columns`] to read multiple columns at once
/// and convert them to [`ArrayIter`] via [`to_deserializer`].
pub fn read_columns_many<R: Read + Seek>(
    reader: &mut R,
    row_group: &RowGroupMetadata,
    fields: &ArrowSchema,
    filter: Option<Filter>,
) -> PolarsResult<Vec<Box<dyn Array>>> {
    // reads all the necessary columns for all fields from the row group
    // This operation is IO-bounded `O(C)` where C is the number of columns in the row group
    let field_columns = fields
        .iter_values()
        .map(|field| read_columns(reader, row_group, &field.name))
        .collect::<PolarsResult<Vec<_>>>()?;

    field_columns
        .into_iter()
        .zip(fields.iter_values().cloned())
        .map(|(columns, field)| to_deserializer(columns.clone(), field, filter.clone()))
        .collect()
}
