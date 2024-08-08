use std::io::{Read, Seek};

use arrow::array::Array;
use arrow::datatypes::Field;
use arrow::record_batch::RecordBatchT;
use polars_error::PolarsResult;
use polars_utils::mmap::MemReader;

use super::{ArrayIter, RowGroupMetaData};
use crate::arrow::read::column_iter_to_arrays;
use crate::arrow::read::deserialize::Filter;
use crate::parquet::metadata::ColumnChunkMetaData;
use crate::parquet::read::{BasicDecompressor, PageReader};

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
    column_chunks: Vec<ArrayIter<'static>>,
}

impl RowGroupDeserializer {
    /// Creates a new [`RowGroupDeserializer`].
    ///
    /// # Panic
    /// This function panics iff any of the `column_chunks`
    /// do not return an array with an equal length.
    pub fn new(
        column_chunks: Vec<ArrayIter<'static>>,
        num_rows: usize,
        limit: Option<usize>,
    ) -> Self {
        Self {
            num_rows,
            remaining_rows: limit.unwrap_or(usize::MAX).min(num_rows),
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
        let chunk = self
            .column_chunks
            .iter_mut()
            .map(|iter| iter.next().unwrap())
            .collect::<PolarsResult<Vec<_>>>()
            .and_then(RecordBatchT::try_new);
        self.remaining_rows = self.remaining_rows.saturating_sub(
            chunk
                .as_ref()
                .map(|x| x.len())
                .unwrap_or(self.remaining_rows),
        );

        Some(chunk)
    }
}

/// Returns all [`ColumnChunkMetaData`] associated to `field_name`.
/// For non-nested parquet types, this returns a single column
pub fn get_field_columns<'a>(
    columns: &'a [ColumnChunkMetaData],
    field_name: &str,
) -> Vec<&'a ColumnChunkMetaData> {
    columns
        .iter()
        .filter(|x| x.descriptor().path_in_schema[0] == field_name)
        .collect()
}

/// Returns all [`ColumnChunkMetaData`] associated to `field_name`.
/// For non-nested parquet types, this returns a single column
pub fn get_field_pages<'a, T>(
    columns: &'a [ColumnChunkMetaData],
    items: &'a [T],
    field_name: &str,
) -> Vec<&'a T> {
    columns
        .iter()
        .zip(items)
        .filter(|(metadata, _)| metadata.descriptor().path_in_schema[0] == field_name)
        .map(|(_, item)| item)
        .collect()
}

/// Reads all columns that are part of the parquet field `field_name`
/// # Implementation
/// This operation is IO-bounded `O(C)` where C is the number of columns associated to
/// the field (one for non-nested types)
pub fn read_columns<'a, R: Read + Seek>(
    reader: &mut R,
    columns: &'a [ColumnChunkMetaData],
    field_name: &str,
) -> PolarsResult<Vec<(&'a ColumnChunkMetaData, Vec<u8>)>> {
    get_field_columns(columns, field_name)
        .into_iter()
        .map(|meta| _read_single_column(reader, meta))
        .collect()
}

fn _read_single_column<'a, R>(
    reader: &mut R,
    meta: &'a ColumnChunkMetaData,
) -> PolarsResult<(&'a ColumnChunkMetaData, Vec<u8>)>
where
    R: Read + Seek,
{
    let (start, length) = meta.byte_range();
    reader.seek(std::io::SeekFrom::Start(start))?;

    let mut chunk = vec![];
    chunk.try_reserve(length as usize)?;
    reader.by_ref().take(length).read_to_end(&mut chunk)?;
    Ok((meta, chunk))
}

/// Converts a vector of columns associated with the parquet field whose name is [`Field`]
/// to an iterator of [`Array`], [`ArrayIter`] of chunk size `chunk_size`.
pub fn to_deserializer<'a>(
    columns: Vec<(&ColumnChunkMetaData, Vec<u8>)>,
    field: Field,
    filter: Option<Filter>,
) -> PolarsResult<ArrayIter<'a>> {
    let (columns, types): (Vec<_>, Vec<_>) = columns
        .into_iter()
        .map(|(column_meta, chunk)| {
            let len = chunk.len();
            let pages = PageReader::new(
                MemReader::from_vec(chunk),
                column_meta,
                std::sync::Arc::new(|_, _| true),
                vec![],
                len * 2 + 1024,
            );
            (
                BasicDecompressor::new(pages, vec![]),
                &column_meta.descriptor().descriptor.primitive_type,
            )
        })
        .unzip();

    column_iter_to_arrays(columns, types, field, filter)
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
pub fn read_columns_many<'a, R: Read + Seek>(
    reader: &mut R,
    row_group: &RowGroupMetaData,
    fields: Vec<Field>,
    filter: Option<Filter>,
) -> PolarsResult<Vec<ArrayIter<'a>>> {
    // reads all the necessary columns for all fields from the row group
    // This operation is IO-bounded `O(C)` where C is the number of columns in the row group
    let field_columns = fields
        .iter()
        .map(|field| read_columns(reader, row_group.columns(), &field.name))
        .collect::<PolarsResult<Vec<_>>>()?;

    field_columns
        .into_iter()
        .zip(fields)
        .map(|(columns, field)| to_deserializer(columns, field, filter.clone()))
        .collect()
}
