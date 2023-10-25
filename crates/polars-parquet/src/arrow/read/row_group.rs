use std::io::{Read, Seek};

use arrow::array::Array;
use arrow::chunk::Chunk;
use arrow::datatypes::Field;
use parquet2::indexes::FilteredPage;
use parquet2::metadata::ColumnChunkMetaData;
use parquet2::read::{BasicDecompressor, IndexedPageReader, PageMetaData, PageReader};
use polars_error::PolarsResult;

use super::{ArrayIter, RowGroupMetaData};
use crate::arrow::read::column_iter_to_arrays;

/// An [`Iterator`] of [`Chunk`] that (dynamically) adapts a vector of iterators of [`Array`] into
/// an iterator of [`Chunk`].
///
/// This struct tracks advances each of the iterators individually and combines the
/// result in a single [`Chunk`].
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
    type Item = PolarsResult<Chunk<Box<dyn Array>>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_rows == 0 {
            return None;
        }
        let chunk = self
            .column_chunks
            .iter_mut()
            .map(|iter| iter.next().unwrap())
            .collect::<PolarsResult<Vec<_>>>()
            .and_then(Chunk::try_new);
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

type Pages = Box<
    dyn Iterator<Item = std::result::Result<parquet2::page::CompressedPage, parquet2::error::Error>>
        + Sync
        + Send,
>;

/// Converts a vector of columns associated with the parquet field whose name is [`Field`]
/// to an iterator of [`Array`], [`ArrayIter`] of chunk size `chunk_size`.
pub fn to_deserializer<'a>(
    columns: Vec<(&ColumnChunkMetaData, Vec<u8>)>,
    field: Field,
    num_rows: usize,
    chunk_size: Option<usize>,
    pages: Option<Vec<Vec<FilteredPage>>>,
) -> PolarsResult<ArrayIter<'a>> {
    let chunk_size = chunk_size.map(|c| c.min(num_rows));

    let (columns, types) = if let Some(pages) = pages {
        let (columns, types): (Vec<_>, Vec<_>) = columns
            .into_iter()
            .zip(pages)
            .map(|((column_meta, chunk), mut pages)| {
                // de-offset the start, since we read in chunks (and offset is from start of file)
                let mut meta: PageMetaData = column_meta.into();
                pages
                    .iter_mut()
                    .for_each(|page| page.start -= meta.column_start);
                meta.column_start = 0;
                let pages = IndexedPageReader::new_with_page_meta(
                    std::io::Cursor::new(chunk),
                    meta,
                    pages,
                    vec![],
                    vec![],
                );
                let pages = Box::new(pages) as Pages;
                (
                    BasicDecompressor::new(pages, vec![]),
                    &column_meta.descriptor().descriptor.primitive_type,
                )
            })
            .unzip();

        (columns, types)
    } else {
        let (columns, types): (Vec<_>, Vec<_>) = columns
            .into_iter()
            .map(|(column_meta, chunk)| {
                let len = chunk.len();
                let pages = PageReader::new(
                    std::io::Cursor::new(chunk),
                    column_meta,
                    std::sync::Arc::new(|_, _| true),
                    vec![],
                    len * 2 + 1024,
                );
                let pages = Box::new(pages) as Pages;
                (
                    BasicDecompressor::new(pages, vec![]),
                    &column_meta.descriptor().descriptor.primitive_type,
                )
            })
            .unzip();

        (columns, types)
    };

    column_iter_to_arrays(columns, types, field, chunk_size, num_rows)
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
    chunk_size: Option<usize>,
    limit: Option<usize>,
    pages: Option<Vec<Vec<Vec<FilteredPage>>>>,
) -> PolarsResult<Vec<ArrayIter<'a>>> {
    let num_rows = row_group.num_rows();
    let num_rows = limit.map(|limit| limit.min(num_rows)).unwrap_or(num_rows);

    // reads all the necessary columns for all fields from the row group
    // This operation is IO-bounded `O(C)` where C is the number of columns in the row group
    let field_columns = fields
        .iter()
        .map(|field| read_columns(reader, row_group.columns(), &field.name))
        .collect::<PolarsResult<Vec<_>>>()?;

    if let Some(pages) = pages {
        field_columns
            .into_iter()
            .zip(fields)
            .zip(pages)
            .map(|((columns, field), pages)| {
                to_deserializer(columns, field, num_rows, chunk_size, Some(pages))
            })
            .collect()
    } else {
        field_columns
            .into_iter()
            .zip(fields)
            .map(|(columns, field)| to_deserializer(columns, field, num_rows, chunk_size, None))
            .collect()
    }
}
