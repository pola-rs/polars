mod column;
mod compression;
mod indexes;
pub mod levels;
mod metadata;
mod page;
#[cfg(feature = "async")]
mod stream;

use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

pub use column::*;
pub use compression::{decompress, BasicDecompressor, Decompressor};
pub use indexes::{read_columns_indexes, read_pages_locations};
pub use metadata::{deserialize_metadata, read_metadata, read_metadata_with_size};
#[cfg(feature = "async")]
pub use page::{get_page_stream, get_page_stream_from_column_start};
pub use page::{IndexedPageReader, PageFilter, PageIterator, PageMetaData, PageReader};
#[cfg(feature = "async")]
pub use stream::read_metadata as read_metadata_async;

use crate::parquet::error::Result;
use crate::parquet::metadata::{ColumnChunkMetaData, FileMetaData, RowGroupMetaData};

/// Filters row group metadata to only those row groups,
/// for which the predicate function returns true
pub fn filter_row_groups(
    metadata: &FileMetaData,
    predicate: &dyn Fn(&RowGroupMetaData, usize) -> bool,
) -> FileMetaData {
    let mut filtered_row_groups = Vec::<RowGroupMetaData>::new();
    for (i, row_group_metadata) in metadata.row_groups.iter().enumerate() {
        if predicate(row_group_metadata, i) {
            filtered_row_groups.push(row_group_metadata.clone());
        }
    }
    let mut metadata = metadata.clone();
    metadata.row_groups = filtered_row_groups;
    metadata
}

/// Returns a new [`PageReader`] by seeking `reader` to the beginning of `column_chunk`.
pub fn get_page_iterator<R: Read + Seek>(
    column_chunk: &ColumnChunkMetaData,
    mut reader: R,
    pages_filter: Option<PageFilter>,
    scratch: Vec<u8>,
    max_page_size: usize,
) -> Result<PageReader<R>> {
    let pages_filter = pages_filter.unwrap_or_else(|| Arc::new(|_, _| true));

    let (col_start, _) = column_chunk.byte_range();
    reader.seek(SeekFrom::Start(col_start))?;
    Ok(PageReader::new(
        reader,
        column_chunk,
        pages_filter,
        scratch,
        max_page_size,
    ))
}

/// Returns all [`ColumnChunkMetaData`] associated to `field_name`.
/// For non-nested types, this returns an iterator with a single column
pub fn get_field_columns<'a>(
    columns: &'a [ColumnChunkMetaData],
    field_name: &'a str,
) -> impl Iterator<Item = &'a ColumnChunkMetaData> {
    columns
        .iter()
        .filter(move |x| x.descriptor().path_in_schema[0] == field_name)
}
