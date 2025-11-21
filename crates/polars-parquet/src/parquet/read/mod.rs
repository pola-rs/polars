mod column;
mod compression;
pub mod levels;
mod metadata;
mod page;
#[cfg(feature = "async")]
mod stream;

use std::io::{Seek, SeekFrom};

pub use column::*;
pub use compression::{BasicDecompressor, decompress};
pub use metadata::{deserialize_metadata, read_metadata, read_metadata_with_size};
pub use page::{PageIterator, PageMetaData, PageReader};
#[cfg(feature = "async")]
pub use page::{get_page_stream, get_page_stream_from_column_start};
use polars_utils::mmap::MemReader;
#[cfg(feature = "async")]
pub use stream::read_metadata as read_metadata_async;

use crate::parquet::error::ParquetResult;
use crate::parquet::metadata::ColumnChunkMetadata;

/// Returns a new [`PageReader`] by seeking `reader` to the beginning of `column_chunk`.
pub fn get_page_iterator(
    column_chunk: &ColumnChunkMetadata,
    mut reader: MemReader,
    scratch: Vec<u8>,
    max_page_size: usize,
) -> ParquetResult<PageReader> {
    let col_start = column_chunk.byte_range().start;
    reader.seek(SeekFrom::Start(col_start))?;
    Ok(PageReader::new(
        reader,
        column_chunk,
        scratch,
        max_page_size,
    ))
}
