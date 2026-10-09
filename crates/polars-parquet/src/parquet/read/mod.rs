mod column;
mod compression;
pub mod levels;
mod metadata;
mod page;

use std::io::{Cursor, Seek, SeekFrom};

pub use column::*;
pub use compression::{BasicDecompressor, decompress};
pub use metadata::{
    deserialize_file_crypto_metadata, deserialize_metadata, deserialize_num_rows, read_metadata,
    read_metadata_with_decryption, read_num_rows,
};
pub use page::{PageIterator, PageMetaData, PageReader};
use polars_buffer::Buffer;

use crate::parquet::error::ParquetResult;
use crate::parquet::metadata::ColumnChunkMetadata;

/// Returns a new [`PageReader`] by seeking `reader` to the beginning of `column_chunk`.
pub fn get_page_iterator(
    column_chunk: &ColumnChunkMetadata,
    mut reader: Cursor<Buffer<u8>>,
    scratch: Vec<u8>,
    max_page_size: usize,
) -> ParquetResult<PageReader> {
    let col_start = column_chunk.byte_range()?.start;
    reader.seek(SeekFrom::Start(col_start))?;
    PageReader::new(reader, column_chunk, scratch, max_page_size)
}
