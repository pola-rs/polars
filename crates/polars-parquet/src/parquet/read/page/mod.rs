mod reader;
#[cfg(feature = "async")]
mod stream;

pub use reader::{PageMetaData, PageReader};

use crate::parquet::error::ParquetError;
use crate::parquet::page::CompressedPage;

pub trait PageIterator: Iterator<Item = Result<CompressedPage, ParquetError>> {
    fn swap_buffer(&mut self, buffer: &mut Vec<u8>);
}

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub use stream::{get_page_stream, get_page_stream_from_column_start};
