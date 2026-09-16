mod reader;

pub use reader::{PageMetaData, PageReader};

use crate::parquet::error::ParquetError;
use crate::parquet::page::CompressedPage;

pub trait PageIterator: Iterator<Item = Result<CompressedPage, ParquetError>> {
    fn swap_buffer(&mut self, buffer: &mut Vec<u8>);
}
