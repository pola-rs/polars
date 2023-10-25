mod indexed_reader;
mod reader;
#[cfg(feature = "async")]
mod stream;

use crate::{error::Error, page::CompressedPage};

pub use indexed_reader::IndexedPageReader;
pub use reader::{PageFilter, PageMetaData, PageReader};

pub trait PageIterator: Iterator<Item = Result<CompressedPage, Error>> {
    fn swap_buffer(&mut self, buffer: &mut Vec<u8>);
}

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub use stream::{get_page_stream, get_page_stream_from_column_start};
