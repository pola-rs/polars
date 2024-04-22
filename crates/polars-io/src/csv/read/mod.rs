pub(crate) mod buffer;
pub mod options;
pub(crate) mod parser;
mod read_impl;
mod reader;
pub(super) mod splitfields;
mod utils;

pub use options::{CommentPrefix, CsvEncoding, CsvParserOptions, NullValues};
pub use parser::count_rows;
use read_impl::{
    to_batched_owned_mmap, to_batched_owned_read, CoreReader, OwnedBatchedCsvReader,
    OwnedBatchedCsvReaderMmap,
};
pub use read_impl::{BatchedCsvReaderMmap, BatchedCsvReaderRead};
pub use reader::CsvReader;
pub use utils::{infer_file_schema, is_compressed};
