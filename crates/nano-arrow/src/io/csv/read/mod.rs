//! APIs to read from CSV
mod deserialize;
mod reader;

// Re-export for usage by consumers.
pub use csv::{ByteRecord, Reader, ReaderBuilder};

mod infer_schema;

pub use super::utils::infer;
pub use deserialize::{deserialize_batch, deserialize_column};
pub use infer_schema::infer_schema;
pub use reader::*;
