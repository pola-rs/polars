//! Functionality for reading and writing Apache Parquet files.

mod batched_writer;
mod options;
mod writer;

pub use batched_writer::BatchedWriter;
pub use options::{BrotliLevel, GzipLevel, ParquetCompression, ParquetWriteOptions, ZstdLevel, SortingColumnBehavior};
pub use polars_parquet::write::{RowGroupIterColumns, StatisticsOptions};
pub use writer::{SortingColumns, ParquetWriter};
