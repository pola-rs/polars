//! Functionality for reading and writing Apache Parquet files.

mod batched_writer;
mod key_value_metadata;
mod options;
mod writer;

pub use batched_writer::BatchedWriter;
pub use key_value_metadata::{KeyValueMetadata, ParquetMetadataContext};
pub use options::{
    BrotliLevel, ChildFieldOverwrites, GzipLevel, MetadataKeyValue, ParquetCompression,
    ParquetFieldOverwrites, ParquetWriteOptions, ZstdLevel,
};
pub use polars_parquet::write::{RowGroupIterColumns, StatisticsOptions};
pub use writer::{ParquetWriter, get_column_write_options};
