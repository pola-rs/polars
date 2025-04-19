//! Apache Parquet file metadata.

use std::sync::Arc;

pub use polars_parquet::parquet::metadata::FileMetadata;
pub use polars_parquet::read::statistics::{Statistics as ParquetStatistics, deserialize};

pub type FileMetadataRef = Arc<FileMetadata>;
