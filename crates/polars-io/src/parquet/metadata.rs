//! Apache Parquet file metadata.

use std::sync::Arc;

pub use polars_parquet::parquet::metadata::FileMetaData;
pub use polars_parquet::read::statistics::{deserialize, Statistics as ParquetStatistics};

pub type FileMetaDataRef = Arc<FileMetaData>;
