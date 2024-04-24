use std::sync::Arc;

pub use polars_parquet::parquet::metadata::FileMetaData;
pub type FileMetaDataRef = Arc<FileMetaData>;
