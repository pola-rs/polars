pub mod read;
pub mod write;

#[cfg(feature = "io_parquet_bloom_filter")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_parquet_bloom_filter")))]
pub use parquet2::bloom_filter;

const ARROW_SCHEMA_META_KEY: &str = "ARROW:schema";
