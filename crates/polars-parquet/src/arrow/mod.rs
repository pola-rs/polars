pub mod read;
pub mod write;

#[cfg(feature = "bloom_filter")]
#[cfg_attr(docsrs, doc(cfg(feature = "bloom_filter")))]
pub use crate::parquet::bloom_filter;

const ARROW_SCHEMA_META_KEY: &str = "ARROW:schema";
