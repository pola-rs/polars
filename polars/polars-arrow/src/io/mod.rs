#[cfg(feature = "parquet")]
mod parquet;

#[cfg(feature = "parquet")]
pub use parquet::read_parquet;
