use std::borrow::Cow;
use thiserror::Error as ThisError;

type ErrString = Cow<'static, str>;

#[derive(Debug, ThisError)]
pub enum PolarsError {
    #[error(transparent)]
    ArrowError(#[from] arrow::error::ArrowError),
    #[error("Invalid operation {0}")]
    InvalidOperation(ErrString),
    #[error("Data types don't match: {0}")]
    DataTypeMisMatch(ErrString),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Lengths don't match: {0}")]
    ShapeMisMatch(ErrString),
    #[error("{0}")]
    Other(ErrString),
    #[error("Out of bounds: {0}")]
    OutOfBounds(ErrString),
    #[error("Not contiguous or null values")]
    NoSlice,
    #[error("Such empty...: {0}")]
    NoData(ErrString),
    #[error("Invalid value: {0}")]
    ValueError(ErrString),
    #[error("Memory should be 64 byte aligned")]
    MemoryNotAligned,
    #[cfg(feature = "parquet")]
    #[error(transparent)]
    ParquetError(#[from] parquet::errors::ParquetError),
    #[cfg(feature = "random")]
    #[error("{0}")]
    RandError(String),
    #[error("This operation requires data without Null values")]
    HasNullValues(ErrString),
    #[error("{0}")]
    UnknownSchema(ErrString),
    #[error(transparent)]
    Various(#[from] anyhow::Error),
    #[error(transparent)]
    IO(#[from] std::io::Error),
    #[error(transparent)]
    Regex(#[from] regex::Error),
}

pub type Result<T> = std::result::Result<T, PolarsError>;
