use std::borrow::Cow;
use thiserror::Error as ThisError;

type ErrString = Cow<'static, str>;

#[derive(Debug, ThisError)]
pub enum PolarsError {
    #[error(transparent)]
    ArrowError(#[from] arrow::error::ArrowError),
    #[error("Invalid operation")]
    InvalidOperation,
    #[error("Chunks don't match")]
    ChunkMisMatch,
    #[error("Data types don't match")]
    DataTypeMisMatch,
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Lengths don't match")]
    ShapeMisMatch,
    #[error("{0}")]
    Other(ErrString),
    #[error("No selection was made")]
    NoSelection,
    #[error("Out of bounds")]
    OutOfBounds,
    #[error("Not contiguous or null values")]
    NoSlice,
    #[error("Such empty...")]
    NoData,
    #[error("Memory should be 64 byte aligned")]
    MemoryNotAligned,
    #[cfg(feature = "parquet")]
    #[error(transparent)]
    ParquetError(#[from] parquet::errors::ParquetError),
    #[cfg(feature = "random")]
    #[error("{0}")]
    RandError(String),
    #[error("This operation requires data without Null values")]
    HasNullValues,
    #[error("{0}")]
    UnknownSchema(ErrString),
}

pub type Result<T> = std::result::Result<T, PolarsError>;
