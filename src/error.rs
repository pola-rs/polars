use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum PolarsError {
    #[error(transparent)]
    ArrowError(#[from] arrow::error::ArrowError),
    #[error("Invalid operation")]
    InvalidOperation,
}

pub type Result<T> = std::result::Result<T, PolarsError>;
