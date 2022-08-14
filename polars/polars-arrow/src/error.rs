use std::borrow::Cow;

use thiserror::Error as ThisError;

type ErrString = Cow<'static, str>;

#[derive(Debug, ThisError)]
pub enum PolarsError {
    #[error(transparent)]
    ArrowError(#[from] arrow::error::Error),
    #[error("{0}")]
    ComputeError(ErrString),
    #[error("Out of bounds: {0}")]
    OutOfBounds(ErrString),
}

pub type Result<T> = std::result::Result<T, PolarsError>;
