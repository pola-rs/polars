use std::fmt::{Display, Formatter};

use anyhow::Error;
use thiserror::Error as ThisError;

#[derive(Debug)]
pub enum ErrString {
    Owned(String),
    Borrowed(&'static str),
}

impl From<&'static str> for ErrString {
    fn from(msg: &'static str) -> Self {
        if std::env::var("POLARS_PANIC_ON_ERR").is_ok() {
            panic!("{}", msg)
        } else {
            ErrString::Borrowed(msg)
        }
    }
}

impl From<String> for ErrString {
    fn from(msg: String) -> Self {
        if std::env::var("POLARS_PANIC_ON_ERR").is_ok() {
            panic!("{}", msg)
        } else {
            ErrString::Owned(msg)
        }
    }
}

impl Display for ErrString {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            ErrString::Owned(msg) => msg.as_str(),
            ErrString::Borrowed(msg) => msg,
        };
        write!(f, "{msg}")
    }
}

#[derive(Debug, ThisError)]
pub enum PolarsError {
    #[error(transparent)]
    ArrowError(Box<ArrowError>),
    #[error("Invalid operation {0}")]
    InvalidOperation(ErrString),
    #[error("Data types don't match: {0}")]
    SchemaMisMatch(ErrString),
    #[error("Not found: {0}")]
    NotFound(ErrString),
    #[error("Lengths don't match: {0}")]
    ShapeMisMatch(ErrString),
    #[error("{0}")]
    ComputeError(ErrString),
    #[error("Such empty...: {0}")]
    NoData(ErrString),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("DuplicateError: {0}")]
    Duplicate(ErrString),
}

impl From<ArrowError> for PolarsError {
    fn from(err: ArrowError) -> Self {
        Self::ArrowError(Box::new(err))
    }
}

impl From<anyhow::Error> for PolarsError {
    fn from(err: Error) -> Self {
        PolarsError::ComputeError(format!("{err:?}").into())
    }
}

impl From<polars_arrow::error::PolarsError> for PolarsError {
    fn from(err: polars_arrow::error::PolarsError) -> Self {
        PolarsError::ComputeError(format!("{err:?}").into())
    }
}

#[cfg(any(feature = "strings", feature = "temporal"))]
impl From<regex::Error> for PolarsError {
    fn from(err: regex::Error) -> Self {
        PolarsError::ComputeError(format!("regex error: {err:?}").into())
    }
}

pub type PolarsResult<T> = std::result::Result<T, PolarsError>;
pub use arrow::error::Error as ArrowError;
