use std::fmt::{Display, Formatter};
use std::ops::Deref;

use anyhow::Error;
use thiserror::Error as ThisError;

#[derive(Debug)]
pub enum ErrString {
    Owned(String),
    Borrowed(&'static str),
}

impl Deref for ErrString {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        match self {
            ErrString::Owned(s) => s,
            ErrString::Borrowed(s) => s,
        }
    }
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
    #[error("Not found: {0}")]
    ColumnNotFound(ErrString),
    #[error("{0}")]
    ComputeError(ErrString),
    #[error("DuplicateError: {0}")]
    Duplicate(ErrString),
    #[error("Invalid operation {0}")]
    InvalidOperation(ErrString),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("Such empty...: {0}")]
    NoData(ErrString),
    #[error("Not found: {0}")]
    SchemaFieldNotFound(ErrString),
    #[error("Data types don't match: {0}")]
    SchemaMisMatch(ErrString),
    #[error("Lengths don't match: {0}")]
    ShapeMisMatch(ErrString),
    #[error("Not found: {0}")]
    StructFieldNotFound(ErrString),
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

impl PolarsError {
    pub fn wrap_msg(&self, func: &dyn Fn(&str) -> String) -> Self {
        use PolarsError::*;
        match self {
            ArrowError(err) => ComputeError(func(&format!("ArrowError: {err}")).into()),
            ColumnNotFound(msg) => ColumnNotFound(func(msg).into()),
            ComputeError(msg) => ComputeError(func(msg).into()),
            Duplicate(msg) => Duplicate(func(msg).into()),
            InvalidOperation(msg) => InvalidOperation(func(msg).into()),
            Io(err) => ComputeError(func(&format!("IO: {err}")).into()),
            NoData(msg) => NoData(func(msg).into()),
            SchemaFieldNotFound(msg) => SchemaFieldNotFound(func(msg).into()),
            SchemaMisMatch(msg) => SchemaMisMatch(func(msg).into()),
            ShapeMisMatch(msg) => ShapeMisMatch(func(msg).into()),
            StructFieldNotFound(msg) => StructFieldNotFound(func(msg).into()),
        }
    }
}
