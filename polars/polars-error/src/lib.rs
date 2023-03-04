use std::borrow::Cow;
use std::fmt::{self, Display, Formatter};
use std::ops::Deref;
use std::{env, io};

#[derive(Debug)]
pub struct ErrString(Cow<'static, str>);

impl<T> From<T> for ErrString
where
    T: Into<Cow<'static, str>>,
{
    fn from(msg: T) -> Self {
        if env::var("POLARS_PANIC_ON_ERR").is_ok() {
            panic!("{}", msg.into())
        } else {
            ErrString(msg.into())
        }
    }
}

impl Deref for ErrString {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for ErrString {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, thiserror::Error)]
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
    Io(#[from] io::Error),
    #[error("Such empty...: {0}")]
    NoData(ErrString),
    #[error("Not found: {0}")]
    SchemaFieldNotFound(ErrString),
    #[error("Data types don't match: {0}")]
    SchemaMismatch(ErrString),
    #[error("Lengths don't match: {0}")]
    ShapeMismatch(ErrString),
    #[error("Not found: {0}")]
    StructFieldNotFound(ErrString),
}

impl From<ArrowError> for PolarsError {
    fn from(err: ArrowError) -> Self {
        Self::ArrowError(Box::new(err))
    }
}

#[cfg(feature = "regex")]
impl From<regex::Error> for PolarsError {
    fn from(err: regex::Error) -> Self {
        PolarsError::ComputeError(format!("regex error: {err:?}").into())
    }
}

pub type PolarsResult<T> = Result<T, PolarsError>;

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
            SchemaMismatch(msg) => SchemaMismatch(func(msg).into()),
            ShapeMismatch(msg) => ShapeMismatch(func(msg).into()),
            StructFieldNotFound(msg) => StructFieldNotFound(func(msg).into()),
        }
    }

    pub fn from_any(err: impl Display) -> Self {
        Self::ComputeError(err.to_string().into())
    }
}
