use std::borrow::Cow;
use std::error::Error;
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
    #[error("not found: {0}")]
    ColumnNotFound(ErrString),
    #[error("{0}")]
    ComputeError(ErrString),
    #[error("duplicate: {0}")]
    Duplicate(ErrString),
    #[error("invalid operation: {0}")]
    InvalidOperation(ErrString),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("no data: {0}")]
    NoData(ErrString),
    #[error("field not found: {0}")]
    SchemaFieldNotFound(ErrString),
    #[error("data types don't match: {0}")]
    SchemaMismatch(ErrString),
    #[error("lengths don't match: {0}")]
    ShapeMismatch(ErrString),
    #[error("field not found: {0}")]
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
        PolarsError::ComputeError(format!("regex error: {err}").into())
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
}

pub fn map_err<E: Error>(error: E) -> PolarsError {
    PolarsError::ComputeError(format!("{error}").into())
}

#[macro_export]
macro_rules! polars_err {
    ($variant:ident: $err:expr $(,)?) => {
        $crate::__private::must_use(
            $crate::PolarsError::$variant($err.into())
        )
    };
    ($variant:ident: $fmt:literal, $($arg:tt)+) => {
        $crate::__private::must_use(
            $crate::PolarsError::$variant(format!($fmt, $($arg)+).into())
        )
    };
    (expr = $expr:expr, $variant:ident: $err:expr $(,)?) => {
        $crate::__private::must_use(
            $crate::PolarsError::$variant(
                format!("{}\n\nError originated in expression: '{:?}'", $err, $expr).into()
            )
        )
    };
    (expr = $expr:expr, $variant:ident: $fmt:literal, $($arg:tt)+) => {
        polars_err!(expr = $expr, $variant: format!($fmt, $($arg)+))
    };
    (op = $op:expr, got = $arg:expr, expected = $expected:expr) => {
        $crate::polars_err!(
            InvalidOperation: "{} operation not supported for dtype `{}` (expected: {})",
            $op, $arg, $expected
        )
    };
    (opq = $op:ident, got = $arg:expr, expected = $expected:expr) => {
        $crate::polars_err!(
            op = concat!("`", stringify!($op), "`"), got = $arg, expected = $expected
        )
    };
    (op = $op:expr, $arg:expr) => {
        $crate::polars_err!(
            InvalidOperation: "{} operation not supported for dtype `{}`", $op, $arg
        )
    };
    (op = $op:expr, $lhs:expr, $rhs:expr) => {
        $crate::polars_err!(
            InvalidOperation: "{} operation not supported for dtypes `{}` and `{}`", $op, $lhs, $rhs
        )
    };
    (opq = $op:ident, $arg:expr) => {
        $crate::polars_err!(op = concat!("`", stringify!($op), "`"), $arg)
    };
    (opq = $op:ident, $lhs:expr, $rhs:expr) => {
        $crate::polars_err!(op = stringify!($op), $lhs, $rhs)
    };
    (append) => {
        polars_err!(SchemaMismatch: "cannot append series, data types don't match")
    };
    (extend) => {
        polars_err!(SchemaMismatch: "cannot extend series, data types don't match")
    };
    (unpack) => {
        polars_err!(SchemaMismatch: "cannot unpack series, data types don't match")
    };
    (duplicate = $name:expr) => {
        polars_err!(Duplicate: "column with name '{}' has more than one occurrences", $name)
    };
    (oob = $idx:expr, $len:expr) => {
        polars_err!(ComputeError: "index {} is out of bounds for sequence of size {}", $idx, $len)
    };
    (agg_len = $agg_len:expr, $groups_len:expr) => {
        polars_err!(
            ComputeError:
            "returned aggregation is of different length: {} than the groups length: {}",
            $agg_len, $groups_len
        )
    };
    (parse_fmt_idk = $dtype:expr) => {
        polars_err!(
            ComputeError: "could not find an appropriate format to parse {}s, please define a fmt",
            $dtype,
        )
    };
}

#[macro_export]
macro_rules! polars_bail {
    ($($tt:tt)+) => {
        return Err($crate::polars_err!($($tt)+))
    };
}

#[macro_export]
macro_rules! polars_ensure {
    ($cond:expr, $($tt:tt)+) => {
        if !$cond {
            polars_bail!($($tt)+);
        }
    };
}

#[inline]
#[cold]
#[must_use]
pub fn to_compute_err(err: impl Display) -> PolarsError {
    PolarsError::ComputeError(err.to_string().into())
}

#[macro_export]
macro_rules! feature_gated {
    ($feature:expr, $content:expr) => {{
        #[cfg(feature = $feature)]
        {
            $content
        }
        #[cfg(not(feature = $feature))]
        {
            panic!("activate '{}' feature", $feature)
        }
    }};
}

// Not public, referenced by macros only.
#[doc(hidden)]
pub mod __private {
    #[doc(hidden)]
    #[inline]
    #[cold]
    #[must_use]
    pub fn must_use(error: crate::PolarsError) -> crate::PolarsError {
        error
    }
}
