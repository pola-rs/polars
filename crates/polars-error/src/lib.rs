pub mod constants;
mod warning;

use std::borrow::Cow;
use std::collections::TryReserveError;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::ops::Deref;
use std::{env, io};

pub use warning::*;

#[derive(Debug)]
pub struct ErrString(Cow<'static, str>);

impl<T> From<T> for ErrString
where
    T: Into<Cow<'static, str>>,
{
    fn from(msg: T) -> Self {
        if env::var("POLARS_PANIC_ON_ERR").as_deref().unwrap_or("") == "1" {
            panic!("{}", msg.into())
        } else {
            ErrString(msg.into())
        }
    }
}

impl AsRef<str> for ErrString {
    fn as_ref(&self) -> &str {
        &self.0
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
    #[error("{0}")]
    OutOfBounds(ErrString),
    #[error("field not found: {0}")]
    SchemaFieldNotFound(ErrString),
    #[error("data types don't match: {0}")]
    SchemaMismatch(ErrString),
    #[error("lengths don't match: {0}")]
    ShapeMismatch(ErrString),
    #[error("string caches don't match: {0}")]
    StringCacheMismatch(ErrString),
    #[error("field not found: {0}")]
    StructFieldNotFound(ErrString),
}

#[cfg(feature = "regex")]
impl From<regex::Error> for PolarsError {
    fn from(err: regex::Error) -> Self {
        PolarsError::ComputeError(format!("regex error: {err}").into())
    }
}

#[cfg(feature = "object_store")]
impl From<object_store::Error> for PolarsError {
    fn from(err: object_store::Error) -> Self {
        PolarsError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("object-store error: {err:?}"),
        ))
    }
}

#[cfg(feature = "parquet2")]
impl From<parquet2::error::Error> for PolarsError {
    fn from(err: parquet2::error::Error) -> Self {
        polars_err!(ComputeError: "parquet error: {err:?}")
    }
}

#[cfg(feature = "avro-schema")]
impl From<avro_schema::error::Error> for PolarsError {
    fn from(value: avro_schema::error::Error) -> Self {
        polars_err!(ComputeError: "avro-error: {}", value)
    }
}

#[cfg(feature = "parquet2")]
impl From<PolarsError> for parquet2::error::Error {
    fn from(value: PolarsError) -> Self {
        // catch all needed :(.
        parquet2::error::Error::OutOfSpec(format!("error: {value}"))
    }
}

impl From<simdutf8::basic::Utf8Error> for PolarsError {
    fn from(value: simdutf8::basic::Utf8Error) -> Self {
        polars_err!(ComputeError: "invalid utf8: {}", value)
    }
}
#[cfg(feature = "arrow-format")]
impl From<arrow_format::ipc::planus::Error> for PolarsError {
    fn from(err: arrow_format::ipc::planus::Error) -> Self {
        polars_err!(ComputeError: "parquet error: {err:?}")
    }
}

impl From<TryReserveError> for PolarsError {
    fn from(value: TryReserveError) -> Self {
        polars_err!(ComputeError: "OOM: {}", value)
    }
}

pub type PolarsResult<T> = Result<T, PolarsError>;

impl PolarsError {
    pub fn wrap_msg(&self, func: &dyn Fn(&str) -> String) -> Self {
        use PolarsError::*;
        match self {
            ColumnNotFound(msg) => ColumnNotFound(func(msg).into()),
            ComputeError(msg) => ComputeError(func(msg).into()),
            Duplicate(msg) => Duplicate(func(msg).into()),
            InvalidOperation(msg) => InvalidOperation(func(msg).into()),
            Io(err) => ComputeError(func(&format!("IO: {err}")).into()),
            NoData(msg) => NoData(func(msg).into()),
            OutOfBounds(msg) => OutOfBounds(func(msg).into()),
            SchemaFieldNotFound(msg) => SchemaFieldNotFound(func(msg).into()),
            SchemaMismatch(msg) => SchemaMismatch(func(msg).into()),
            ShapeMismatch(msg) => ShapeMismatch(func(msg).into()),
            StringCacheMismatch(msg) => StringCacheMismatch(func(msg).into()),
            StructFieldNotFound(msg) => StructFieldNotFound(func(msg).into()),
        }
    }
}

pub fn map_err<E: Error>(error: E) -> PolarsError {
    PolarsError::ComputeError(format!("{error}").into())
}

#[macro_export]
macro_rules! polars_err {
    ($variant:ident: $fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::__private::must_use(
            $crate::PolarsError::$variant(format!($fmt, $($arg),*).into())
        )
    };
    ($variant:ident: $err:expr $(,)?) => {
        $crate::__private::must_use(
            $crate::PolarsError::$variant($err.into())
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
    (un_impl = $op:ident) => {
        $crate::polars_err!(
            InvalidOperation: "{} operation is not implemented.", concat!("`", stringify!($op), "`")
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
    (oos = $($tt:tt)+) => {
        $crate::polars_err!(ComputeError: "out-of-spec: {}", $($tt)+)
    };
    (nyi = $($tt:tt)+) => {
        $crate::polars_err!(ComputeError: "not yet implemented: {}", format!($($tt)+) )
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
    (string_cache_mismatch) => {
        polars_err!(StringCacheMismatch: r#"
cannot compare categoricals coming from different sources, consider setting a global StringCache.

Help: if you're using Python, this may look something like:

    with pl.StringCache():
        # Initialize Categoricals.
        df1 = pl.DataFrame({'a': ['1', '2']}, schema={'a': pl.Categorical})
        df2 = pl.DataFrame({'a': ['1', '3']}, schema={'a': pl.Categorical})
        # Your operations go here.
        pl.concat([df1, df2])

Alternatively, if the performance cost is acceptable, you could just set:

    import polars as pl
    pl.enable_string_cache()

on startup."#.trim_start())
    };
    (duplicate = $name:expr) => {
        polars_err!(Duplicate: "column with name '{}' has more than one occurrences", $name)
    };
    (oob = $idx:expr, $len:expr) => {
        polars_err!(OutOfBounds: "index {} is out of bounds for sequence of length {}", $idx, $len)
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
            ComputeError: "could not find an appropriate format to parse {}s, please define a format",
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
