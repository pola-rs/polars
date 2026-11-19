pub mod constants;
mod warning;

use std::borrow::Cow;
use std::collections::TryReserveError;
use std::convert::Infallible;
use std::error::Error;
use std::fmt::{self, Display, Formatter, Write};
use std::ops::Deref;
use std::sync::{Arc, LazyLock};
use std::{env, io};
pub mod abort;
mod sql_location;

pub use sql_location::*;
pub use warning::*;

#[cfg(feature = "python")]
mod python;

enum ErrorStrategy {
    Panic,
    WithBacktrace,
    Normal,
}

static ERROR_STRATEGY: LazyLock<ErrorStrategy> = LazyLock::new(|| {
    if env::var("POLARS_PANIC_ON_ERR").as_deref() == Ok("1") {
        ErrorStrategy::Panic
    } else if env::var("POLARS_BACKTRACE_IN_ERR").as_deref() == Ok("1") {
        ErrorStrategy::WithBacktrace
    } else {
        ErrorStrategy::Normal
    }
});

#[derive(Debug, Clone)]
pub struct ErrString(Cow<'static, str>);

impl ErrString {
    pub const fn new_static(s: &'static str) -> Self {
        Self(Cow::Borrowed(s))
    }
}

impl<T> From<T> for ErrString
where
    T: Into<Cow<'static, str>>,
{
    #[track_caller]
    fn from(msg: T) -> Self {
        match &*ERROR_STRATEGY {
            ErrorStrategy::Panic => panic!("{}", msg.into()),
            ErrorStrategy::WithBacktrace => ErrString(Cow::Owned(format!(
                "{}\n\nRust backtrace:\n{}",
                msg.into(),
                std::backtrace::Backtrace::force_capture()
            ))),
            ErrorStrategy::Normal => ErrString(msg.into()),
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

#[derive(Debug, Clone)]
pub enum PolarsError {
    AssertionError(ErrString),
    ColumnNotFound(ErrString),
    ComputeError(ErrString),
    Duplicate(ErrString),
    InvalidOperation(ErrString),
    IO {
        error: Arc<io::Error>,
        msg: Option<ErrString>,
    },
    NoData(ErrString),
    OutOfBounds(ErrString),
    SchemaFieldNotFound(ErrString),
    SchemaMismatch(ErrString),
    ShapeMismatch(ErrString),
    SQLInterface(ErrString),
    SQLSyntax(ErrString),
    StringCacheMismatch(ErrString),
    StructFieldNotFound(ErrString),
    Context {
        error: Box<PolarsError>,
        msg: ErrString,
    },
    ExprContext {
        error: Box<PolarsError>,
        expr: ErrString,
    },
    /// An error with data attached, such as the plan it happened in.
    Attached {
        error: Box<PolarsError>,
        attachment: Box<ErrorAttachment>,
    },
    #[cfg(feature = "python")]
    Python {
        error: python::PyErrWrap,
    },
}

/// Data attached to an error.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorAttachment {
    /// The plan as far as it was resolved when the error happened.
    ResolvedPlan {
        node: String,
        plan: String,
    },
    SqlLocation(SqlLocation),
    Hint(Cow<'static, str>),
}

impl ErrorAttachment {
    fn write_after_message(&self, out: &mut String) {
        match self {
            ErrorAttachment::ResolvedPlan { node, plan } => write!(
                out,
                "\n\nResolved plan until failure:\n\n\t---> FAILED HERE RESOLVING {node} <---\n{plan}"
            )
            .unwrap(),
            ErrorAttachment::SqlLocation(location) => location.write_after_message(out),
            ErrorAttachment::Hint(hint) => write!(out, "\n\nHint: {hint}").unwrap(),
        }
    }
}

/// Writes `attachments` (innermost first): the innermost SQL location, the hints, the rest.
fn write_attachments<'a>(
    out: &mut String,
    attachments: impl Iterator<Item = &'a ErrorAttachment> + Clone,
) {
    let is_location = |a: &ErrorAttachment| matches!(a, ErrorAttachment::SqlLocation(_));
    let is_hint = |a: &ErrorAttachment| matches!(a, ErrorAttachment::Hint(_));
    if let Some(location) = attachments.clone().find(|a| is_location(a)) {
        location.write_after_message(out);
    }
    let mut hints = Vec::new();
    for hint in attachments.clone().filter(|a| is_hint(a)) {
        if !hints.contains(&hint) {
            hint.write_after_message(out);
            hints.push(hint);
        }
    }
    for attachment in attachments.filter(|a| !is_location(a) && !is_hint(a)) {
        attachment.write_after_message(out);
    }
}

impl Error for PolarsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            PolarsError::AssertionError(_)
            | PolarsError::ColumnNotFound(_)
            | PolarsError::ComputeError(_)
            | PolarsError::Duplicate(_)
            | PolarsError::InvalidOperation(_)
            | PolarsError::NoData(_)
            | PolarsError::OutOfBounds(_)
            | PolarsError::SchemaFieldNotFound(_)
            | PolarsError::SchemaMismatch(_)
            | PolarsError::ShapeMismatch(_)
            | PolarsError::SQLInterface(_)
            | PolarsError::SQLSyntax(_)
            | PolarsError::StringCacheMismatch(_)
            | PolarsError::StructFieldNotFound(_) => None,
            PolarsError::IO { error, .. } => Some(error.as_ref()),
            PolarsError::Context { error, .. } => Some(error.as_ref()),
            PolarsError::ExprContext { error, .. } => Some(error.as_ref()),
            PolarsError::Attached { error, .. } => Some(error.as_ref()),
            #[cfg(feature = "python")]
            PolarsError::Python { error } => error.deref().source(),
        }
    }
}

impl Display for PolarsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use PolarsError::*;
        match self.clone().context_trace() {
            ComputeError(msg)
            | InvalidOperation(msg)
            | OutOfBounds(msg)
            | SchemaMismatch(msg)
            | SQLInterface(msg)
            | SQLSyntax(msg) => write!(f, "{msg}"),

            AssertionError(msg) => write!(f, "assertion failed: {msg}"),
            ColumnNotFound(msg) => write!(f, "not found: {msg}"),
            Duplicate(msg) => write!(f, "duplicate: {msg}"),
            IO { error, msg } => match msg {
                Some(m) => write!(f, "{m}"),
                None => write!(f, "{error}"),
            },
            NoData(msg) => write!(f, "no data: {msg}"),
            SchemaFieldNotFound(msg) => write!(f, "field not found: {msg}"),
            ShapeMismatch(msg) => write!(f, "lengths don't match: {msg}"),
            StringCacheMismatch(msg) => write!(f, "string caches don't match: {msg}"),
            StructFieldNotFound(msg) => write!(f, "field not found: {msg}"),
            Context { error, msg } => write!(f, "{error}: {msg}"),
            ExprContext { error, expr: _ } => write!(f, "{error}"),
            Attached { error, .. } => write!(f, "{error}"),
            #[cfg(feature = "python")]
            Python { error } => write!(f, "python: {error}"),
        }
    }
}

impl From<io::Error> for PolarsError {
    fn from(mut value: io::Error) -> Self {
        if let Some(polars_err) = value
            .get_mut()
            .and_then(|e| e.downcast_mut::<PolarsError>())
        {
            std::mem::replace(
                polars_err,
                PolarsError::ComputeError(ErrString::new_static("")),
            )
        } else {
            PolarsError::IO {
                error: Arc::new(value),
                msg: None,
            }
        }
    }
}

impl From<PolarsError> for io::Error {
    fn from(value: PolarsError) -> Self {
        io::Error::other(value)
    }
}

#[cfg(feature = "regex")]
impl From<regex::Error> for PolarsError {
    fn from(err: regex::Error) -> Self {
        PolarsError::ComputeError(format!("regex error: {err}").into())
    }
}

#[cfg(feature = "avro-schema")]
impl From<avro_schema::error::Error> for PolarsError {
    fn from(value: avro_schema::error::Error) -> Self {
        polars_err!(ComputeError: "avro-error: {}", value)
    }
}

impl From<simdutf8::basic::Utf8Error> for PolarsError {
    fn from(value: simdutf8::basic::Utf8Error) -> Self {
        polars_err!(ComputeError: "invalid utf8: {}", value)
    }
}
#[cfg(feature = "polars-arrow-format")]
impl From<polars_arrow_format::ipc::planus::Error> for PolarsError {
    fn from(err: polars_arrow_format::ipc::planus::Error) -> Self {
        polars_err!(ComputeError: "parquet error: {err:?}")
    }
}

impl From<TryReserveError> for PolarsError {
    fn from(value: TryReserveError) -> Self {
        polars_err!(ComputeError: "OOM: {}", value)
    }
}

impl From<Infallible> for PolarsError {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

pub type PolarsResult<T> = Result<T, PolarsError>;

impl PolarsError {
    pub fn context_trace(self) -> Self {
        use PolarsError::*;
        if !matches!(self, Context { .. } | ExprContext { .. } | Attached { .. }) {
            return self;
        }

        let mut context_msgs = Vec::new();
        let mut context_exprs = Vec::new();
        let mut attachments = Vec::new();
        let mut error = self;
        loop {
            match error {
                Context { error: e, msg } => {
                    context_msgs.push(msg);
                    error = *e;
                },

                ExprContext { error: e, expr } => {
                    context_exprs.push(expr);
                    error = *e;
                },

                Attached {
                    error: e,
                    attachment,
                } => {
                    attachments.push(attachment);
                    error = *e;
                },

                e => {
                    error = e;
                    break;
                },
            }
        }

        error.wrap_msg(|msg| {
            let mut out = msg.to_string();
            write_attachments(&mut out, attachments.iter().rev().map(|a| a.as_ref()));
            if !context_exprs.is_empty() {
                let first = context_exprs.first().unwrap();
                let last = context_exprs.last().unwrap();
                writeln!(
                    &mut out,
                    "\n\nThis error occurred in the following expression:"
                )
                .unwrap();
                writeln!(&mut out, "\t{last}").unwrap();
                if first.0 != last.0 {
                    writeln!(&mut out, "while evaluating this larger expression:").unwrap();
                    writeln!(&mut out, "\t{first}").unwrap();
                }
            }

            if !context_msgs.is_empty() {
                writeln!(
                    &mut out,
                    "\n\nThis error occurred with the following context stack:"
                )
                .unwrap();
                for (i, m) in context_msgs.into_iter().rev().enumerate() {
                    writeln!(&mut out, "\t[{}] {}", i + 1, m).unwrap();
                }
            }
            out
        })
    }

    pub fn wrap_msg<F: FnOnce(&str) -> String>(&self, func: F) -> Self {
        use PolarsError::*;
        match self {
            AssertionError(msg) => AssertionError(func(msg).into()),
            ColumnNotFound(msg) => ColumnNotFound(func(msg).into()),
            ComputeError(msg) => ComputeError(func(msg).into()),
            Duplicate(msg) => Duplicate(func(msg).into()),
            InvalidOperation(msg) => InvalidOperation(func(msg).into()),
            IO { error, msg } => {
                let msg = match msg {
                    Some(msg) => func(msg),
                    None => func(&format!("{error}")),
                };
                IO {
                    error: error.clone(),
                    msg: Some(msg.into()),
                }
            },
            NoData(msg) => NoData(func(msg).into()),
            OutOfBounds(msg) => OutOfBounds(func(msg).into()),
            SchemaFieldNotFound(msg) => SchemaFieldNotFound(func(msg).into()),
            SchemaMismatch(msg) => SchemaMismatch(func(msg).into()),
            ShapeMismatch(msg) => ShapeMismatch(func(msg).into()),
            StringCacheMismatch(msg) => StringCacheMismatch(func(msg).into()),
            StructFieldNotFound(msg) => StructFieldNotFound(func(msg).into()),
            SQLInterface(msg) => SQLInterface(func(msg).into()),
            SQLSyntax(msg) => SQLSyntax(func(msg).into()),
            Context {
                error,
                msg: context_msg,
            } => Context {
                error: Box::new(error.wrap_msg(func)),
                msg: context_msg.clone(),
            },
            ExprContext { error, expr } => ExprContext {
                error: Box::new(error.wrap_msg(func)),
                expr: expr.clone(),
            },
            Attached { error, attachment } => Attached {
                error: Box::new(error.wrap_msg(func)),
                attachment: attachment.clone(),
            },
            #[cfg(feature = "python")]
            Python { error } => pyo3::Python::attach(|py| {
                use pyo3::types::{PyAnyMethods, PyStringMethods};
                use pyo3::{IntoPyObject, PyErr};

                let value = error.value(py);

                let msg = if let Ok(s) = value.str() {
                    func(&s.to_string_lossy())
                } else {
                    func("<exception str() failed>")
                };

                let cls = value.get_type();

                let out = PyErr::from_type(cls, (msg,));

                let out = if let Ok(out_with_traceback) = (|| {
                    out.clone_ref(py)
                        .into_pyobject(py)?
                        .getattr("with_traceback")
                        .unwrap()
                        .call1((value.getattr("__traceback__").unwrap(),))
                })() {
                    PyErr::from_value(out_with_traceback)
                } else {
                    out
                };

                Python {
                    error: python::PyErrWrap(out),
                }
            }),
        }
    }

    pub fn context(self, msg: ErrString) -> Self {
        PolarsError::Context {
            error: Box::new(self),
            msg,
        }
    }

    /// Removes the outer `Context` frames; attachments are kept.
    pub fn remove_context(self) -> Self {
        match self {
            Self::Context { error, .. } => error.remove_context(),
            Self::Attached { error, attachment } => {
                error.remove_context().with_attachment(*attachment)
            },
            error => error,
        }
    }

    pub fn with_expr_context(self, expr: ErrString) -> Self {
        PolarsError::ExprContext {
            error: Box::new(self),
            expr,
        }
    }

    pub fn with_attachment(self, attachment: ErrorAttachment) -> Self {
        PolarsError::Attached {
            error: Box::new(self),
            attachment: Box::new(attachment),
        }
    }

    pub fn with_hint(self, hint: impl Into<Cow<'static, str>>) -> Self {
        self.with_attachment(ErrorAttachment::Hint(hint.into()))
    }

    /// The hints, innermost first, without duplicates.
    pub fn hints(&self) -> Vec<&str> {
        let mut hints = Vec::new();
        for attachment in self.attachments().into_iter().rev() {
            if let ErrorAttachment::Hint(hint) = attachment
                && !hints.contains(&hint.as_ref())
            {
                hints.push(hint.as_ref());
            }
        }
        hints
    }

    /// The innermost SQL location.
    pub fn sql_location(&self) -> Option<&SqlLocation> {
        self.attachments()
            .into_iter()
            .rev()
            .find_map(|attachment| match attachment {
                ErrorAttachment::SqlLocation(location) => Some(location),
                _ => None,
            })
    }

    /// The attachments, outermost first.
    pub fn attachments(&self) -> Vec<&ErrorAttachment> {
        use PolarsError::*;
        let mut attachments = Vec::new();
        let mut error = self;
        loop {
            match error {
                Attached {
                    error: e,
                    attachment,
                } => {
                    attachments.push(attachment.as_ref());
                    error = e;
                },
                Context { error: e, .. } | ExprContext { error: e, .. } => error = e,
                _ => return attachments,
            }
        }
    }

    /// Keeps the attachments for which `keep` returns true.
    pub fn retain_attachments<F: FnMut(&ErrorAttachment) -> bool>(self, mut keep: F) -> Self {
        fn retain<F: FnMut(&ErrorAttachment) -> bool>(
            error: PolarsError,
            keep: &mut F,
        ) -> PolarsError {
            use PolarsError::*;
            match error {
                Attached { error, attachment } => {
                    let kept = keep(&attachment);
                    let error = retain(*error, keep);
                    match kept {
                        true => error.with_attachment(*attachment),
                        false => error,
                    }
                },
                Context { error, msg } => retain(*error, keep).context(msg),
                ExprContext { error, expr } => retain(*error, keep).with_expr_context(expr),
                error => error,
            }
        }
        retain(self, &mut keep)
    }

    /// Writes the attachments into the innermost message.
    pub fn inline_attachments(self) -> Self {
        let attachments: Vec<ErrorAttachment> = self.attachments().into_iter().cloned().collect();
        if attachments.is_empty() {
            return self;
        }
        self.retain_attachments(|_| false).wrap_msg(|msg| {
            let mut out = msg.to_string();
            write_attachments(&mut out, attachments.iter().rev());
            out
        })
    }
}

pub trait PolarsContext<T> {
    fn context(self, ctx: &'static str) -> PolarsResult<T>;

    fn with_context<F>(self, f: F) -> PolarsResult<T>
    where
        F: FnOnce() -> String;
}

impl<T> PolarsContext<T> for PolarsResult<T> {
    fn context(self, ctx: &'static str) -> PolarsResult<T> {
        self.map_err(|e| e.context(ErrString::new_static(ctx)))
    }

    fn with_context<F>(self, f: F) -> PolarsResult<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| e.context(f().into()))
    }
}

pub fn map_err<E: Error>(error: E) -> PolarsError {
    PolarsError::ComputeError(format!("{error}").into())
}

#[macro_export]
macro_rules! polars_err {
    ($variant:ident: format!($_:tt) $(, _:tt)* $(,)?) => {
        const { panic!("remove unnecessary format! from polars_(bail|err)! macro") }
    };
    ($variant:ident: $fmt:literal $(,)?) => {{
        if const { $crate::__private::has_brace($fmt) } {
            $crate::__private::must_use(
                $crate::PolarsError::$variant(format!($fmt).into())
            )
        } else {
            const {
                $crate::__private::must_use(
                    $crate::PolarsError::$variant($crate::ErrString::new_static($fmt))
                )
            }
        }
    }};
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
    (op = $op:expr, $arg:expr, hint = $hint:literal) => {
        $crate::polars_err!(
            InvalidOperation: "{} operation not supported for dtype `{}`\n\nHint: {}", $op, $arg, $hint
        )
    };
    (op = $op:expr, $lhs:expr, $rhs:expr) => {
        $crate::polars_err!(
            InvalidOperation: "{} operation not supported for dtypes `{}` and `{}`", $op, $lhs, $rhs
        )
    };
    (op = $op:expr, $arg1:expr, $arg2:expr, $arg3:expr) => {
        $crate::polars_err!(
            InvalidOperation: "{} operation not supported for dtypes `{}`, `{}` and `{}`", $op, $arg1, $arg2, $arg3
        )
    };
    (opidx = $op:expr, idx = $idx:expr, $arg:expr) => {
        $crate::polars_err!(
            InvalidOperation: "`{}` operation not supported for dtype `{}` as argument {}", $op, $arg, $idx
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
    (bigidx, ctx = $ctx:expr, size = $size:expr) => {
        $crate::polars_err!(ComputeError: "\
{} produces {} rows which is more than maximum allowed pow(2, 32)-1 rows; \
consider compiling with bigidx feature (pip install polars[rt64])",
            $ctx,
            $size,
        )
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
    (not_in_enum,value=$value:expr,categories=$categories:expr) =>{
        polars_err!(ComputeError: "value '{}' is not present in Enum: {:?}",$value,$categories)
    };
    (string_cache_mismatch) => {
        polars_err!(StringCacheMismatch: r#"
cannot compare categoricals coming from different sources, consider setting a global StringCache.

Help: if you're using Python, this may look something like:

    with pl.StringCache():
        df1 = pl.DataFrame({'a': ['1', '2']}, schema={'a': pl.Categorical})
        df2 = pl.DataFrame({'a': ['1', '3']}, schema={'a': pl.Categorical})
        pl.concat([df1, df2])

Alternatively, if the performance cost is acceptable, you could just set:

    import polars as pl
    pl.enable_string_cache()

on startup."#.trim_start())
    };
    (duplicate = $name:expr) => {
        $crate::polars_err!(Duplicate: "column with name '{}' has more than one occurrence", $name)
    };
    (duplicate_field = $name:expr) => {
        $crate::polars_err!(Duplicate: "multiple fields with name '{}' found", $name)
    };
    (col_not_found = $name:expr) => {
        $crate::polars_err!(ColumnNotFound: "{:?} not found", $name)
    };
    (mismatch, col=$name:expr, expected=$expected:expr, found=$found:expr) => {
        $crate::polars_err!(
            SchemaMismatch: "data type mismatch for column {}: expected: {}, found: {}",
            $name,
            $expected,
            $found,
        )
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
    (length_mismatch = $operation:expr, $lhs:expr, $rhs:expr) => {
        $crate::polars_err!(
            ShapeMismatch: "arguments for `{}` have different lengths ({} != {})",
            $operation, $lhs, $rhs
        )
    };
    (length_mismatch = $operation:expr, $lhs:expr, $rhs:expr, argument = $argument:expr, argument_idx = $argument_idx:expr) => {
        $crate::polars_err!(
            ShapeMismatch: "argument {} called '{}' for `{}` have different lengths ({} != {})",
            $argument_idx, $argument, $operation, $lhs, $rhs
        )
    };
    (invalid_element_use) => {
        $crate::polars_err!(InvalidOperation: "`element` is not allowed in this context")
    };
    (invalid_field_use) => {
        $crate::polars_err!(InvalidOperation: "`field` is not allowed in this context")
    };
    (non_utf8_path) => {
        $crate::polars_err!(ComputeError: "encountered non UTF-8 path characters")
    };
    (assertion_error = $objects:expr, $detail:expr, $lhs:expr, $rhs:expr) => {
        $crate::polars_err!(
            AssertionError: "{} are different ({})\n[left]: {}\n[right]: {}",
            $objects, $detail, $lhs, $rhs
        )
    };
    (to_datetime_tz_mismatch) => {
        $crate::polars_err!(
            ComputeError: "`strptime` / `to_datetime` was called with no format and no time zone, but a time zone is part of the data.\n\nThis was previously allowed but led to unpredictable and erroneous results. Give a format string, set a time zone or perform the operation eagerly on a Series instead of on an Expr."
        )
    };
    (item_agg_count_not_one = $n:expr, allow_empty = $allow_empty:expr) => {
        if $n == 0 && !$allow_empty {
            polars_err!(ComputeError:
                "aggregation 'item' expected a single value, got none"
            )
        } else if $n > 100 {
            if $allow_empty {
                polars_err!(ComputeError: "aggregation 'item' expected no or a single value, got 100+ values")
            } else {
                polars_err!(ComputeError: "aggregation 'item' expected a single value, got 100+ values")
            }
        } else if $n > 1 {
            if $allow_empty {
                polars_err!(ComputeError:
                    "aggregation 'item' expected no or a single value, got {} values", $n
                )
            } else {
                polars_err!(ComputeError:
                    "aggregation 'item' expected a single value, got {} values", $n
                )
            }
        } else {
            unreachable!()
        }
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
            $crate::polars_bail!($($tt)+);
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
    ($($feature:literal);*, $content:expr) => {{
        #[cfg(all($(feature = $feature),*))]
        {
            $content
        }
        #[cfg(not(all($(feature = $feature),*)))]
        {
            panic!("activate '{}' feature", concat!($($feature, ", "),*))
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
    pub const fn must_use(error: crate::PolarsError) -> crate::PolarsError {
        error
    }

    pub const fn has_brace(s: &str) -> bool {
        let bytes = s.as_bytes();
        let mut i: usize = 0;

        while i < bytes.len() {
            if bytes[i] == b'{' || bytes[i] == b'}' {
                return true;
            }

            i += 1;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use crate::{ErrString, ErrorAttachment, PolarsError, SqlLocation};

    #[test]
    fn test_attachments_are_written_location_then_hints_then_plan() {
        let error = polars_err!(SQLInterface: "bad")
            .with_attachment(ErrorAttachment::SqlLocation(SqlLocation::statement(
                "SELECT 1",
            )))
            .with_attachment(ErrorAttachment::ResolvedPlan {
                node: "'select'".to_string(),
                plan: "DF []".to_string(),
            })
            .with_hint("try this")
            .with_hint("try this");
        let expected = "bad\n\nIn the SQL statement: SELECT 1\n\nHint: try this\n\nResolved plan until failure:\n\n\t---> FAILED HERE RESOLVING 'select' <---\nDF []";
        assert_eq!(error.hints(), vec!["try this"]);
        assert_eq!(error.to_string(), expected);
        assert_eq!(error.inline_attachments().to_string(), expected);
    }

    #[test]
    fn test_polars_error_roundtrips_through_std_io_error() {
        use PolarsError::ComputeError;

        let error_magic = "err_magic_3";
        let error = ComputeError(ErrString::new_static(error_magic));

        let io_error: std::io::Error = error.into();
        let error: PolarsError = io_error.into();

        match error {
            ComputeError(s) if &*s == error_magic => {},
            e => panic!("error type mismatch: {e}"),
        };
    }

    #[test]
    fn test_polars_error_format_str() {
        use PolarsError::ComputeError;

        let a = "A";

        match polars_err!(ComputeError: "{a}") {
            ComputeError(out) if &*out == a => {},
            e => panic!("{e}"),
        }

        match polars_err!(ComputeError: a) {
            ComputeError(out) if &*out == a => {},
            e => panic!("{e}"),
        }

        match polars_err!(ComputeError: "{{") {
            ComputeError(out) if &*out == "{" => {},
            e => panic!("{e}"),
        }

        match polars_err!(ComputeError: "}}") {
            ComputeError(out) if &*out == "}" => {},
            e => panic!("{e}"),
        }
    }
}
