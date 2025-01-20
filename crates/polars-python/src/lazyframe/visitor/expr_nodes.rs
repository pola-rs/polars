#[cfg(feature = "iejoin")]
use polars::prelude::InequalityOperator;
use polars::series::ops::NullBehavior;
use polars_core::series::IsSorted;
#[cfg(feature = "string_normalize")]
use polars_ops::chunked_array::UnicodeForm;
use polars_ops::series::InterpolationMethod;
#[cfg(feature = "search_sorted")]
use polars_ops::series::SearchSortedSide;
use polars_plan::dsl::function_expr::rolling::RollingFunction;
use polars_plan::dsl::function_expr::rolling_by::RollingFunctionBy;
use polars_plan::dsl::{BooleanFunction, StringFunction, TemporalFunction};
use polars_plan::prelude::{
    AExpr, FunctionExpr, GroupbyOptions, IRAggExpr, LiteralValue, Operator, PowFunction,
    WindowMapping, WindowType,
};
use polars_time::prelude::RollingGroupOptions;
use polars_time::{Duration, DynamicGroupOptions};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::IntoPyObjectExt;

use crate::series::PySeries;
use crate::Wrap;

#[pyclass]
pub struct Alias {
    #[pyo3(get)]
    expr: usize,
    #[pyo3(get)]
    name: PyObject,
}

#[pyclass]
pub struct Column {
    #[pyo3(get)]
    name: PyObject,
}

#[pyclass]
pub struct Literal {
    #[pyo3(get)]
    value: PyObject,
    #[pyo3(get)]
    dtype: PyObject,
}

#[pyclass(name = "Operator", eq)]
#[derive(Copy, Clone, PartialEq)]
pub enum PyOperator {
    Eq,
    EqValidity,
    NotEq,
    NotEqValidity,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Plus,
    Minus,
    Multiply,
    Divide,
    TrueDivide,
    FloorDivide,
    Modulus,
    And,
    Or,
    Xor,
    LogicalAnd,
    LogicalOr,
}

#[pymethods]
impl PyOperator {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl<'py> IntoPyObject<'py> for Wrap<Operator> {
    type Target = PyOperator;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self.0 {
            Operator::Eq => PyOperator::Eq,
            Operator::EqValidity => PyOperator::EqValidity,
            Operator::NotEq => PyOperator::NotEq,
            Operator::NotEqValidity => PyOperator::NotEqValidity,
            Operator::Lt => PyOperator::Lt,
            Operator::LtEq => PyOperator::LtEq,
            Operator::Gt => PyOperator::Gt,
            Operator::GtEq => PyOperator::GtEq,
            Operator::Plus => PyOperator::Plus,
            Operator::Minus => PyOperator::Minus,
            Operator::Multiply => PyOperator::Multiply,
            Operator::Divide => PyOperator::Divide,
            Operator::TrueDivide => PyOperator::TrueDivide,
            Operator::FloorDivide => PyOperator::FloorDivide,
            Operator::Modulus => PyOperator::Modulus,
            Operator::And => PyOperator::And,
            Operator::Or => PyOperator::Or,
            Operator::Xor => PyOperator::Xor,
            Operator::LogicalAnd => PyOperator::LogicalAnd,
            Operator::LogicalOr => PyOperator::LogicalOr,
        }
        .into_pyobject(py)
    }
}

#[cfg(feature = "iejoin")]
impl<'py> IntoPyObject<'py> for Wrap<InequalityOperator> {
    type Target = PyOperator;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self.0 {
            InequalityOperator::Lt => PyOperator::Lt,
            InequalityOperator::LtEq => PyOperator::LtEq,
            InequalityOperator::Gt => PyOperator::Gt,
            InequalityOperator::GtEq => PyOperator::GtEq,
        }
        .into_pyobject(py)
    }
}

#[pyclass(name = "StringFunction", eq)]
#[derive(Copy, Clone, PartialEq)]
pub enum PyStringFunction {
    ConcatHorizontal,
    ConcatVertical,
    Contains,
    CountMatches,
    EndsWith,
    Extract,
    ExtractAll,
    ExtractGroups,
    Find,
    ToInteger,
    LenBytes,
    LenChars,
    Lowercase,
    JsonDecode,
    JsonPathMatch,
    Replace,
    Reverse,
    PadStart,
    PadEnd,
    Slice,
    Head,
    Tail,
    HexEncode,
    HexDecode,
    Base64Encode,
    Base64Decode,
    StartsWith,
    StripChars,
    StripCharsStart,
    StripCharsEnd,
    StripPrefix,
    StripSuffix,
    SplitExact,
    SplitN,
    Strptime,
    Split,
    ToDecimal,
    Titlecase,
    Uppercase,
    ZFill,
    ContainsAny,
    ReplaceMany,
    EscapeRegex,
    Normalize,
}

#[pymethods]
impl PyStringFunction {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

#[pyclass(name = "BooleanFunction", eq)]
#[derive(Copy, Clone, PartialEq)]
pub enum PyBooleanFunction {
    Any,
    All,
    IsNull,
    IsNotNull,
    IsFinite,
    IsInfinite,
    IsNan,
    IsNotNan,
    IsFirstDistinct,
    IsLastDistinct,
    IsUnique,
    IsDuplicated,
    IsBetween,
    IsIn,
    AllHorizontal,
    AnyHorizontal,
    Not,
}

#[pymethods]
impl PyBooleanFunction {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

#[pyclass(name = "TemporalFunction", eq)]
#[derive(Copy, Clone, PartialEq)]
pub enum PyTemporalFunction {
    Millennium,
    Century,
    Year,
    IsLeapYear,
    IsoYear,
    Quarter,
    Month,
    Week,
    WeekDay,
    Day,
    OrdinalDay,
    Time,
    Date,
    Datetime,
    Duration,
    Hour,
    Minute,
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
    TotalDays,
    TotalHours,
    TotalMinutes,
    TotalSeconds,
    TotalMilliseconds,
    TotalMicroseconds,
    TotalNanoseconds,
    ToString,
    CastTimeUnit,
    WithTimeUnit,
    ConvertTimeZone,
    TimeStamp,
    Truncate,
    OffsetBy,
    MonthStart,
    MonthEnd,
    BaseUtcOffset,
    DSTOffset,
    Round,
    Replace,
    ReplaceTimeZone,
    Combine,
    DatetimeFunction,
}

#[pymethods]
impl PyTemporalFunction {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

#[pyclass]
pub struct BinaryExpr {
    #[pyo3(get)]
    left: usize,
    #[pyo3(get)]
    op: PyObject,
    #[pyo3(get)]
    right: usize,
}

#[pyclass]
pub struct Cast {
    #[pyo3(get)]
    expr: usize,
    #[pyo3(get)]
    dtype: PyObject,
    // 0: strict
    // 1: non-strict
    // 2: overflow
    #[pyo3(get)]
    options: u8,
}

#[pyclass]
pub struct Sort {
    #[pyo3(get)]
    expr: usize,
    #[pyo3(get)]
    /// maintain_order, nulls_last, descending
    options: (bool, bool, bool),
}

#[pyclass]
pub struct Gather {
    #[pyo3(get)]
    expr: usize,
    #[pyo3(get)]
    idx: usize,
    #[pyo3(get)]
    scalar: bool,
}

#[pyclass]
pub struct Filter {
    #[pyo3(get)]
    input: usize,
    #[pyo3(get)]
    by: usize,
}

#[pyclass]
pub struct SortBy {
    #[pyo3(get)]
    expr: usize,
    #[pyo3(get)]
    by: Vec<usize>,
    #[pyo3(get)]
    /// maintain_order, nulls_last, descending
    sort_options: (bool, Vec<bool>, Vec<bool>),
}

#[pyclass]
pub struct Agg {
    #[pyo3(get)]
    name: PyObject,
    #[pyo3(get)]
    arguments: Vec<usize>,
    #[pyo3(get)]
    // Arbitrary control options
    options: PyObject,
}

#[pyclass]
pub struct Ternary {
    #[pyo3(get)]
    predicate: usize,
    #[pyo3(get)]
    truthy: usize,
    #[pyo3(get)]
    falsy: usize,
}

#[pyclass]
pub struct Function {
    #[pyo3(get)]
    input: Vec<usize>,
    #[pyo3(get)]
    function_data: PyObject,
    #[pyo3(get)]
    options: PyObject,
}

#[pyclass]
pub struct Slice {
    #[pyo3(get)]
    input: usize,
    #[pyo3(get)]
    offset: usize,
    #[pyo3(get)]
    length: usize,
}

#[pyclass]
pub struct Len {}

#[pyclass]
pub struct Window {
    #[pyo3(get)]
    function: usize,
    #[pyo3(get)]
    partition_by: Vec<usize>,
    #[pyo3(get)]
    order_by: Option<usize>,
    #[pyo3(get)]
    order_by_descending: bool,
    #[pyo3(get)]
    order_by_nulls_last: bool,
    #[pyo3(get)]
    options: PyObject,
}

#[pyclass(name = "WindowMapping")]
pub struct PyWindowMapping {
    inner: WindowMapping,
}

#[pymethods]
impl PyWindowMapping {
    #[getter]
    fn kind(&self) -> &str {
        self.inner.into()
    }
}

impl<'py> IntoPyObject<'py> for Wrap<Duration> {
    type Target = PyTuple;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        (
            self.0.months(),
            self.0.weeks(),
            self.0.days(),
            self.0.nanoseconds(),
            self.0.parsed_int,
            self.0.negative(),
        )
            .into_pyobject(py)
    }
}

#[pyclass(name = "RollingGroupOptions")]
pub struct PyRollingGroupOptions {
    inner: RollingGroupOptions,
}

#[pymethods]
impl PyRollingGroupOptions {
    #[getter]
    fn index_column(&self) -> &str {
        self.inner.index_column.as_str()
    }

    #[getter]
    fn period(&self) -> Wrap<Duration> {
        Wrap(self.inner.period)
    }

    #[getter]
    fn offset(&self) -> Wrap<Duration> {
        Wrap(self.inner.offset)
    }

    #[getter]
    fn closed_window(&self) -> &str {
        self.inner.closed_window.into()
    }
}

#[pyclass(name = "DynamicGroupOptions")]
pub struct PyDynamicGroupOptions {
    inner: DynamicGroupOptions,
}

#[pymethods]
impl PyDynamicGroupOptions {
    #[getter]
    fn index_column(&self) -> &str {
        self.inner.index_column.as_str()
    }

    #[getter]
    fn every(&self) -> Wrap<Duration> {
        Wrap(self.inner.every)
    }

    #[getter]
    fn period(&self) -> Wrap<Duration> {
        Wrap(self.inner.period)
    }

    #[getter]
    fn offset(&self) -> Wrap<Duration> {
        Wrap(self.inner.offset)
    }

    #[getter]
    fn label(&self) -> &str {
        self.inner.label.into()
    }

    #[getter]
    fn include_boundaries(&self) -> bool {
        self.inner.include_boundaries
    }

    #[getter]
    fn closed_window(&self) -> &str {
        self.inner.closed_window.into()
    }
    #[getter]
    fn start_by(&self) -> &str {
        self.inner.start_by.into()
    }
}

#[pyclass(name = "GroupbyOptions")]
pub struct PyGroupbyOptions {
    inner: GroupbyOptions,
}

impl PyGroupbyOptions {
    pub(crate) fn new(inner: GroupbyOptions) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyGroupbyOptions {
    #[getter]
    fn slice(&self) -> Option<(i64, usize)> {
        self.inner.slice
    }

    #[getter]
    fn dynamic(&self) -> Option<PyDynamicGroupOptions> {
        self.inner
            .dynamic
            .as_ref()
            .map(|f| PyDynamicGroupOptions { inner: f.clone() })
    }

    #[getter]
    fn rolling(&self) -> Option<PyRollingGroupOptions> {
        self.inner
            .rolling
            .as_ref()
            .map(|f| PyRollingGroupOptions { inner: f.clone() })
    }
}

pub(crate) fn into_py(py: Python<'_>, expr: &AExpr) -> PyResult<PyObject> {
    match expr {
        AExpr::Explode(_) => Err(PyNotImplementedError::new_err("explode")),
        AExpr::Alias(inner, name) => Alias {
            expr: inner.0,
            name: name.into_py_any(py)?,
        }
        .into_py_any(py),
        AExpr::Column(name) => Column {
            name: name.into_py_any(py)?,
        }
        .into_py_any(py),
        AExpr::Literal(lit) => {
            use LiteralValue::*;
            let dtype: PyObject = Wrap(lit.get_datatype()).into_py_any(py)?;
            match lit {
                Float(v) => Literal {
                    value: v.into_py_any(py)?,
                    dtype,
                },
                Float32(v) => Literal {
                    value: v.into_py_any(py)?,
                    dtype,
                },
                Float64(v) => Literal {
                    value: v.into_py_any(py)?,
                    dtype,
                },
                Int(v) => Literal {
                    value: v.into_py_any(py)?,
                    dtype,
                },
                Int8(v) => Literal {
                    value: v.into_py_any(py)?,
                    dtype,
                },
                Int16(v) => Literal {
                    value: v.into_py_any(py)?,
                    dtype,
                },
                Int32(v) => Literal {
                    value: v.into_py_any(py)?,
                    dtype,
                },
                Int64(v) => Literal {
                    value: v.into_py_any(py)?,
                    dtype,
                },
                Int128(v) => Literal {
                    value: v.into_py_any(py)?,
                    dtype,
                },
                UInt8(v) => Literal {
                    value: v.into_py_any(py)?,
                    dtype,
                },
                UInt16(v) => Literal {
                    value: v.into_py_any(py)?,
                    dtype,
                },
                UInt32(v) => Literal {
                    value: v.into_py_any(py)?,
                    dtype,
                },
                UInt64(v) => Literal {
                    value: v.into_py_any(py)?,
                    dtype,
                },
                Boolean(v) => Literal {
                    value: v.into_py_any(py)?,
                    dtype,
                },
                StrCat(v) => Literal {
                    value: v.into_py_any(py)?,
                    dtype,
                },
                String(v) => Literal {
                    value: v.into_py_any(py)?,
                    dtype,
                },
                Null => Literal {
                    value: py.None(),
                    dtype,
                },
                Binary(_) => return Err(PyNotImplementedError::new_err("binary literal")),
                Range { .. } => return Err(PyNotImplementedError::new_err("range literal")),
                OtherScalar { .. } => return Err(PyNotImplementedError::new_err("scalar literal")),
                Date(..) | DateTime(..) | Decimal(..) => Literal {
                    value: Wrap(lit.to_any_value().unwrap()).into_py_any(py)?,
                    dtype,
                },
                Duration(v, _) => Literal {
                    value: v.into_py_any(py)?,
                    dtype,
                },
                Time(ns) => Literal {
                    value: ns.into_py_any(py)?,
                    dtype,
                },
                Series(s) => Literal {
                    value: PySeries::new((**s).clone()).into_py_any(py)?,
                    dtype,
                },
            }
        }
        .into_py_any(py),
        AExpr::BinaryExpr { left, op, right } => BinaryExpr {
            left: left.0,
            op: Wrap(*op).into_py_any(py)?,
            right: right.0,
        }
        .into_py_any(py),
        AExpr::Cast {
            expr,
            dtype,
            options,
        } => Cast {
            expr: expr.0,
            dtype: Wrap(dtype.clone()).into_py_any(py)?,
            options: *options as u8,
        }
        .into_py_any(py),
        AExpr::Sort { expr, options } => Sort {
            expr: expr.0,
            options: (
                options.maintain_order,
                options.nulls_last,
                options.descending,
            ),
        }
        .into_py_any(py),
        AExpr::Gather {
            expr,
            idx,
            returns_scalar,
        } => Gather {
            expr: expr.0,
            idx: idx.0,
            scalar: *returns_scalar,
        }
        .into_py_any(py),
        AExpr::Filter { input, by } => Filter {
            input: input.0,
            by: by.0,
        }
        .into_py_any(py),
        AExpr::SortBy {
            expr,
            by,
            sort_options,
        } => SortBy {
            expr: expr.0,
            by: by.iter().map(|n| n.0).collect(),
            sort_options: (
                sort_options.maintain_order,
                sort_options.nulls_last.clone(),
                sort_options.descending.clone(),
            ),
        }
        .into_py_any(py),
        AExpr::Agg(aggexpr) => match aggexpr {
            IRAggExpr::Min {
                input,
                propagate_nans,
            } => Agg {
                name: "min".into_py_any(py)?,
                arguments: vec![input.0],
                options: propagate_nans.into_py_any(py)?,
            },
            IRAggExpr::Max {
                input,
                propagate_nans,
            } => Agg {
                name: "max".into_py_any(py)?,
                arguments: vec![input.0],
                options: propagate_nans.into_py_any(py)?,
            },
            IRAggExpr::Median(n) => Agg {
                name: "median".into_py_any(py)?,
                arguments: vec![n.0],
                options: py.None(),
            },
            IRAggExpr::NUnique(n) => Agg {
                name: "n_unique".into_py_any(py)?,
                arguments: vec![n.0],
                options: py.None(),
            },
            IRAggExpr::First(n) => Agg {
                name: "first".into_py_any(py)?,
                arguments: vec![n.0],
                options: py.None(),
            },
            IRAggExpr::Last(n) => Agg {
                name: "last".into_py_any(py)?,
                arguments: vec![n.0],
                options: py.None(),
            },
            IRAggExpr::Mean(n) => Agg {
                name: "mean".into_py_any(py)?,
                arguments: vec![n.0],
                options: py.None(),
            },
            IRAggExpr::Implode(n) => Agg {
                name: "implode".into_py_any(py)?,
                arguments: vec![n.0],
                options: py.None(),
            },
            IRAggExpr::Quantile {
                expr,
                quantile,
                method: interpol,
            } => Agg {
                name: "quantile".into_py_any(py)?,
                arguments: vec![expr.0, quantile.0],
                options: Into::<&str>::into(interpol).into_py_any(py)?,
            },
            IRAggExpr::Sum(n) => Agg {
                name: "sum".into_py_any(py)?,
                arguments: vec![n.0],
                options: py.None(),
            },
            IRAggExpr::Count(n, include_null) => Agg {
                name: "count".into_py_any(py)?,
                arguments: vec![n.0],
                options: include_null.into_py_any(py)?,
            },
            IRAggExpr::Std(n, ddof) => Agg {
                name: "std".into_py_any(py)?,
                arguments: vec![n.0],
                options: ddof.into_py_any(py)?,
            },
            IRAggExpr::Var(n, ddof) => Agg {
                name: "var".into_py_any(py)?,
                arguments: vec![n.0],
                options: ddof.into_py_any(py)?,
            },
            IRAggExpr::AggGroups(n) => Agg {
                name: "agg_groups".into_py_any(py)?,
                arguments: vec![n.0],
                options: py.None(),
            },
        }
        .into_py_any(py),
        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => Ternary {
            predicate: predicate.0,
            truthy: truthy.0,
            falsy: falsy.0,
        }
        .into_py_any(py),
        AExpr::AnonymousFunction { .. } => Err(PyNotImplementedError::new_err("anonymousfunction")),
        AExpr::Function {
            input,
            function,
            // TODO: expose options
            options: _,
        } => Function {
            input: input.iter().map(|n| n.node().0).collect(),
            function_data: match function {
                FunctionExpr::ArrayExpr(_) => {
                    return Err(PyNotImplementedError::new_err("array expr"))
                },
                FunctionExpr::BinaryExpr(_) => {
                    return Err(PyNotImplementedError::new_err("binary expr"))
                },
                FunctionExpr::Categorical(_) => {
                    return Err(PyNotImplementedError::new_err("categorical expr"))
                },
                FunctionExpr::ListExpr(_) => {
                    return Err(PyNotImplementedError::new_err("list expr"))
                },
                FunctionExpr::Bitwise(_) => {
                    return Err(PyNotImplementedError::new_err("bitwise expr"))
                },
                FunctionExpr::StringExpr(strfun) => match strfun {
                    StringFunction::ConcatHorizontal {
                        delimiter,
                        ignore_nulls,
                    } => (
                        PyStringFunction::ConcatHorizontal,
                        delimiter.as_str(),
                        ignore_nulls,
                    )
                        .into_py_any(py),
                    StringFunction::ConcatVertical {
                        delimiter,
                        ignore_nulls,
                    } => (
                        PyStringFunction::ConcatVertical,
                        delimiter.as_str(),
                        ignore_nulls,
                    )
                        .into_py_any(py),
                    #[cfg(feature = "regex")]
                    StringFunction::Contains { literal, strict } => {
                        (PyStringFunction::Contains, literal, strict).into_py_any(py)
                    },
                    StringFunction::CountMatches(literal) => {
                        (PyStringFunction::CountMatches, literal).into_py_any(py)
                    },
                    StringFunction::EndsWith => (PyStringFunction::EndsWith,).into_py_any(py),
                    StringFunction::Extract(group_index) => {
                        (PyStringFunction::Extract, group_index).into_py_any(py)
                    },
                    StringFunction::ExtractAll => (PyStringFunction::ExtractAll,).into_py_any(py),
                    #[cfg(feature = "extract_groups")]
                    StringFunction::ExtractGroups { dtype, pat } => (
                        PyStringFunction::ExtractGroups,
                        &Wrap(dtype.clone()),
                        pat.as_str(),
                    )
                        .into_py_any(py),
                    #[cfg(feature = "regex")]
                    StringFunction::Find { literal, strict } => {
                        (PyStringFunction::Find, literal, strict).into_py_any(py)
                    },
                    StringFunction::ToInteger(strict) => {
                        (PyStringFunction::ToInteger, strict).into_py_any(py)
                    },
                    StringFunction::LenBytes => (PyStringFunction::LenBytes,).into_py_any(py),
                    StringFunction::LenChars => (PyStringFunction::LenChars,).into_py_any(py),
                    StringFunction::Lowercase => (PyStringFunction::Lowercase,).into_py_any(py),
                    #[cfg(feature = "extract_jsonpath")]
                    StringFunction::JsonDecode {
                        dtype: _,
                        infer_schema_len,
                    } => (PyStringFunction::JsonDecode, infer_schema_len).into_py_any(py),
                    #[cfg(feature = "extract_jsonpath")]
                    StringFunction::JsonPathMatch => {
                        (PyStringFunction::JsonPathMatch,).into_py_any(py)
                    },
                    #[cfg(feature = "regex")]
                    StringFunction::Replace { n, literal } => {
                        (PyStringFunction::Replace, n, literal).into_py_any(py)
                    },
                    StringFunction::Normalize { form } => (
                        PyStringFunction::Normalize,
                        match form {
                            UnicodeForm::NFC => "nfc",
                            UnicodeForm::NFKC => "nfkc",
                            UnicodeForm::NFD => "nfd",
                            UnicodeForm::NFKD => "nfkd",
                        },
                    )
                        .into_py_any(py),
                    StringFunction::Reverse => (PyStringFunction::Reverse,).into_py_any(py),
                    StringFunction::PadStart { length, fill_char } => {
                        (PyStringFunction::PadStart, length, fill_char).into_py_any(py)
                    },
                    StringFunction::PadEnd { length, fill_char } => {
                        (PyStringFunction::PadEnd, length, fill_char).into_py_any(py)
                    },
                    StringFunction::Slice => (PyStringFunction::Slice,).into_py_any(py),
                    StringFunction::Head => (PyStringFunction::Head,).into_py_any(py),
                    StringFunction::Tail => (PyStringFunction::Tail,).into_py_any(py),
                    StringFunction::HexEncode => (PyStringFunction::HexEncode,).into_py_any(py),
                    #[cfg(feature = "binary_encoding")]
                    StringFunction::HexDecode(strict) => {
                        (PyStringFunction::HexDecode, strict).into_py_any(py)
                    },
                    StringFunction::Base64Encode => {
                        (PyStringFunction::Base64Encode,).into_py_any(py)
                    },
                    #[cfg(feature = "binary_encoding")]
                    StringFunction::Base64Decode(strict) => {
                        (PyStringFunction::Base64Decode, strict).into_py_any(py)
                    },
                    StringFunction::StartsWith => (PyStringFunction::StartsWith,).into_py_any(py),
                    StringFunction::StripChars => (PyStringFunction::StripChars,).into_py_any(py),
                    StringFunction::StripCharsStart => {
                        (PyStringFunction::StripCharsStart,).into_py_any(py)
                    },
                    StringFunction::StripCharsEnd => {
                        (PyStringFunction::StripCharsEnd,).into_py_any(py)
                    },
                    StringFunction::StripPrefix => (PyStringFunction::StripPrefix,).into_py_any(py),
                    StringFunction::StripSuffix => (PyStringFunction::StripSuffix,).into_py_any(py),
                    StringFunction::SplitExact { n, inclusive } => {
                        (PyStringFunction::SplitExact, n, inclusive).into_py_any(py)
                    },
                    StringFunction::SplitN(n) => (PyStringFunction::SplitN, n).into_py_any(py),
                    StringFunction::Strptime(_, options) => (
                        PyStringFunction::Strptime,
                        options.format.as_ref().map(|s| s.as_str()),
                        options.strict,
                        options.exact,
                        options.cache,
                    )
                        .into_py_any(py),
                    StringFunction::Split(inclusive) => {
                        (PyStringFunction::Split, inclusive).into_py_any(py)
                    },
                    StringFunction::ToDecimal(inference_length) => {
                        (PyStringFunction::ToDecimal, inference_length).into_py_any(py)
                    },
                    #[cfg(feature = "nightly")]
                    StringFunction::Titlecase => (PyStringFunction::Titlecase,).into_py_any(py),
                    StringFunction::Uppercase => (PyStringFunction::Uppercase,).into_py_any(py),
                    StringFunction::ZFill => (PyStringFunction::ZFill,).into_py_any(py),
                    #[cfg(feature = "find_many")]
                    StringFunction::ContainsAny {
                        ascii_case_insensitive,
                    } => (PyStringFunction::ContainsAny, ascii_case_insensitive).into_py_any(py),
                    #[cfg(feature = "find_many")]
                    StringFunction::ReplaceMany {
                        ascii_case_insensitive,
                    } => (PyStringFunction::ReplaceMany, ascii_case_insensitive).into_py_any(py),
                    #[cfg(feature = "find_many")]
                    StringFunction::ExtractMany { .. } => {
                        return Err(PyNotImplementedError::new_err("extract_many"))
                    },
                    #[cfg(feature = "find_many")]
                    StringFunction::FindMany { .. } => {
                        return Err(PyNotImplementedError::new_err("find_many"))
                    },
                    #[cfg(feature = "regex")]
                    StringFunction::EscapeRegex => (PyStringFunction::EscapeRegex,).into_py_any(py),
                },
                FunctionExpr::StructExpr(_) => {
                    return Err(PyNotImplementedError::new_err("struct expr"))
                },
                FunctionExpr::TemporalExpr(fun) => match fun {
                    TemporalFunction::Millennium => {
                        (PyTemporalFunction::Millennium,).into_py_any(py)
                    },
                    TemporalFunction::Century => (PyTemporalFunction::Century,).into_py_any(py),
                    TemporalFunction::Year => (PyTemporalFunction::Year,).into_py_any(py),
                    TemporalFunction::IsLeapYear => {
                        (PyTemporalFunction::IsLeapYear,).into_py_any(py)
                    },
                    TemporalFunction::IsoYear => (PyTemporalFunction::IsoYear,).into_py_any(py),
                    TemporalFunction::Quarter => (PyTemporalFunction::Quarter,).into_py_any(py),
                    TemporalFunction::Month => (PyTemporalFunction::Month,).into_py_any(py),
                    TemporalFunction::Week => (PyTemporalFunction::Week,).into_py_any(py),
                    TemporalFunction::WeekDay => (PyTemporalFunction::WeekDay,).into_py_any(py),
                    TemporalFunction::Day => (PyTemporalFunction::Day,).into_py_any(py),
                    TemporalFunction::OrdinalDay => {
                        (PyTemporalFunction::OrdinalDay,).into_py_any(py)
                    },
                    TemporalFunction::Time => (PyTemporalFunction::Time,).into_py_any(py),
                    TemporalFunction::Date => (PyTemporalFunction::Date,).into_py_any(py),
                    TemporalFunction::Datetime => (PyTemporalFunction::Datetime,).into_py_any(py),
                    TemporalFunction::Duration(time_unit) => {
                        (PyTemporalFunction::Duration, Wrap(*time_unit)).into_py_any(py)
                    },
                    TemporalFunction::Hour => (PyTemporalFunction::Hour,).into_py_any(py),
                    TemporalFunction::Minute => (PyTemporalFunction::Minute,).into_py_any(py),
                    TemporalFunction::Second => (PyTemporalFunction::Second,).into_py_any(py),
                    TemporalFunction::Millisecond => {
                        (PyTemporalFunction::Millisecond,).into_py_any(py)
                    },
                    TemporalFunction::Microsecond => {
                        (PyTemporalFunction::Microsecond,).into_py_any(py)
                    },
                    TemporalFunction::Nanosecond => {
                        (PyTemporalFunction::Nanosecond,).into_py_any(py)
                    },
                    TemporalFunction::TotalDays => (PyTemporalFunction::TotalDays,).into_py_any(py),
                    TemporalFunction::TotalHours => {
                        (PyTemporalFunction::TotalHours,).into_py_any(py)
                    },
                    TemporalFunction::TotalMinutes => {
                        (PyTemporalFunction::TotalMinutes,).into_py_any(py)
                    },
                    TemporalFunction::TotalSeconds => {
                        (PyTemporalFunction::TotalSeconds,).into_py_any(py)
                    },
                    TemporalFunction::TotalMilliseconds => {
                        (PyTemporalFunction::TotalMilliseconds,).into_py_any(py)
                    },
                    TemporalFunction::TotalMicroseconds => {
                        (PyTemporalFunction::TotalMicroseconds,).into_py_any(py)
                    },
                    TemporalFunction::TotalNanoseconds => {
                        (PyTemporalFunction::TotalNanoseconds,).into_py_any(py)
                    },
                    TemporalFunction::ToString(format) => {
                        (PyTemporalFunction::ToString, format).into_py_any(py)
                    },
                    TemporalFunction::CastTimeUnit(time_unit) => {
                        (PyTemporalFunction::CastTimeUnit, Wrap(*time_unit)).into_py_any(py)
                    },
                    TemporalFunction::WithTimeUnit(time_unit) => {
                        (PyTemporalFunction::WithTimeUnit, Wrap(*time_unit)).into_py_any(py)
                    },
                    #[cfg(feature = "timezones")]
                    TemporalFunction::ConvertTimeZone(time_zone) => {
                        (PyTemporalFunction::ConvertTimeZone, time_zone.as_str()).into_py_any(py)
                    },
                    TemporalFunction::TimeStamp(time_unit) => {
                        (PyTemporalFunction::TimeStamp, Wrap(*time_unit)).into_py_any(py)
                    },
                    TemporalFunction::Truncate => (PyTemporalFunction::Truncate,).into_py_any(py),
                    TemporalFunction::OffsetBy => (PyTemporalFunction::OffsetBy,).into_py_any(py),
                    TemporalFunction::MonthStart => {
                        (PyTemporalFunction::MonthStart,).into_py_any(py)
                    },
                    TemporalFunction::MonthEnd => (PyTemporalFunction::MonthEnd,).into_py_any(py),
                    #[cfg(feature = "timezones")]
                    TemporalFunction::BaseUtcOffset => {
                        (PyTemporalFunction::BaseUtcOffset,).into_py_any(py)
                    },
                    #[cfg(feature = "timezones")]
                    TemporalFunction::DSTOffset => (PyTemporalFunction::DSTOffset,).into_py_any(py),
                    TemporalFunction::Round => (PyTemporalFunction::Round,).into_py_any(py),
                    TemporalFunction::Replace => (PyTemporalFunction::Replace).into_py_any(py),
                    #[cfg(feature = "timezones")]
                    TemporalFunction::ReplaceTimeZone(time_zone, non_existent) => (
                        PyTemporalFunction::ReplaceTimeZone,
                        time_zone.as_ref().map(|s| s.as_str()),
                        Into::<&str>::into(non_existent),
                    )
                        .into_py_any(py),
                    TemporalFunction::Combine(time_unit) => {
                        (PyTemporalFunction::Combine, Wrap(*time_unit)).into_py_any(py)
                    },
                    TemporalFunction::DatetimeFunction {
                        time_unit,
                        time_zone,
                    } => (
                        PyTemporalFunction::DatetimeFunction,
                        Wrap(*time_unit),
                        time_zone.as_ref().map(|s| s.as_str()),
                    )
                        .into_py_any(py),
                },
                FunctionExpr::Boolean(boolfun) => match boolfun {
                    BooleanFunction::Any { ignore_nulls } => {
                        (PyBooleanFunction::Any, *ignore_nulls).into_py_any(py)
                    },
                    BooleanFunction::All { ignore_nulls } => {
                        (PyBooleanFunction::All, *ignore_nulls).into_py_any(py)
                    },
                    BooleanFunction::IsNull => (PyBooleanFunction::IsNull,).into_py_any(py),
                    BooleanFunction::IsNotNull => (PyBooleanFunction::IsNotNull,).into_py_any(py),
                    BooleanFunction::IsFinite => (PyBooleanFunction::IsFinite,).into_py_any(py),
                    BooleanFunction::IsInfinite => (PyBooleanFunction::IsInfinite,).into_py_any(py),
                    BooleanFunction::IsNan => (PyBooleanFunction::IsNan,).into_py_any(py),
                    BooleanFunction::IsNotNan => (PyBooleanFunction::IsNotNan,).into_py_any(py),
                    BooleanFunction::IsFirstDistinct => {
                        (PyBooleanFunction::IsFirstDistinct,).into_py_any(py)
                    },
                    BooleanFunction::IsLastDistinct => {
                        (PyBooleanFunction::IsLastDistinct,).into_py_any(py)
                    },
                    BooleanFunction::IsUnique => (PyBooleanFunction::IsUnique,).into_py_any(py),
                    BooleanFunction::IsDuplicated => {
                        (PyBooleanFunction::IsDuplicated,).into_py_any(py)
                    },
                    BooleanFunction::IsBetween { closed } => {
                        (PyBooleanFunction::IsBetween, Into::<&str>::into(closed)).into_py_any(py)
                    },
                    #[cfg(feature = "is_in")]
                    BooleanFunction::IsIn => (PyBooleanFunction::IsIn,).into_py_any(py),
                    BooleanFunction::AllHorizontal => {
                        (PyBooleanFunction::AllHorizontal,).into_py_any(py)
                    },
                    BooleanFunction::AnyHorizontal => {
                        (PyBooleanFunction::AnyHorizontal,).into_py_any(py)
                    },
                    BooleanFunction::Not => (PyBooleanFunction::Not,).into_py_any(py),
                },
                FunctionExpr::Abs => ("abs",).into_py_any(py),
                #[cfg(feature = "hist")]
                FunctionExpr::Hist {
                    bin_count,
                    include_category,
                    include_breakpoint,
                } => ("hist", bin_count, include_category, include_breakpoint).into_py_any(py),
                FunctionExpr::NullCount => ("null_count",).into_py_any(py),
                FunctionExpr::Pow(f) => match f {
                    PowFunction::Generic => ("pow",).into_py_any(py),
                    PowFunction::Sqrt => ("sqrt",).into_py_any(py),
                    PowFunction::Cbrt => ("cbrt",).into_py_any(py),
                },
                FunctionExpr::Hash(seed, seed_1, seed_2, seed_3) => {
                    ("hash", seed, seed_1, seed_2, seed_3).into_py_any(py)
                },
                FunctionExpr::ArgWhere => ("argwhere",).into_py_any(py),
                #[cfg(feature = "index_of")]
                FunctionExpr::IndexOf => ("index_of",).into_py_any(py),
                #[cfg(feature = "search_sorted")]
                FunctionExpr::SearchSorted(side) => (
                    "search_sorted",
                    match side {
                        SearchSortedSide::Any => "any",
                        SearchSortedSide::Left => "left",
                        SearchSortedSide::Right => "right",
                    },
                )
                    .into_py_any(py),
                FunctionExpr::Range(_) => return Err(PyNotImplementedError::new_err("range")),
                #[cfg(feature = "trigonometry")]
                FunctionExpr::Trigonometry(trigfun) => {
                    use polars_plan::dsl::function_expr::trigonometry::TrigonometricFunction;

                    match trigfun {
                        TrigonometricFunction::Cos => ("cos",),
                        TrigonometricFunction::Cot => ("cot",),
                        TrigonometricFunction::Sin => ("sin",),
                        TrigonometricFunction::Tan => ("tan",),
                        TrigonometricFunction::ArcCos => ("arccos",),
                        TrigonometricFunction::ArcSin => ("arcsin",),
                        TrigonometricFunction::ArcTan => ("arctan",),
                        TrigonometricFunction::Cosh => ("cosh",),
                        TrigonometricFunction::Sinh => ("sinh",),
                        TrigonometricFunction::Tanh => ("tanh",),
                        TrigonometricFunction::ArcCosh => ("arccosh",),
                        TrigonometricFunction::ArcSinh => ("arcsinh",),
                        TrigonometricFunction::ArcTanh => ("arctanh",),
                        TrigonometricFunction::Degrees => ("degrees",),
                        TrigonometricFunction::Radians => ("radians",),
                    }
                    .into_py_any(py)
                },
                #[cfg(feature = "trigonometry")]
                FunctionExpr::Atan2 => ("atan2",).into_py_any(py),
                #[cfg(feature = "sign")]
                FunctionExpr::Sign => ("sign",).into_py_any(py),
                FunctionExpr::FillNull => ("fill_null",).into_py_any(py),
                FunctionExpr::RollingExpr(rolling) => match rolling {
                    RollingFunction::Min(_) => {
                        return Err(PyNotImplementedError::new_err("rolling min"))
                    },
                    RollingFunction::Max(_) => {
                        return Err(PyNotImplementedError::new_err("rolling max"))
                    },
                    RollingFunction::Mean(_) => {
                        return Err(PyNotImplementedError::new_err("rolling mean"))
                    },
                    RollingFunction::Sum(_) => {
                        return Err(PyNotImplementedError::new_err("rolling sum"))
                    },
                    RollingFunction::Quantile(_) => {
                        return Err(PyNotImplementedError::new_err("rolling quantile"))
                    },
                    RollingFunction::Var(_) => {
                        return Err(PyNotImplementedError::new_err("rolling var"))
                    },
                    RollingFunction::Std(_) => {
                        return Err(PyNotImplementedError::new_err("rolling std"))
                    },
                    RollingFunction::Skew(_, _) => {
                        return Err(PyNotImplementedError::new_err("rolling skew"))
                    },
                    RollingFunction::CorrCov { .. } => {
                        return Err(PyNotImplementedError::new_err("rolling cor_cov"))
                    },
                },
                FunctionExpr::RollingExprBy(rolling) => match rolling {
                    RollingFunctionBy::MinBy(_) => {
                        return Err(PyNotImplementedError::new_err("rolling min by"))
                    },
                    RollingFunctionBy::MaxBy(_) => {
                        return Err(PyNotImplementedError::new_err("rolling max by"))
                    },
                    RollingFunctionBy::MeanBy(_) => {
                        return Err(PyNotImplementedError::new_err("rolling mean by"))
                    },
                    RollingFunctionBy::SumBy(_) => {
                        return Err(PyNotImplementedError::new_err("rolling sum by"))
                    },
                    RollingFunctionBy::QuantileBy(_) => {
                        return Err(PyNotImplementedError::new_err("rolling quantile by"))
                    },
                    RollingFunctionBy::VarBy(_) => {
                        return Err(PyNotImplementedError::new_err("rolling var by"))
                    },
                    RollingFunctionBy::StdBy(_) => {
                        return Err(PyNotImplementedError::new_err("rolling std by"))
                    },
                },
                FunctionExpr::ShiftAndFill => ("shift_and_fill",).into_py_any(py),
                FunctionExpr::Shift => ("shift",).into_py_any(py),
                FunctionExpr::DropNans => ("drop_nans",).into_py_any(py),
                FunctionExpr::DropNulls => ("drop_nulls",).into_py_any(py),
                FunctionExpr::Mode => ("mode",).into_py_any(py),
                FunctionExpr::Skew(bias) => ("skew", bias).into_py_any(py),
                FunctionExpr::Kurtosis(fisher, bias) => ("kurtosis", fisher, bias).into_py_any(py),
                FunctionExpr::Reshape(_) => return Err(PyNotImplementedError::new_err("reshape")),
                #[cfg(feature = "repeat_by")]
                FunctionExpr::RepeatBy => ("repeat_by",).into_py_any(py),
                FunctionExpr::ArgUnique => ("arg_unique",).into_py_any(py),
                FunctionExpr::Repeat => ("repeat",).into_py_any(py),
                FunctionExpr::Rank {
                    options: _,
                    seed: _,
                } => return Err(PyNotImplementedError::new_err("rank")),
                FunctionExpr::Clip { has_min, has_max } => {
                    ("clip", has_min, has_max).into_py_any(py)
                },
                FunctionExpr::AsStruct => ("as_struct",).into_py_any(py),
                #[cfg(feature = "top_k")]
                FunctionExpr::TopK { descending } => ("top_k", descending).into_py_any(py),
                FunctionExpr::CumCount { reverse } => ("cum_count", reverse).into_py_any(py),
                FunctionExpr::CumSum { reverse } => ("cum_sum", reverse).into_py_any(py),
                FunctionExpr::CumProd { reverse } => ("cum_prod", reverse).into_py_any(py),
                FunctionExpr::CumMin { reverse } => ("cum_min", reverse).into_py_any(py),
                FunctionExpr::CumMax { reverse } => ("cum_max", reverse).into_py_any(py),
                FunctionExpr::Reverse => ("reverse",).into_py_any(py),
                FunctionExpr::ValueCounts {
                    sort,
                    parallel,
                    name,
                    normalize,
                } => ("value_counts", sort, parallel, name.as_str(), normalize).into_py_any(py),
                FunctionExpr::UniqueCounts => ("unique_counts",).into_py_any(py),
                FunctionExpr::ApproxNUnique => ("approx_n_unique",).into_py_any(py),
                FunctionExpr::Coalesce => ("coalesce",).into_py_any(py),
                FunctionExpr::ShrinkType => ("shrink_dtype",).into_py_any(py),
                FunctionExpr::Diff(n, null_behaviour) => (
                    "diff",
                    n,
                    match null_behaviour {
                        NullBehavior::Drop => "drop",
                        NullBehavior::Ignore => "ignore",
                    },
                )
                    .into_py_any(py),
                #[cfg(feature = "pct_change")]
                FunctionExpr::PctChange => ("pct_change",).into_py_any(py),
                FunctionExpr::Interpolate(method) => (
                    "interpolate",
                    match method {
                        InterpolationMethod::Linear => "linear",
                        InterpolationMethod::Nearest => "nearest",
                    },
                )
                    .into_py_any(py),
                FunctionExpr::InterpolateBy => ("interpolate_by",).into_py_any(py),
                FunctionExpr::Entropy { base, normalize } => {
                    ("entropy", base, normalize).into_py_any(py)
                },
                FunctionExpr::Log { base } => ("log", base).into_py_any(py),
                FunctionExpr::Log1p => ("log1p",).into_py_any(py),
                FunctionExpr::Exp => ("exp",).into_py_any(py),
                FunctionExpr::Unique(maintain_order) => ("unique", maintain_order).into_py_any(py),
                FunctionExpr::Round { decimals } => ("round", decimals).into_py_any(py),
                FunctionExpr::RoundSF { digits } => ("round_sig_figs", digits).into_py_any(py),
                FunctionExpr::Floor => ("floor",).into_py_any(py),
                FunctionExpr::Ceil => ("ceil",).into_py_any(py),
                FunctionExpr::UpperBound => ("upper_bound",).into_py_any(py),
                FunctionExpr::LowerBound => ("lower_bound",).into_py_any(py),
                FunctionExpr::Fused(_) => return Err(PyNotImplementedError::new_err("fused")),
                FunctionExpr::ConcatExpr(_) => {
                    return Err(PyNotImplementedError::new_err("concat expr"))
                },
                FunctionExpr::Correlation { .. } => {
                    return Err(PyNotImplementedError::new_err("corr"))
                },
                #[cfg(feature = "peaks")]
                FunctionExpr::PeakMin => ("peak_max",).into_py_any(py),
                #[cfg(feature = "peaks")]
                FunctionExpr::PeakMax => ("peak_min",).into_py_any(py),
                #[cfg(feature = "cutqcut")]
                FunctionExpr::Cut { .. } => return Err(PyNotImplementedError::new_err("cut")),
                #[cfg(feature = "cutqcut")]
                FunctionExpr::QCut { .. } => return Err(PyNotImplementedError::new_err("qcut")),
                #[cfg(feature = "rle")]
                FunctionExpr::RLE => ("rle",).into_py_any(py),
                #[cfg(feature = "rle")]
                FunctionExpr::RLEID => ("rle_id",).into_py_any(py),
                FunctionExpr::ToPhysical => ("to_physical",).into_py_any(py),
                FunctionExpr::Random { .. } => {
                    return Err(PyNotImplementedError::new_err("random"))
                },
                FunctionExpr::SetSortedFlag(sorted) => (
                    "set_sorted",
                    match sorted {
                        IsSorted::Ascending => "ascending",
                        IsSorted::Descending => "descending",
                        IsSorted::Not => "not",
                    },
                )
                    .into_py_any(py),
                #[cfg(feature = "ffi_plugin")]
                FunctionExpr::FfiPlugin { .. } => {
                    return Err(PyNotImplementedError::new_err("ffi plugin"))
                },
                FunctionExpr::BackwardFill { limit } => ("backward_fill", limit).into_py_any(py),
                FunctionExpr::ForwardFill { limit } => ("forward_fill", limit).into_py_any(py),
                FunctionExpr::SumHorizontal { ignore_nulls } => {
                    ("sum_horizontal", ignore_nulls).into_py_any(py)
                },
                FunctionExpr::MaxHorizontal => ("max_horizontal",).into_py_any(py),
                FunctionExpr::MeanHorizontal { ignore_nulls } => {
                    ("mean_horizontal", ignore_nulls).into_py_any(py)
                },
                FunctionExpr::MinHorizontal => ("min_horizontal",).into_py_any(py),
                FunctionExpr::EwmMean { options: _ } => {
                    return Err(PyNotImplementedError::new_err("ewm mean"))
                },
                FunctionExpr::EwmStd { options: _ } => {
                    return Err(PyNotImplementedError::new_err("ewm std"))
                },
                FunctionExpr::EwmVar { options: _ } => {
                    return Err(PyNotImplementedError::new_err("ewm var"))
                },
                FunctionExpr::Replace => ("replace",).into_py_any(py),
                FunctionExpr::ReplaceStrict { return_dtype: _ } => {
                    // Can ignore the return dtype because it is encoded in the schema.
                    ("replace_strict",).into_py_any(py)
                },
                FunctionExpr::Negate => ("negate",).into_py_any(py),
                FunctionExpr::FillNullWithStrategy(_) => {
                    return Err(PyNotImplementedError::new_err("fill null with strategy"))
                },
                FunctionExpr::GatherEvery { n, offset } => {
                    ("gather_every", offset, n).into_py_any(py)
                },
                FunctionExpr::Reinterpret(signed) => ("reinterpret", signed).into_py_any(py),
                FunctionExpr::ExtendConstant => ("extend_constant",).into_py_any(py),
                FunctionExpr::Business(_) => {
                    return Err(PyNotImplementedError::new_err("business"))
                },
                #[cfg(feature = "top_k")]
                FunctionExpr::TopKBy { descending } => ("top_k_by", descending).into_py_any(py),
                FunctionExpr::EwmMeanBy { half_life: _ } => {
                    return Err(PyNotImplementedError::new_err("ewm_mean_by"))
                },
            }?,
            options: py.None(),
        }
        .into_py_any(py),
        AExpr::Window {
            function,
            partition_by,
            order_by,
            options,
        } => {
            let function = function.0;
            let partition_by = partition_by.iter().map(|n| n.0).collect();
            let order_by_descending = order_by
                .map(|(_, options)| options.descending)
                .unwrap_or(false);
            let order_by_nulls_last = order_by
                .map(|(_, options)| options.nulls_last)
                .unwrap_or(false);
            let order_by = order_by.map(|(n, _)| n.0);

            let options = match options {
                WindowType::Over(options) => PyWindowMapping { inner: *options }.into_py_any(py)?,
                WindowType::Rolling(options) => PyRollingGroupOptions {
                    inner: options.clone(),
                }
                .into_py_any(py)?,
            };
            Window {
                function,
                partition_by,
                order_by,
                order_by_descending,
                order_by_nulls_last,
                options,
            }
            .into_py_any(py)
        },
        AExpr::Slice {
            input,
            offset,
            length,
        } => Slice {
            input: input.0,
            offset: offset.0,
            length: length.0,
        }
        .into_py_any(py),
        AExpr::Len => Len {}.into_py_any(py),
    }
}
