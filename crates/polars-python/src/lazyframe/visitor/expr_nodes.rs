#[cfg(feature = "iejoin")]
use polars::prelude::InequalityOperator;
use polars::series::ops::NullBehavior;
use polars_core::chunked_array::ops::FillNullStrategy;
use polars_core::series::IsSorted;
#[cfg(feature = "string_normalize")]
use polars_ops::chunked_array::UnicodeForm;
use polars_ops::prelude::RankMethod;
use polars_ops::series::InterpolationMethod;
#[cfg(feature = "search_sorted")]
use polars_ops::series::SearchSortedSide;
use polars_plan::plans::{
    DynLiteralValue, IRBooleanFunction, IRFunctionExpr, IRPowFunction, IRRollingFunctionBy,
    IRStringFunction, IRStructFunction, IRTemporalFunction,
};
use polars_plan::prelude::{
    AExpr, GroupbyOptions, IRAggExpr, LiteralValue, Operator, WindowMapping,
};
use polars_time::prelude::RollingGroupOptions;
use polars_time::{Duration, DynamicGroupOptions};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyInt, PyTuple};

use crate::Wrap;
use crate::series::PySeries;

#[pyclass(frozen)]
pub struct Alias {
    #[pyo3(get)]
    expr: usize,
    #[pyo3(get)]
    name: Py<PyAny>,
}

#[pyclass(frozen)]
pub struct Column {
    #[pyo3(get)]
    name: Py<PyAny>,
}

#[pyclass(frozen)]
pub struct Literal {
    #[pyo3(get)]
    value: Py<PyAny>,
    #[pyo3(get)]
    dtype: Py<PyAny>,
}

#[pyclass(name = "Operator", eq, frozen)]
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
            Operator::RustDivide => PyOperator::Divide,
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

#[pyclass(name = "StringFunction", eq, frozen)]
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
    SplitRegex,
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

#[pyclass(name = "BooleanFunction", eq, frozen)]
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
    IsClose,
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

#[pyclass(name = "TemporalFunction", eq, frozen)]
#[derive(Copy, Clone, PartialEq)]
pub enum PyTemporalFunction {
    Millennium,
    Century,
    Year,
    IsLeapYear,
    IsoYear,
    Quarter,
    Month,
    DaysInMonth,
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

#[pyclass(name = "StructFunction", eq, frozen)]
#[derive(Copy, Clone, PartialEq)]
pub enum PyStructFunction {
    FieldByName,
    RenameFields,
    PrefixFields,
    SuffixFields,
    JsonEncode,
    WithFields,
    MapFieldNames,
}

#[pymethods]
impl PyStructFunction {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

#[pyclass(frozen)]
pub struct BinaryExpr {
    #[pyo3(get)]
    left: usize,
    #[pyo3(get)]
    op: Py<PyAny>,
    #[pyo3(get)]
    right: usize,
}

#[pyclass(frozen)]
pub struct Cast {
    #[pyo3(get)]
    expr: usize,
    #[pyo3(get)]
    dtype: Py<PyAny>,
    // 0: strict
    // 1: non-strict
    // 2: overflow
    #[pyo3(get)]
    options: u8,
}

#[pyclass(frozen)]
pub struct Sort {
    #[pyo3(get)]
    expr: usize,
    #[pyo3(get)]
    /// maintain_order, nulls_last, descending
    options: (bool, bool, bool),
}

#[pyclass(frozen)]
pub struct Gather {
    #[pyo3(get)]
    expr: usize,
    #[pyo3(get)]
    idx: usize,
    #[pyo3(get)]
    scalar: bool,
}

#[pyclass(frozen)]
pub struct Filter {
    #[pyo3(get)]
    input: usize,
    #[pyo3(get)]
    by: usize,
}

#[pyclass(frozen)]
pub struct SortBy {
    #[pyo3(get)]
    expr: usize,
    #[pyo3(get)]
    by: Vec<usize>,
    #[pyo3(get)]
    /// maintain_order, nulls_last, descending
    sort_options: (bool, Vec<bool>, Vec<bool>),
}

#[pyclass(frozen)]
pub struct Agg {
    #[pyo3(get)]
    name: Py<PyAny>,
    #[pyo3(get)]
    arguments: Vec<usize>,
    #[pyo3(get)]
    // Arbitrary control options
    options: Py<PyAny>,
}

#[pyclass(frozen)]
pub struct Ternary {
    #[pyo3(get)]
    predicate: usize,
    #[pyo3(get)]
    truthy: usize,
    #[pyo3(get)]
    falsy: usize,
}

#[pyclass(frozen)]
pub struct Function {
    #[pyo3(get)]
    input: Vec<usize>,
    #[pyo3(get)]
    function_data: Py<PyAny>,
    #[pyo3(get)]
    options: Py<PyAny>,
}

#[pyclass(frozen)]
pub struct Slice {
    #[pyo3(get)]
    input: usize,
    #[pyo3(get)]
    offset: usize,
    #[pyo3(get)]
    length: usize,
}

#[pyclass(frozen)]
pub struct Len {}

#[pyclass(frozen)]
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
    options: Py<PyAny>,
}

#[pyclass(name = "WindowMapping", frozen)]
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

#[pyclass(name = "RollingGroupOptions", frozen)]
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

#[pyclass(name = "DynamicGroupOptions", frozen)]
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

#[pyclass(name = "GroupbyOptions", frozen)]
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

pub(crate) fn into_py(py: Python<'_>, expr: &AExpr) -> PyResult<Py<PyAny>> {
    match expr {
        AExpr::Element => Err(PyNotImplementedError::new_err("element")),
        AExpr::Explode { .. } => Err(PyNotImplementedError::new_err("explode")),
        AExpr::Column(name) => Column {
            name: name.into_py_any(py)?,
        }
        .into_py_any(py),
        AExpr::StructField(_) => Err(PyNotImplementedError::new_err("field")),
        AExpr::Literal(lit) => {
            use polars_core::prelude::AnyValue;
            let dtype: Py<PyAny> = Wrap(lit.get_datatype()).into_py_any(py)?;
            let py_value = match lit {
                LiteralValue::Dyn(d) => match d {
                    DynLiteralValue::Int(v) => v.into_py_any(py)?,
                    DynLiteralValue::Float(v) => v.into_py_any(py)?,
                    DynLiteralValue::Str(v) => v.into_py_any(py)?,
                    DynLiteralValue::List(_) => todo!(),
                },
                LiteralValue::Scalar(sc) => {
                    match sc.as_any_value() {
                        // AnyValue conversion of duration to python's
                        // datetime.timedelta drops nanoseconds because
                        // there is no support for them. See
                        // https://github.com/python/cpython/issues/59648
                        AnyValue::Duration(delta, _) => delta.into_py_any(py)?,
                        any => Wrap(any).into_py_any(py)?,
                    }
                },
                LiteralValue::Range(_) => {
                    return Err(PyNotImplementedError::new_err("range literal"));
                },
                LiteralValue::Series(s) => PySeries::new((**s).clone()).into_py_any(py)?,
            };

            Literal {
                value: py_value,
                dtype,
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
            null_on_oob: _,
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
            IRAggExpr::FirstNonNull(n) => Agg {
                name: "first_non_null".into_py_any(py)?,
                arguments: vec![n.0],
                options: py.None(),
            },
            IRAggExpr::Last(n) => Agg {
                name: "last".into_py_any(py)?,
                arguments: vec![n.0],
                options: py.None(),
            },
            IRAggExpr::LastNonNull(n) => Agg {
                name: "last_non_null".into_py_any(py)?,
                arguments: vec![n.0],
                options: py.None(),
            },
            IRAggExpr::Item {
                input: n,
                allow_empty,
            } => Agg {
                name: "item".into_py_any(py)?,
                arguments: vec![n.0],
                options: allow_empty.into_py_any(py)?,
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
            IRAggExpr::Count {
                input: n,
                include_nulls,
            } => Agg {
                name: "count".into_py_any(py)?,
                arguments: vec![n.0],
                options: include_nulls.into_py_any(py)?,
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
        AExpr::AnonymousAgg { .. } => {
            Err(PyNotImplementedError::new_err("anonymous_streaming_agg"))
        },
        AExpr::Function {
            input,
            function,
            // TODO: expose options
            options: _,
        } => Function {
            input: input.iter().map(|n| n.node().0).collect(),
            function_data: match function {
                IRFunctionExpr::ArrayExpr(_) => {
                    return Err(PyNotImplementedError::new_err("array expr"));
                },
                IRFunctionExpr::BinaryExpr(_) => {
                    return Err(PyNotImplementedError::new_err("binary expr"));
                },
                IRFunctionExpr::Categorical(_) => {
                    return Err(PyNotImplementedError::new_err("categorical expr"));
                },
                IRFunctionExpr::Extension(_) => {
                    return Err(PyNotImplementedError::new_err("extension expr"));
                },
                IRFunctionExpr::ListExpr(_) => {
                    return Err(PyNotImplementedError::new_err("list expr"));
                },
                IRFunctionExpr::Bitwise(_) => {
                    return Err(PyNotImplementedError::new_err("bitwise expr"));
                },
                IRFunctionExpr::StringExpr(strfun) => match strfun {
                    IRStringFunction::Format { .. } => {
                        return Err(PyNotImplementedError::new_err("bitwise expr"));
                    },
                    IRStringFunction::ConcatHorizontal {
                        delimiter,
                        ignore_nulls,
                    } => (
                        PyStringFunction::ConcatHorizontal,
                        delimiter.as_str(),
                        ignore_nulls,
                    )
                        .into_py_any(py),
                    IRStringFunction::ConcatVertical {
                        delimiter,
                        ignore_nulls,
                    } => (
                        PyStringFunction::ConcatVertical,
                        delimiter.as_str(),
                        ignore_nulls,
                    )
                        .into_py_any(py),
                    #[cfg(feature = "regex")]
                    IRStringFunction::Contains { literal, strict } => {
                        (PyStringFunction::Contains, literal, strict).into_py_any(py)
                    },
                    IRStringFunction::CountMatches(literal) => {
                        (PyStringFunction::CountMatches, literal).into_py_any(py)
                    },
                    IRStringFunction::EndsWith => (PyStringFunction::EndsWith,).into_py_any(py),
                    IRStringFunction::Extract(group_index) => {
                        (PyStringFunction::Extract, group_index).into_py_any(py)
                    },
                    IRStringFunction::ExtractAll => (PyStringFunction::ExtractAll,).into_py_any(py),
                    #[cfg(feature = "extract_groups")]
                    IRStringFunction::ExtractGroups { dtype, pat } => (
                        PyStringFunction::ExtractGroups,
                        &Wrap(dtype.clone()),
                        pat.as_str(),
                    )
                        .into_py_any(py),
                    #[cfg(feature = "regex")]
                    IRStringFunction::Find { literal, strict } => {
                        (PyStringFunction::Find, literal, strict).into_py_any(py)
                    },
                    IRStringFunction::ToInteger { dtype: _, strict } => {
                        (PyStringFunction::ToInteger, strict).into_py_any(py)
                    },
                    IRStringFunction::LenBytes => (PyStringFunction::LenBytes,).into_py_any(py),
                    IRStringFunction::LenChars => (PyStringFunction::LenChars,).into_py_any(py),
                    IRStringFunction::Lowercase => (PyStringFunction::Lowercase,).into_py_any(py),
                    #[cfg(feature = "extract_jsonpath")]
                    IRStringFunction::JsonDecode(_) => {
                        (PyStringFunction::JsonDecode, <Option<usize>>::None).into_py_any(py)
                    },
                    #[cfg(feature = "extract_jsonpath")]
                    IRStringFunction::JsonPathMatch => {
                        (PyStringFunction::JsonPathMatch,).into_py_any(py)
                    },
                    #[cfg(feature = "regex")]
                    IRStringFunction::Replace { n, literal } => {
                        (PyStringFunction::Replace, n, literal).into_py_any(py)
                    },
                    #[cfg(feature = "string_normalize")]
                    IRStringFunction::Normalize { form } => (
                        PyStringFunction::Normalize,
                        match form {
                            UnicodeForm::NFC => "nfc",
                            UnicodeForm::NFKC => "nfkc",
                            UnicodeForm::NFD => "nfd",
                            UnicodeForm::NFKD => "nfkd",
                        },
                    )
                        .into_py_any(py),
                    IRStringFunction::Reverse => (PyStringFunction::Reverse,).into_py_any(py),
                    IRStringFunction::PadStart { fill_char } => {
                        (PyStringFunction::PadStart, fill_char).into_py_any(py)
                    },
                    IRStringFunction::PadEnd { fill_char } => {
                        (PyStringFunction::PadEnd, fill_char).into_py_any(py)
                    },
                    IRStringFunction::Slice => (PyStringFunction::Slice,).into_py_any(py),
                    IRStringFunction::Head => (PyStringFunction::Head,).into_py_any(py),
                    IRStringFunction::Tail => (PyStringFunction::Tail,).into_py_any(py),
                    IRStringFunction::HexEncode => (PyStringFunction::HexEncode,).into_py_any(py),
                    #[cfg(feature = "binary_encoding")]
                    IRStringFunction::HexDecode(strict) => {
                        (PyStringFunction::HexDecode, strict).into_py_any(py)
                    },
                    IRStringFunction::Base64Encode => {
                        (PyStringFunction::Base64Encode,).into_py_any(py)
                    },
                    #[cfg(feature = "binary_encoding")]
                    IRStringFunction::Base64Decode(strict) => {
                        (PyStringFunction::Base64Decode, strict).into_py_any(py)
                    },
                    IRStringFunction::StartsWith => (PyStringFunction::StartsWith,).into_py_any(py),
                    IRStringFunction::StripChars => (PyStringFunction::StripChars,).into_py_any(py),
                    IRStringFunction::StripCharsStart => {
                        (PyStringFunction::StripCharsStart,).into_py_any(py)
                    },
                    IRStringFunction::StripCharsEnd => {
                        (PyStringFunction::StripCharsEnd,).into_py_any(py)
                    },
                    IRStringFunction::StripPrefix => {
                        (PyStringFunction::StripPrefix,).into_py_any(py)
                    },
                    IRStringFunction::StripSuffix => {
                        (PyStringFunction::StripSuffix,).into_py_any(py)
                    },
                    IRStringFunction::SplitExact { n, inclusive } => {
                        (PyStringFunction::SplitExact, n, inclusive).into_py_any(py)
                    },
                    IRStringFunction::SplitN(n) => (PyStringFunction::SplitN, n).into_py_any(py),
                    IRStringFunction::Strptime(_, options) => (
                        PyStringFunction::Strptime,
                        options.format.as_ref().map(|s| s.as_str()),
                        options.strict,
                        options.exact,
                        options.cache,
                    )
                        .into_py_any(py),
                    IRStringFunction::Split(inclusive) => {
                        (PyStringFunction::Split, inclusive).into_py_any(py)
                    },
                    IRStringFunction::SplitRegex { inclusive, strict } => {
                        (PyStringFunction::SplitRegex, inclusive, strict).into_py_any(py)
                    },
                    IRStringFunction::ToDecimal { scale } => {
                        (PyStringFunction::ToDecimal, scale).into_py_any(py)
                    },
                    #[cfg(feature = "nightly")]
                    IRStringFunction::Titlecase => (PyStringFunction::Titlecase,).into_py_any(py),
                    IRStringFunction::Uppercase => (PyStringFunction::Uppercase,).into_py_any(py),
                    IRStringFunction::ZFill => (PyStringFunction::ZFill,).into_py_any(py),
                    #[cfg(feature = "find_many")]
                    IRStringFunction::ContainsAny {
                        ascii_case_insensitive,
                    } => (PyStringFunction::ContainsAny, ascii_case_insensitive).into_py_any(py),
                    #[cfg(feature = "find_many")]
                    IRStringFunction::ReplaceMany {
                        ascii_case_insensitive,
                        leftmost,
                    } => (
                        PyStringFunction::ReplaceMany,
                        ascii_case_insensitive,
                        leftmost,
                    )
                        .into_py_any(py),
                    #[cfg(feature = "find_many")]
                    IRStringFunction::ExtractMany { .. } => {
                        return Err(PyNotImplementedError::new_err("extract_many"));
                    },
                    #[cfg(feature = "find_many")]
                    IRStringFunction::FindMany { .. } => {
                        return Err(PyNotImplementedError::new_err("find_many"));
                    },
                    #[cfg(feature = "regex")]
                    IRStringFunction::EscapeRegex => {
                        (PyStringFunction::EscapeRegex,).into_py_any(py)
                    },
                },
                IRFunctionExpr::StructExpr(fun) => match fun {
                    IRStructFunction::FieldByName(name) => {
                        (PyStructFunction::FieldByName, name.as_str()).into_py_any(py)
                    },
                    IRStructFunction::RenameFields(names) => {
                        (PyStructFunction::RenameFields, names[0].as_str()).into_py_any(py)
                    },
                    IRStructFunction::PrefixFields(prefix) => {
                        (PyStructFunction::PrefixFields, prefix.as_str()).into_py_any(py)
                    },
                    IRStructFunction::SuffixFields(prefix) => {
                        (PyStructFunction::SuffixFields, prefix.as_str()).into_py_any(py)
                    },
                    #[cfg(feature = "json")]
                    IRStructFunction::JsonEncode => (PyStructFunction::JsonEncode,).into_py_any(py),
                    IRStructFunction::MapFieldNames(_) => {
                        return Err(PyNotImplementedError::new_err("map_field_names"));
                    },
                },
                IRFunctionExpr::TemporalExpr(fun) => match fun {
                    IRTemporalFunction::Millennium => {
                        (PyTemporalFunction::Millennium,).into_py_any(py)
                    },
                    IRTemporalFunction::Century => (PyTemporalFunction::Century,).into_py_any(py),
                    IRTemporalFunction::Year => (PyTemporalFunction::Year,).into_py_any(py),
                    IRTemporalFunction::IsLeapYear => {
                        (PyTemporalFunction::IsLeapYear,).into_py_any(py)
                    },
                    IRTemporalFunction::IsoYear => (PyTemporalFunction::IsoYear,).into_py_any(py),
                    IRTemporalFunction::Quarter => (PyTemporalFunction::Quarter,).into_py_any(py),
                    IRTemporalFunction::Month => (PyTemporalFunction::Month,).into_py_any(py),
                    IRTemporalFunction::Week => (PyTemporalFunction::Week,).into_py_any(py),
                    IRTemporalFunction::WeekDay => (PyTemporalFunction::WeekDay,).into_py_any(py),
                    IRTemporalFunction::Day => (PyTemporalFunction::Day,).into_py_any(py),
                    IRTemporalFunction::OrdinalDay => {
                        (PyTemporalFunction::OrdinalDay,).into_py_any(py)
                    },
                    IRTemporalFunction::Time => (PyTemporalFunction::Time,).into_py_any(py),
                    IRTemporalFunction::Date => (PyTemporalFunction::Date,).into_py_any(py),
                    IRTemporalFunction::Datetime => (PyTemporalFunction::Datetime,).into_py_any(py),
                    IRTemporalFunction::Duration(time_unit) => {
                        (PyTemporalFunction::Duration, Wrap(*time_unit)).into_py_any(py)
                    },
                    IRTemporalFunction::Hour => (PyTemporalFunction::Hour,).into_py_any(py),
                    IRTemporalFunction::Minute => (PyTemporalFunction::Minute,).into_py_any(py),
                    IRTemporalFunction::Second => (PyTemporalFunction::Second,).into_py_any(py),
                    IRTemporalFunction::Millisecond => {
                        (PyTemporalFunction::Millisecond,).into_py_any(py)
                    },
                    IRTemporalFunction::Microsecond => {
                        (PyTemporalFunction::Microsecond,).into_py_any(py)
                    },
                    IRTemporalFunction::Nanosecond => {
                        (PyTemporalFunction::Nanosecond,).into_py_any(py)
                    },
                    IRTemporalFunction::DaysInMonth => {
                        (PyTemporalFunction::DaysInMonth,).into_py_any(py)
                    },
                    IRTemporalFunction::TotalDays { fractional } => {
                        (PyTemporalFunction::TotalDays, fractional).into_py_any(py)
                    },
                    IRTemporalFunction::TotalHours { fractional } => {
                        (PyTemporalFunction::TotalHours, fractional).into_py_any(py)
                    },
                    IRTemporalFunction::TotalMinutes { fractional } => {
                        (PyTemporalFunction::TotalMinutes, fractional).into_py_any(py)
                    },
                    IRTemporalFunction::TotalSeconds { fractional } => {
                        (PyTemporalFunction::TotalSeconds, fractional).into_py_any(py)
                    },
                    IRTemporalFunction::TotalMilliseconds { fractional } => {
                        (PyTemporalFunction::TotalMilliseconds, fractional).into_py_any(py)
                    },
                    IRTemporalFunction::TotalMicroseconds { fractional } => {
                        (PyTemporalFunction::TotalMicroseconds, fractional).into_py_any(py)
                    },
                    IRTemporalFunction::TotalNanoseconds { fractional } => {
                        (PyTemporalFunction::TotalNanoseconds, fractional).into_py_any(py)
                    },
                    IRTemporalFunction::ToString(format) => {
                        (PyTemporalFunction::ToString, format).into_py_any(py)
                    },
                    IRTemporalFunction::CastTimeUnit(time_unit) => {
                        (PyTemporalFunction::CastTimeUnit, Wrap(*time_unit)).into_py_any(py)
                    },
                    IRTemporalFunction::WithTimeUnit(time_unit) => {
                        (PyTemporalFunction::WithTimeUnit, Wrap(*time_unit)).into_py_any(py)
                    },
                    #[cfg(feature = "timezones")]
                    IRTemporalFunction::ConvertTimeZone(time_zone) => {
                        (PyTemporalFunction::ConvertTimeZone, time_zone.as_str()).into_py_any(py)
                    },
                    IRTemporalFunction::TimeStamp(time_unit) => {
                        (PyTemporalFunction::TimeStamp, Wrap(*time_unit)).into_py_any(py)
                    },
                    IRTemporalFunction::Truncate => (PyTemporalFunction::Truncate,).into_py_any(py),
                    IRTemporalFunction::OffsetBy => (PyTemporalFunction::OffsetBy,).into_py_any(py),
                    IRTemporalFunction::MonthStart => {
                        (PyTemporalFunction::MonthStart,).into_py_any(py)
                    },
                    IRTemporalFunction::MonthEnd => (PyTemporalFunction::MonthEnd,).into_py_any(py),
                    #[cfg(feature = "timezones")]
                    IRTemporalFunction::BaseUtcOffset => {
                        (PyTemporalFunction::BaseUtcOffset,).into_py_any(py)
                    },
                    #[cfg(feature = "timezones")]
                    IRTemporalFunction::DSTOffset => {
                        (PyTemporalFunction::DSTOffset,).into_py_any(py)
                    },
                    IRTemporalFunction::Round => (PyTemporalFunction::Round,).into_py_any(py),
                    IRTemporalFunction::Replace => (PyTemporalFunction::Replace).into_py_any(py),
                    #[cfg(feature = "timezones")]
                    IRTemporalFunction::ReplaceTimeZone(time_zone, non_existent) => (
                        PyTemporalFunction::ReplaceTimeZone,
                        time_zone.as_ref().map(|s| s.as_str()),
                        Into::<&str>::into(non_existent),
                    )
                        .into_py_any(py),
                    IRTemporalFunction::Combine(time_unit) => {
                        (PyTemporalFunction::Combine, Wrap(*time_unit)).into_py_any(py)
                    },
                    IRTemporalFunction::DatetimeFunction {
                        time_unit,
                        time_zone,
                    } => (
                        PyTemporalFunction::DatetimeFunction,
                        Wrap(*time_unit),
                        time_zone.as_ref().map(|s| s.as_str()),
                    )
                        .into_py_any(py),
                },
                IRFunctionExpr::Boolean(boolfun) => match boolfun {
                    IRBooleanFunction::Any { ignore_nulls } => {
                        (PyBooleanFunction::Any, *ignore_nulls).into_py_any(py)
                    },
                    IRBooleanFunction::All { ignore_nulls } => {
                        (PyBooleanFunction::All, *ignore_nulls).into_py_any(py)
                    },
                    IRBooleanFunction::IsNull => (PyBooleanFunction::IsNull,).into_py_any(py),
                    IRBooleanFunction::IsNotNull => (PyBooleanFunction::IsNotNull,).into_py_any(py),
                    IRBooleanFunction::IsFinite => (PyBooleanFunction::IsFinite,).into_py_any(py),
                    IRBooleanFunction::IsInfinite => {
                        (PyBooleanFunction::IsInfinite,).into_py_any(py)
                    },
                    IRBooleanFunction::IsNan => (PyBooleanFunction::IsNan,).into_py_any(py),
                    IRBooleanFunction::IsNotNan => (PyBooleanFunction::IsNotNan,).into_py_any(py),
                    IRBooleanFunction::IsFirstDistinct => {
                        (PyBooleanFunction::IsFirstDistinct,).into_py_any(py)
                    },
                    IRBooleanFunction::IsLastDistinct => {
                        (PyBooleanFunction::IsLastDistinct,).into_py_any(py)
                    },
                    IRBooleanFunction::IsUnique => (PyBooleanFunction::IsUnique,).into_py_any(py),
                    IRBooleanFunction::IsDuplicated => {
                        (PyBooleanFunction::IsDuplicated,).into_py_any(py)
                    },
                    IRBooleanFunction::IsBetween { closed } => {
                        (PyBooleanFunction::IsBetween, Into::<&str>::into(closed)).into_py_any(py)
                    },
                    #[cfg(feature = "is_in")]
                    IRBooleanFunction::IsIn { nulls_equal } => {
                        (PyBooleanFunction::IsIn, nulls_equal).into_py_any(py)
                    },
                    IRBooleanFunction::IsClose {
                        abs_tol,
                        rel_tol,
                        nans_equal,
                    } => (PyBooleanFunction::IsClose, abs_tol.0, rel_tol.0, nans_equal)
                        .into_py_any(py),
                    IRBooleanFunction::AllHorizontal => {
                        (PyBooleanFunction::AllHorizontal,).into_py_any(py)
                    },
                    IRBooleanFunction::AnyHorizontal => {
                        (PyBooleanFunction::AnyHorizontal,).into_py_any(py)
                    },
                    IRBooleanFunction::Not => (PyBooleanFunction::Not,).into_py_any(py),
                },
                IRFunctionExpr::Abs => ("abs",).into_py_any(py),
                #[cfg(feature = "hist")]
                IRFunctionExpr::Hist {
                    bin_count,
                    include_category,
                    include_breakpoint,
                } => ("hist", bin_count, include_category, include_breakpoint).into_py_any(py),
                IRFunctionExpr::NullCount => ("null_count",).into_py_any(py),
                IRFunctionExpr::Pow(f) => match f {
                    IRPowFunction::Generic => ("pow",).into_py_any(py),
                    IRPowFunction::Sqrt => ("sqrt",).into_py_any(py),
                    IRPowFunction::Cbrt => ("cbrt",).into_py_any(py),
                },
                IRFunctionExpr::Hash(seed, seed_1, seed_2, seed_3) => {
                    ("hash", seed, seed_1, seed_2, seed_3).into_py_any(py)
                },
                IRFunctionExpr::ArgWhere => ("argwhere",).into_py_any(py),
                #[cfg(feature = "index_of")]
                IRFunctionExpr::IndexOf => ("index_of",).into_py_any(py),
                #[cfg(feature = "search_sorted")]
                IRFunctionExpr::SearchSorted { side, descending } => (
                    "search_sorted",
                    match side {
                        SearchSortedSide::Any => "any",
                        SearchSortedSide::Left => "left",
                        SearchSortedSide::Right => "right",
                    },
                    descending,
                )
                    .into_py_any(py),
                IRFunctionExpr::Range(_) => return Err(PyNotImplementedError::new_err("range")),
                #[cfg(feature = "trigonometry")]
                IRFunctionExpr::Trigonometry(trigfun) => {
                    use polars_plan::plans::IRTrigonometricFunction;

                    match trigfun {
                        IRTrigonometricFunction::Cos => ("cos",),
                        IRTrigonometricFunction::Cot => ("cot",),
                        IRTrigonometricFunction::Sin => ("sin",),
                        IRTrigonometricFunction::Tan => ("tan",),
                        IRTrigonometricFunction::ArcCos => ("arccos",),
                        IRTrigonometricFunction::ArcSin => ("arcsin",),
                        IRTrigonometricFunction::ArcTan => ("arctan",),
                        IRTrigonometricFunction::Cosh => ("cosh",),
                        IRTrigonometricFunction::Sinh => ("sinh",),
                        IRTrigonometricFunction::Tanh => ("tanh",),
                        IRTrigonometricFunction::ArcCosh => ("arccosh",),
                        IRTrigonometricFunction::ArcSinh => ("arcsinh",),
                        IRTrigonometricFunction::ArcTanh => ("arctanh",),
                        IRTrigonometricFunction::Degrees => ("degrees",),
                        IRTrigonometricFunction::Radians => ("radians",),
                    }
                    .into_py_any(py)
                },
                #[cfg(feature = "trigonometry")]
                IRFunctionExpr::Atan2 => ("atan2",).into_py_any(py),
                #[cfg(feature = "sign")]
                IRFunctionExpr::Sign => ("sign",).into_py_any(py),
                IRFunctionExpr::FillNull => ("fill_null",).into_py_any(py),
                IRFunctionExpr::RollingExpr { function, .. } => {
                    return Err(PyNotImplementedError::new_err(format!("{function}")));
                },
                IRFunctionExpr::RollingExprBy { function_by, .. } => match function_by {
                    IRRollingFunctionBy::MinBy => {
                        return Err(PyNotImplementedError::new_err("rolling min by"));
                    },
                    IRRollingFunctionBy::MaxBy => {
                        return Err(PyNotImplementedError::new_err("rolling max by"));
                    },
                    IRRollingFunctionBy::MeanBy => {
                        return Err(PyNotImplementedError::new_err("rolling mean by"));
                    },
                    IRRollingFunctionBy::SumBy => {
                        return Err(PyNotImplementedError::new_err("rolling sum by"));
                    },
                    IRRollingFunctionBy::QuantileBy => {
                        return Err(PyNotImplementedError::new_err("rolling quantile by"));
                    },
                    IRRollingFunctionBy::VarBy => {
                        return Err(PyNotImplementedError::new_err("rolling var by"));
                    },
                    IRRollingFunctionBy::StdBy => {
                        return Err(PyNotImplementedError::new_err("rolling std by"));
                    },
                    IRRollingFunctionBy::RankBy => {
                        return Err(PyNotImplementedError::new_err("rolling rank by"));
                    },
                },
                IRFunctionExpr::Rechunk => ("rechunk",).into_py_any(py),
                IRFunctionExpr::Append { upcast } => ("append", upcast).into_py_any(py),
                IRFunctionExpr::ShiftAndFill => ("shift_and_fill",).into_py_any(py),
                IRFunctionExpr::Shift => ("shift",).into_py_any(py),
                IRFunctionExpr::DropNans => ("drop_nans",).into_py_any(py),
                IRFunctionExpr::DropNulls => ("drop_nulls",).into_py_any(py),
                IRFunctionExpr::Mode { maintain_order } => {
                    ("mode", *maintain_order).into_py_any(py)
                },
                IRFunctionExpr::Skew(bias) => ("skew", bias).into_py_any(py),
                IRFunctionExpr::Kurtosis(fisher, bias) => {
                    ("kurtosis", fisher, bias).into_py_any(py)
                },
                IRFunctionExpr::Reshape(_) => {
                    return Err(PyNotImplementedError::new_err("reshape"));
                },
                #[cfg(feature = "repeat_by")]
                IRFunctionExpr::RepeatBy => ("repeat_by",).into_py_any(py),
                IRFunctionExpr::ArgUnique => ("arg_unique",).into_py_any(py),
                IRFunctionExpr::ArgMin => ("arg_min",).into_py_any(py),
                IRFunctionExpr::ArgMax => ("arg_max",).into_py_any(py),
                IRFunctionExpr::MinBy => ("min_by",).into_py_any(py),
                IRFunctionExpr::MaxBy => ("max_by",).into_py_any(py),
                IRFunctionExpr::ArgSort {
                    descending,
                    nulls_last,
                } => ("arg_max", descending, nulls_last).into_py_any(py),
                IRFunctionExpr::Product => ("product",).into_py_any(py),
                IRFunctionExpr::Repeat => ("repeat",).into_py_any(py),
                IRFunctionExpr::Rank { options, seed } => {
                    let method = match options.method {
                        RankMethod::Average => "average",
                        RankMethod::Min => "min",
                        RankMethod::Max => "max",
                        RankMethod::Dense => "dense",
                        RankMethod::Ordinal => "ordinal",
                        RankMethod::Random => "random",
                    };
                    ("rank", method, options.descending, seed.map(|s| s as i64)).into_py_any(py)
                },
                IRFunctionExpr::Clip { has_min, has_max } => {
                    ("clip", has_min, has_max).into_py_any(py)
                },
                IRFunctionExpr::AsStruct => ("as_struct",).into_py_any(py),
                #[cfg(feature = "top_k")]
                IRFunctionExpr::TopK { descending } => ("top_k", descending).into_py_any(py),
                IRFunctionExpr::CumCount { reverse } => ("cum_count", reverse).into_py_any(py),
                IRFunctionExpr::CumSum { reverse } => ("cum_sum", reverse).into_py_any(py),
                IRFunctionExpr::CumProd { reverse } => ("cum_prod", reverse).into_py_any(py),
                IRFunctionExpr::CumMin { reverse } => ("cum_min", reverse).into_py_any(py),
                IRFunctionExpr::CumMax { reverse } => ("cum_max", reverse).into_py_any(py),
                IRFunctionExpr::Reverse => ("reverse",).into_py_any(py),
                IRFunctionExpr::ValueCounts {
                    sort,
                    parallel,
                    name,
                    normalize,
                } => ("value_counts", sort, parallel, name.as_str(), normalize).into_py_any(py),
                IRFunctionExpr::UniqueCounts => ("unique_counts",).into_py_any(py),
                IRFunctionExpr::ApproxNUnique => ("approx_n_unique",).into_py_any(py),
                IRFunctionExpr::Coalesce => ("coalesce",).into_py_any(py),
                IRFunctionExpr::Diff(null_behaviour) => (
                    "diff",
                    match null_behaviour {
                        NullBehavior::Drop => "drop",
                        NullBehavior::Ignore => "ignore",
                    },
                )
                    .into_py_any(py),
                #[cfg(feature = "pct_change")]
                IRFunctionExpr::PctChange => ("pct_change",).into_py_any(py),
                IRFunctionExpr::Interpolate(method) => (
                    "interpolate",
                    match method {
                        InterpolationMethod::Linear => "linear",
                        InterpolationMethod::Nearest => "nearest",
                    },
                )
                    .into_py_any(py),
                IRFunctionExpr::InterpolateBy => ("interpolate_by",).into_py_any(py),
                IRFunctionExpr::Entropy { base, normalize } => {
                    ("entropy", base, normalize).into_py_any(py)
                },
                IRFunctionExpr::Log => ("log",).into_py_any(py),
                IRFunctionExpr::Log1p => ("log1p",).into_py_any(py),
                IRFunctionExpr::Exp => ("exp",).into_py_any(py),
                IRFunctionExpr::Unique(maintain_order) => {
                    ("unique", maintain_order).into_py_any(py)
                },
                IRFunctionExpr::Round { decimals, mode } => {
                    ("round", decimals, Into::<&str>::into(mode)).into_py_any(py)
                },
                IRFunctionExpr::RoundSF { digits } => ("round_sig_figs", digits).into_py_any(py),
                IRFunctionExpr::Floor => ("floor",).into_py_any(py),
                IRFunctionExpr::Ceil => ("ceil",).into_py_any(py),
                IRFunctionExpr::Fused(_) => return Err(PyNotImplementedError::new_err("fused")),
                IRFunctionExpr::ConcatExpr(_) => {
                    return Err(PyNotImplementedError::new_err("concat expr"));
                },
                IRFunctionExpr::Correlation { .. } => {
                    return Err(PyNotImplementedError::new_err("corr"));
                },
                #[cfg(feature = "peaks")]
                IRFunctionExpr::PeakMin => ("peak_max",).into_py_any(py),
                #[cfg(feature = "peaks")]
                IRFunctionExpr::PeakMax => ("peak_min",).into_py_any(py),
                #[cfg(feature = "cutqcut")]
                IRFunctionExpr::Cut { .. } => return Err(PyNotImplementedError::new_err("cut")),
                #[cfg(feature = "cutqcut")]
                IRFunctionExpr::QCut { .. } => return Err(PyNotImplementedError::new_err("qcut")),
                #[cfg(feature = "rle")]
                IRFunctionExpr::RLE => ("rle",).into_py_any(py),
                #[cfg(feature = "rle")]
                IRFunctionExpr::RLEID => ("rle_id",).into_py_any(py),
                IRFunctionExpr::ToPhysical => ("to_physical",).into_py_any(py),
                IRFunctionExpr::Random { .. } => {
                    return Err(PyNotImplementedError::new_err("random"));
                },
                IRFunctionExpr::SetSortedFlag(sorted) => (
                    "set_sorted",
                    match sorted {
                        IsSorted::Ascending => "ascending",
                        IsSorted::Descending => "descending",
                        IsSorted::Not => "not",
                    },
                )
                    .into_py_any(py),
                #[cfg(feature = "ffi_plugin")]
                IRFunctionExpr::FfiPlugin { .. } => {
                    return Err(PyNotImplementedError::new_err("ffi plugin"));
                },
                IRFunctionExpr::FoldHorizontal { .. } => {
                    Err(PyNotImplementedError::new_err("fold"))
                },
                IRFunctionExpr::ReduceHorizontal { .. } => {
                    Err(PyNotImplementedError::new_err("reduce"))
                },
                IRFunctionExpr::CumReduceHorizontal { .. } => {
                    Err(PyNotImplementedError::new_err("cum_reduce"))
                },
                IRFunctionExpr::CumFoldHorizontal { .. } => {
                    Err(PyNotImplementedError::new_err("cum_fold"))
                },
                IRFunctionExpr::SumHorizontal { ignore_nulls } => {
                    ("sum_horizontal", ignore_nulls).into_py_any(py)
                },
                IRFunctionExpr::MaxHorizontal => ("max_horizontal",).into_py_any(py),
                IRFunctionExpr::MeanHorizontal { ignore_nulls } => {
                    ("mean_horizontal", ignore_nulls).into_py_any(py)
                },
                IRFunctionExpr::MinHorizontal => ("min_horizontal",).into_py_any(py),
                IRFunctionExpr::EwmMean { options: _ } => {
                    return Err(PyNotImplementedError::new_err("ewm mean"));
                },
                IRFunctionExpr::EwmStd { options: _ } => {
                    return Err(PyNotImplementedError::new_err("ewm std"));
                },
                IRFunctionExpr::EwmVar { options: _ } => {
                    return Err(PyNotImplementedError::new_err("ewm var"));
                },
                IRFunctionExpr::Replace => ("replace",).into_py_any(py),
                IRFunctionExpr::ReplaceStrict { return_dtype: _ } => {
                    // Can ignore the return dtype because it is encoded in the schema.
                    ("replace_strict",).into_py_any(py)
                },
                IRFunctionExpr::Negate => ("negate",).into_py_any(py),
                IRFunctionExpr::FillNullWithStrategy(strategy) => {
                    let (strategy_str, py_limit): (&str, Py<PyAny>) = match strategy {
                        FillNullStrategy::Forward(limit) => {
                            let py_limit = limit
                                .map(|v| PyInt::new(py, v).into())
                                .unwrap_or_else(|| py.None());
                            ("forward", py_limit)
                        },
                        FillNullStrategy::Backward(limit) => {
                            let py_limit = limit
                                .map(|v| PyInt::new(py, v).into())
                                .unwrap_or_else(|| py.None());
                            ("backward", py_limit)
                        },
                        FillNullStrategy::Min => ("min", py.None()),
                        FillNullStrategy::Max => ("max", py.None()),
                        FillNullStrategy::Mean => ("mean", py.None()),
                        FillNullStrategy::Zero => ("zero", py.None()),
                        FillNullStrategy::One => ("one", py.None()),
                    };

                    ("fill_null_with_strategy", strategy_str, py_limit).into_py_any(py)
                },
                IRFunctionExpr::GatherEvery { n, offset } => {
                    ("gather_every", offset, n).into_py_any(py)
                },
                IRFunctionExpr::Reinterpret(signed) => ("reinterpret", signed).into_py_any(py),
                IRFunctionExpr::ExtendConstant => ("extend_constant",).into_py_any(py),
                IRFunctionExpr::Business(_) => {
                    return Err(PyNotImplementedError::new_err("business"));
                },
                #[cfg(feature = "top_k")]
                IRFunctionExpr::TopKBy { descending } => ("top_k_by", descending).into_py_any(py),
                IRFunctionExpr::EwmMeanBy { half_life: _ } => {
                    return Err(PyNotImplementedError::new_err("ewm_mean_by"));
                },
                IRFunctionExpr::RowEncode(..) => {
                    return Err(PyNotImplementedError::new_err("row_encode"));
                },
                IRFunctionExpr::RowDecode(..) => {
                    return Err(PyNotImplementedError::new_err("row_decode"));
                },
                IRFunctionExpr::DynamicPred { .. } => {
                    return Err(PyNotImplementedError::new_err("dynamic_pred"));
                },
            }?,
            options: py.None(),
        }
        .into_py_any(py),
        AExpr::Rolling { .. } => Err(PyNotImplementedError::new_err("rolling")),
        AExpr::Over {
            function,
            partition_by,
            order_by,
            mapping,
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

            let options = PyWindowMapping { inner: *mapping }.into_py_any(py)?;
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
        AExpr::Eval { .. } => Err(PyNotImplementedError::new_err("list.eval")),
        AExpr::StructEval { .. } => Err(PyNotImplementedError::new_err("struct.with_fields")),
    }
}
