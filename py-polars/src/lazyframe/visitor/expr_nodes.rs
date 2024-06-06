use polars::datatypes::TimeUnit;
use polars_core::series::IsSorted;
use polars_core::utils::arrow::legacy::kernels::NonExistent;
use polars_ops::prelude::ClosedInterval;
use polars_plan::dsl::function_expr::rolling::RollingFunction;
use polars_plan::dsl::function_expr::rolling_by::RollingFunctionBy;
use polars_plan::dsl::function_expr::trigonometry::TrigonometricFunction;
use polars_plan::dsl::{BooleanFunction, StringFunction, TemporalFunction};
use polars_plan::prelude::{
    AExpr, FunctionExpr, GroupbyOptions, IRAggExpr, LiteralValue, Operator, PowFunction,
    WindowMapping, WindowType,
};
use polars_time::prelude::RollingGroupOptions;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

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

impl IntoPy<PyObject> for Wrap<ClosedInterval> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self.0 {
            ClosedInterval::Both => "both",
            ClosedInterval::Left => "left",
            ClosedInterval::Right => "right",
            ClosedInterval::None => "none",
        }
        .into_py(py)
    }
}

#[pyclass(name = "Operator")]
#[derive(Copy, Clone)]
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

impl IntoPy<PyObject> for Wrap<Operator> {
    fn into_py(self, py: Python<'_>) -> PyObject {
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
        .into_py(py)
    }
}

#[pyclass(name = "StringFunction")]
#[derive(Copy, Clone)]
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
    ContainsMany,
    ReplaceMany,
}

#[pymethods]
impl PyStringFunction {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

#[pyclass(name = "BooleanFunction")]
#[derive(Copy, Clone)]
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

#[pyclass(name = "TemporalFunction")]
#[derive(Copy, Clone)]
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

impl IntoPy<PyObject> for Wrap<TimeUnit> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.0.to_ascii().into_py(py)
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
    arguments: usize,
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
    fn kind(&self, py: Python<'_>) -> PyResult<PyObject> {
        let result = match self.inner {
            WindowMapping::GroupsToRows => "groups_to_rows".to_object(py),
            WindowMapping::Explode => "explode".to_object(py),
            WindowMapping::Join => "join".to_object(py),
        };
        Ok(result.into_py(py))
    }
}

#[pyclass(name = "RollingGroupOptions")]
pub struct PyRollingGroupOptions {
    inner: RollingGroupOptions,
}

#[pymethods]
impl PyRollingGroupOptions {
    #[getter]
    fn index_column(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.inner.index_column.to_object(py))
    }

    #[getter]
    fn period(&self, py: Python<'_>) -> PyResult<PyObject> {
        let result = vec![
            self.inner.period.months().to_object(py),
            self.inner.period.weeks().to_object(py),
            self.inner.period.days().to_object(py),
            self.inner.period.nanoseconds().to_object(py),
            self.inner.period.parsed_int.to_object(py),
        ]
        .into_py(py);
        Ok(result)
    }

    #[getter]
    fn offset(&self, py: Python<'_>) -> PyResult<PyObject> {
        let result = vec![
            self.inner.offset.months().to_object(py),
            self.inner.offset.weeks().to_object(py),
            self.inner.offset.days().to_object(py),
            self.inner.offset.nanoseconds().to_object(py),
            self.inner.offset.parsed_int.to_object(py),
        ]
        .into_py(py);
        Ok(result)
    }

    #[getter]
    fn closed_window(&self, py: Python<'_>) -> PyResult<PyObject> {
        let result = match self.inner.closed_window {
            polars::time::ClosedWindow::Left => "left".to_object(py),
            polars::time::ClosedWindow::Right => "right".to_object(py),
            polars::time::ClosedWindow::Both => "both".to_object(py),
            polars::time::ClosedWindow::None => "none".to_object(py),
        };
        Ok(result.into_py(py))
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
    fn slice(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self
            .inner
            .slice
            .map_or_else(|| py.None(), |f| f.to_object(py)))
    }

    #[getter]
    fn rolling(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.inner.rolling.as_ref().map_or_else(
            || py.None(),
            |f| PyRollingGroupOptions { inner: f.clone() }.into_py(py),
        ))
    }
}

pub(crate) fn into_py(py: Python<'_>, expr: &AExpr) -> PyResult<PyObject> {
    let result = match expr {
        AExpr::Explode(_) => return Err(PyNotImplementedError::new_err("explode")),
        AExpr::Alias(inner, name) => Alias {
            expr: inner.0,
            name: name.to_object(py),
        }
        .into_py(py),
        AExpr::Column(name) => Column {
            name: name.to_object(py),
        }
        .into_py(py),
        AExpr::Literal(lit) => {
            use LiteralValue::*;
            let dtype: PyObject = Wrap(lit.get_datatype()).to_object(py);
            match lit {
                Float(v) => Literal {
                    value: v.to_object(py),
                    dtype,
                },
                Float32(v) => Literal {
                    value: v.to_object(py),
                    dtype,
                },
                Float64(v) => Literal {
                    value: v.to_object(py),
                    dtype,
                },
                Int(v) => Literal {
                    value: v.to_object(py),
                    dtype,
                },
                Int8(v) => Literal {
                    value: v.to_object(py),
                    dtype,
                },
                Int16(v) => Literal {
                    value: v.to_object(py),
                    dtype,
                },
                Int32(v) => Literal {
                    value: v.to_object(py),
                    dtype,
                },
                Int64(v) => Literal {
                    value: v.to_object(py),
                    dtype,
                },
                UInt8(v) => Literal {
                    value: v.to_object(py),
                    dtype,
                },
                UInt16(v) => Literal {
                    value: v.to_object(py),
                    dtype,
                },
                UInt32(v) => Literal {
                    value: v.to_object(py),
                    dtype,
                },
                UInt64(v) => Literal {
                    value: v.to_object(py),
                    dtype,
                },
                Boolean(v) => Literal {
                    value: v.to_object(py),
                    dtype,
                },
                StrCat(v) => Literal {
                    value: v.to_object(py),
                    dtype,
                },
                String(v) => Literal {
                    value: v.to_object(py),
                    dtype,
                },
                Null => Literal {
                    value: py.None(),
                    dtype,
                },
                Binary(_) => return Err(PyNotImplementedError::new_err("binary literal")),
                Range { .. } => return Err(PyNotImplementedError::new_err("range literal")),
                Date(..) | DateTime(..) => Literal {
                    value: Wrap(lit.to_any_value().unwrap()).to_object(py),
                    dtype,
                },
                Duration(v, _) => Literal {
                    value: v.to_object(py),
                    dtype,
                },
                Time(ns) => Literal {
                    value: ns.to_object(py),
                    dtype,
                },
                Series(s) => Literal {
                    value: PySeries::new((**s).clone()).into_py(py),
                    dtype,
                },
            }
        }
        .into_py(py),
        AExpr::BinaryExpr { left, op, right } => BinaryExpr {
            left: left.0,
            op: Wrap(*op).into_py(py),
            right: right.0,
        }
        .into_py(py),
        AExpr::Cast {
            expr,
            data_type,
            options,
        } => Cast {
            expr: expr.0,
            dtype: Wrap(data_type.clone()).to_object(py),
            options: *options as u8,
        }
        .into_py(py),
        AExpr::Sort { expr, options } => Sort {
            expr: expr.0,
            options: (
                options.maintain_order,
                options.nulls_last,
                options.descending,
            ),
        }
        .into_py(py),
        AExpr::Gather {
            expr,
            idx,
            returns_scalar,
        } => Gather {
            expr: expr.0,
            idx: idx.0,
            scalar: *returns_scalar,
        }
        .into_py(py),
        AExpr::Filter { input, by } => Filter {
            input: input.0,
            by: by.0,
        }
        .into_py(py),
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
        .into_py(py),
        AExpr::Agg(aggexpr) => match aggexpr {
            IRAggExpr::Min {
                input,
                propagate_nans,
            } => Agg {
                name: "min".to_object(py),
                arguments: input.0,
                options: propagate_nans.to_object(py),
            },
            IRAggExpr::Max {
                input,
                propagate_nans,
            } => Agg {
                name: "max".to_object(py),
                arguments: input.0,
                options: propagate_nans.to_object(py),
            },
            IRAggExpr::Median(n) => Agg {
                name: "median".to_object(py),
                arguments: n.0,
                options: py.None(),
            },
            IRAggExpr::NUnique(n) => Agg {
                name: "n_unique".to_object(py),
                arguments: n.0,
                options: py.None(),
            },
            IRAggExpr::First(n) => Agg {
                name: "first".to_object(py),
                arguments: n.0,
                options: py.None(),
            },
            IRAggExpr::Last(n) => Agg {
                name: "last".to_object(py),
                arguments: n.0,
                options: py.None(),
            },
            IRAggExpr::Mean(n) => Agg {
                name: "mean".to_object(py),
                arguments: n.0,
                options: py.None(),
            },
            IRAggExpr::Implode(_) => return Err(PyNotImplementedError::new_err("implode")),
            IRAggExpr::Quantile { .. } => return Err(PyNotImplementedError::new_err("quantile")),
            IRAggExpr::Sum(n) => Agg {
                name: "sum".to_object(py),
                arguments: n.0,
                options: py.None(),
            },
            IRAggExpr::Count(n, include_null) => Agg {
                name: "count".to_object(py),
                arguments: n.0,
                options: include_null.to_object(py),
            },
            IRAggExpr::Std(n, ddof) => Agg {
                name: "std".to_object(py),
                arguments: n.0,
                options: ddof.to_object(py),
            },
            IRAggExpr::Var(n, ddof) => Agg {
                name: "var".to_object(py),
                arguments: n.0,
                options: ddof.to_object(py),
            },
            IRAggExpr::AggGroups(n) => Agg {
                name: "agg_groups".to_object(py),
                arguments: n.0,
                options: py.None(),
            },
        }
        .into_py(py),
        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => Ternary {
            predicate: predicate.0,
            truthy: truthy.0,
            falsy: falsy.0,
        }
        .into_py(py),
        AExpr::AnonymousFunction { .. } => {
            return Err(PyNotImplementedError::new_err("anonymousfunction"))
        },
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
                FunctionExpr::StringExpr(strfun) => match strfun {
                    StringFunction::ConcatHorizontal {
                        delimiter,
                        ignore_nulls,
                    } => (
                        PyStringFunction::ConcatHorizontal.into_py(py),
                        delimiter,
                        ignore_nulls,
                    )
                        .to_object(py),
                    StringFunction::ConcatVertical {
                        delimiter,
                        ignore_nulls,
                    } => (
                        PyStringFunction::ConcatVertical.into_py(py),
                        delimiter,
                        ignore_nulls,
                    )
                        .to_object(py),
                    StringFunction::Contains { literal, strict } => {
                        (PyStringFunction::Contains.into_py(py), literal, strict).to_object(py)
                    },
                    StringFunction::CountMatches(_) => {
                        (PyStringFunction::CountMatches.into_py(py),).to_object(py)
                    },
                    StringFunction::EndsWith => {
                        (PyStringFunction::EndsWith.into_py(py),).to_object(py)
                    },
                    StringFunction::Extract(_) => {
                        (PyStringFunction::Extract.into_py(py),).to_object(py)
                    },
                    StringFunction::ExtractAll => {
                        (PyStringFunction::ExtractAll.into_py(py),).to_object(py)
                    },
                    StringFunction::ExtractGroups { dtype, pat } => (
                        PyStringFunction::ExtractGroups.into_py(py),
                        Wrap(dtype.clone()).to_object(py),
                        pat,
                    )
                        .to_object(py),
                    StringFunction::Find { literal, strict } => {
                        (PyStringFunction::Find.into_py(py), literal, strict).to_object(py)
                    },
                    StringFunction::ToInteger(_) => {
                        (PyStringFunction::ToInteger.into_py(py),).to_object(py)
                    },
                    StringFunction::LenBytes => {
                        (PyStringFunction::LenBytes.into_py(py),).to_object(py)
                    },
                    StringFunction::LenChars => {
                        (PyStringFunction::LenChars.into_py(py),).to_object(py)
                    },
                    StringFunction::Lowercase => {
                        (PyStringFunction::Lowercase.into_py(py),).to_object(py)
                    },
                    StringFunction::JsonDecode {
                        dtype: _,
                        infer_schema_len,
                    } => (PyStringFunction::JsonDecode.into_py(py), infer_schema_len).to_object(py),
                    StringFunction::JsonPathMatch => {
                        (PyStringFunction::JsonPathMatch.into_py(py),).to_object(py)
                    },
                    StringFunction::Replace { n, literal } => {
                        (PyStringFunction::Replace.into_py(py), n, literal).to_object(py)
                    },
                    StringFunction::Reverse => {
                        (PyStringFunction::Reverse.into_py(py),).to_object(py)
                    },
                    StringFunction::PadStart { length, fill_char } => {
                        (PyStringFunction::PadStart.into_py(py), length, fill_char).to_object(py)
                    },
                    StringFunction::PadEnd { length, fill_char } => {
                        (PyStringFunction::PadEnd.into_py(py), length, fill_char).to_object(py)
                    },
                    StringFunction::Slice => (PyStringFunction::Slice.into_py(py),).to_object(py),
                    StringFunction::Head => (PyStringFunction::Head.into_py(py),).to_object(py),
                    StringFunction::Tail => (PyStringFunction::Tail.into_py(py),).to_object(py),
                    StringFunction::HexEncode => {
                        (PyStringFunction::HexEncode.into_py(py),).to_object(py)
                    },
                    StringFunction::HexDecode(_) => {
                        (PyStringFunction::HexDecode.into_py(py),).to_object(py)
                    },
                    StringFunction::Base64Encode => {
                        (PyStringFunction::Base64Encode.into_py(py),).to_object(py)
                    },
                    StringFunction::Base64Decode(_) => {
                        (PyStringFunction::Base64Decode.into_py(py),).to_object(py)
                    },
                    StringFunction::StartsWith => {
                        (PyStringFunction::StartsWith.into_py(py),).to_object(py)
                    },
                    StringFunction::StripChars => {
                        (PyStringFunction::StripChars.into_py(py),).to_object(py)
                    },
                    StringFunction::StripCharsStart => {
                        (PyStringFunction::StripCharsStart.into_py(py),).to_object(py)
                    },
                    StringFunction::StripCharsEnd => {
                        (PyStringFunction::StripCharsEnd.into_py(py),).to_object(py)
                    },
                    StringFunction::StripPrefix => {
                        (PyStringFunction::StripPrefix.into_py(py),).to_object(py)
                    },
                    StringFunction::StripSuffix => {
                        (PyStringFunction::StripSuffix.into_py(py),).to_object(py)
                    },
                    StringFunction::SplitExact { n, inclusive } => {
                        (PyStringFunction::SplitExact.into_py(py), n, inclusive).to_object(py)
                    },
                    StringFunction::SplitN(_) => {
                        (PyStringFunction::SplitN.into_py(py),).to_object(py)
                    },
                    StringFunction::Strptime(_, _) => {
                        (PyStringFunction::Strptime.into_py(py),).to_object(py)
                    },
                    StringFunction::Split(_) => {
                        (PyStringFunction::Split.into_py(py),).to_object(py)
                    },
                    StringFunction::ToDecimal(_) => {
                        (PyStringFunction::ToDecimal.into_py(py),).to_object(py)
                    },
                    StringFunction::Titlecase => {
                        (PyStringFunction::Titlecase.into_py(py),).to_object(py)
                    },
                    StringFunction::Uppercase => {
                        (PyStringFunction::Uppercase.into_py(py),).to_object(py)
                    },
                    StringFunction::ZFill => (PyStringFunction::ZFill.into_py(py),).to_object(py),
                    StringFunction::ContainsMany {
                        ascii_case_insensitive,
                    } => (
                        PyStringFunction::ContainsMany.into_py(py),
                        ascii_case_insensitive,
                    )
                        .to_object(py),
                    StringFunction::ReplaceMany {
                        ascii_case_insensitive,
                    } => (
                        PyStringFunction::ReplaceMany.into_py(py),
                        ascii_case_insensitive,
                    )
                        .to_object(py),
                },
                FunctionExpr::StructExpr(_) => {
                    return Err(PyNotImplementedError::new_err("struct expr"))
                },
                FunctionExpr::TemporalExpr(fun) => match fun {
                    TemporalFunction::Millennium => (PyTemporalFunction::Millennium,).into_py(py),
                    TemporalFunction::Century => (PyTemporalFunction::Century,).into_py(py),
                    TemporalFunction::Year => (PyTemporalFunction::Year,).into_py(py),
                    TemporalFunction::IsLeapYear => (PyTemporalFunction::IsLeapYear,).into_py(py),
                    TemporalFunction::IsoYear => (PyTemporalFunction::IsoYear,).into_py(py),
                    TemporalFunction::Quarter => (PyTemporalFunction::Quarter,).into_py(py),
                    TemporalFunction::Month => (PyTemporalFunction::Month,).into_py(py),
                    TemporalFunction::Week => (PyTemporalFunction::Week,).into_py(py),
                    TemporalFunction::WeekDay => (PyTemporalFunction::WeekDay,).into_py(py),
                    TemporalFunction::Day => (PyTemporalFunction::Day,).into_py(py),
                    TemporalFunction::OrdinalDay => (PyTemporalFunction::OrdinalDay,).into_py(py),
                    TemporalFunction::Time => (PyTemporalFunction::Time,).into_py(py),
                    TemporalFunction::Date => (PyTemporalFunction::Date,).into_py(py),
                    TemporalFunction::Datetime => (PyTemporalFunction::Datetime,).into_py(py),
                    TemporalFunction::Duration(time_unit) => {
                        (PyTemporalFunction::Duration, Wrap(*time_unit)).into_py(py)
                    },
                    TemporalFunction::Hour => (PyTemporalFunction::Hour,).into_py(py),
                    TemporalFunction::Minute => (PyTemporalFunction::Minute,).into_py(py),
                    TemporalFunction::Second => (PyTemporalFunction::Second,).into_py(py),
                    TemporalFunction::Millisecond => (PyTemporalFunction::Millisecond,).into_py(py),
                    TemporalFunction::Microsecond => (PyTemporalFunction::Microsecond,).into_py(py),
                    TemporalFunction::Nanosecond => (PyTemporalFunction::Nanosecond,).into_py(py),
                    TemporalFunction::TotalDays => (PyTemporalFunction::TotalDays,).into_py(py),
                    TemporalFunction::TotalHours => (PyTemporalFunction::TotalHours,).into_py(py),
                    TemporalFunction::TotalMinutes => {
                        (PyTemporalFunction::TotalMinutes,).into_py(py)
                    },
                    TemporalFunction::TotalSeconds => {
                        (PyTemporalFunction::TotalSeconds,).into_py(py)
                    },
                    TemporalFunction::TotalMilliseconds => {
                        (PyTemporalFunction::TotalMilliseconds,).into_py(py)
                    },
                    TemporalFunction::TotalMicroseconds => {
                        (PyTemporalFunction::TotalMicroseconds,).into_py(py)
                    },
                    TemporalFunction::TotalNanoseconds => {
                        (PyTemporalFunction::TotalNanoseconds,).into_py(py)
                    },
                    TemporalFunction::ToString(format) => {
                        (PyTemporalFunction::ToString, format).into_py(py)
                    },
                    TemporalFunction::CastTimeUnit(time_unit) => {
                        (PyTemporalFunction::CastTimeUnit, Wrap(*time_unit)).into_py(py)
                    },
                    TemporalFunction::WithTimeUnit(time_unit) => {
                        (PyTemporalFunction::WithTimeUnit, Wrap(*time_unit)).into_py(py)
                    },
                    TemporalFunction::ConvertTimeZone(time_zone) => {
                        (PyTemporalFunction::ConvertTimeZone, time_zone).into_py(py)
                    },
                    TemporalFunction::TimeStamp(time_unit) => {
                        (PyTemporalFunction::TimeStamp, Wrap(*time_unit)).into_py(py)
                    },
                    TemporalFunction::Truncate => (PyTemporalFunction::Truncate).into_py(py),
                    TemporalFunction::OffsetBy => (PyTemporalFunction::OffsetBy,).into_py(py),
                    TemporalFunction::MonthStart => (PyTemporalFunction::MonthStart,).into_py(py),
                    TemporalFunction::MonthEnd => (PyTemporalFunction::MonthEnd,).into_py(py),
                    TemporalFunction::BaseUtcOffset => {
                        (PyTemporalFunction::BaseUtcOffset,).into_py(py)
                    },
                    TemporalFunction::DSTOffset => (PyTemporalFunction::DSTOffset,).into_py(py),
                    TemporalFunction::Round => (PyTemporalFunction::Round).into_py(py),
                    TemporalFunction::ReplaceTimeZone(time_zone, non_existent) => (
                        PyTemporalFunction::ReplaceTimeZone,
                        time_zone
                            .as_ref()
                            .map_or_else(|| py.None(), |s| s.to_object(py)),
                        match non_existent {
                            NonExistent::Null => "nullify",
                            NonExistent::Raise => "raise",
                        },
                    )
                        .into_py(py),
                    TemporalFunction::Combine(time_unit) => {
                        (PyTemporalFunction::Combine, Wrap(*time_unit)).into_py(py)
                    },
                    TemporalFunction::DatetimeFunction {
                        time_unit,
                        time_zone,
                    } => (
                        PyTemporalFunction::DatetimeFunction,
                        Wrap(*time_unit),
                        time_zone
                            .as_ref()
                            .map_or_else(|| py.None(), |s| s.to_object(py)),
                    )
                        .into_py(py),
                },
                FunctionExpr::Boolean(boolfun) => match boolfun {
                    BooleanFunction::Any { ignore_nulls } => {
                        (PyBooleanFunction::Any, *ignore_nulls).into_py(py)
                    },
                    BooleanFunction::All { ignore_nulls } => {
                        (PyBooleanFunction::All, *ignore_nulls).into_py(py)
                    },
                    BooleanFunction::IsNull => (PyBooleanFunction::IsNull,).into_py(py),
                    BooleanFunction::IsNotNull => (PyBooleanFunction::IsNotNull,).into_py(py),
                    BooleanFunction::IsFinite => (PyBooleanFunction::IsFinite,).into_py(py),
                    BooleanFunction::IsInfinite => (PyBooleanFunction::IsInfinite,).into_py(py),
                    BooleanFunction::IsNan => (PyBooleanFunction::IsNan,).into_py(py),
                    BooleanFunction::IsNotNan => (PyBooleanFunction::IsNotNan,).into_py(py),
                    BooleanFunction::IsFirstDistinct => {
                        (PyBooleanFunction::IsFirstDistinct,).into_py(py)
                    },
                    BooleanFunction::IsLastDistinct => {
                        (PyBooleanFunction::IsLastDistinct,).into_py(py)
                    },
                    BooleanFunction::IsUnique => (PyBooleanFunction::IsUnique,).into_py(py),
                    BooleanFunction::IsDuplicated => (PyBooleanFunction::IsDuplicated,).into_py(py),
                    BooleanFunction::IsBetween { closed } => {
                        (PyBooleanFunction::IsBetween, Wrap(*closed)).into_py(py)
                    },
                    BooleanFunction::IsIn => (PyBooleanFunction::IsIn,).into_py(py),
                    BooleanFunction::AllHorizontal => {
                        (PyBooleanFunction::AllHorizontal,).into_py(py)
                    },
                    BooleanFunction::AnyHorizontal => {
                        (PyBooleanFunction::AnyHorizontal,).into_py(py)
                    },
                    BooleanFunction::Not => (PyBooleanFunction::Not,).into_py(py),
                },
                FunctionExpr::Abs => ("abs",).to_object(py),
                FunctionExpr::Hist { .. } => return Err(PyNotImplementedError::new_err("hist")),
                FunctionExpr::NullCount => ("null_count",).to_object(py),
                FunctionExpr::Pow(f) => match f {
                    PowFunction::Generic => ("pow",).to_object(py),
                    PowFunction::Sqrt => ("sqrt",).to_object(py),
                    PowFunction::Cbrt => ("cbrt",).to_object(py),
                },
                FunctionExpr::Hash(_, _, _, _) => {
                    return Err(PyNotImplementedError::new_err("hash"))
                },
                FunctionExpr::ArgWhere => ("argwhere",).to_object(py),
                FunctionExpr::SearchSorted(_) => {
                    return Err(PyNotImplementedError::new_err("search sorted"))
                },
                FunctionExpr::Range(_) => return Err(PyNotImplementedError::new_err("range")),
                FunctionExpr::Trigonometry(trigfun) => match trigfun {
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
                .to_object(py),
                FunctionExpr::Atan2 => ("atan2",).to_object(py),
                FunctionExpr::Sign => ("sign",).to_object(py),
                FunctionExpr::FillNull => return Err(PyNotImplementedError::new_err("fill null")),
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
                FunctionExpr::ShiftAndFill => {
                    return Err(PyNotImplementedError::new_err("shift and fill"))
                },
                FunctionExpr::Shift => ("shift",).to_object(py),
                FunctionExpr::DropNans => ("dropnan",).to_object(py),
                FunctionExpr::DropNulls => ("dropnull",).to_object(py),
                FunctionExpr::Mode => ("mode",).to_object(py),
                FunctionExpr::Skew(_) => return Err(PyNotImplementedError::new_err("skew")),
                FunctionExpr::Kurtosis(_, _) => {
                    return Err(PyNotImplementedError::new_err("kurtosis"))
                },
                FunctionExpr::Reshape(_, _) => {
                    return Err(PyNotImplementedError::new_err("reshape"))
                },
                FunctionExpr::RepeatBy => return Err(PyNotImplementedError::new_err("repeat by")),
                FunctionExpr::ArgUnique => ("argunique",).to_object(py),
                FunctionExpr::Rank {
                    options: _,
                    seed: _,
                } => return Err(PyNotImplementedError::new_err("rank")),
                FunctionExpr::Clip {
                    has_min: _,
                    has_max: _,
                } => return Err(PyNotImplementedError::new_err("clip")),
                FunctionExpr::AsStruct => return Err(PyNotImplementedError::new_err("as struct")),
                FunctionExpr::TopK { .. } => return Err(PyNotImplementedError::new_err("top k")),
                FunctionExpr::CumCount { reverse } => ("cumcount", reverse).to_object(py),
                FunctionExpr::CumSum { reverse } => ("cumsum", reverse).to_object(py),
                FunctionExpr::CumProd { reverse } => ("cumprod", reverse).to_object(py),
                FunctionExpr::CumMin { reverse } => ("cummin", reverse).to_object(py),
                FunctionExpr::CumMax { reverse } => ("cummax", reverse).to_object(py),
                FunctionExpr::Reverse => return Err(PyNotImplementedError::new_err("reverse")),
                FunctionExpr::ValueCounts {
                    sort: _,
                    parallel: _,
                    name: _,
                } => return Err(PyNotImplementedError::new_err("value counts")),
                FunctionExpr::UniqueCounts => ("unique_counts",).to_object(py),
                FunctionExpr::ApproxNUnique => {
                    return Err(PyNotImplementedError::new_err("approx nunique"))
                },
                FunctionExpr::Coalesce => ("coalesce",).to_object(py),
                FunctionExpr::ShrinkType => {
                    return Err(PyNotImplementedError::new_err("shrink type"))
                },
                FunctionExpr::Diff(_, _) => return Err(PyNotImplementedError::new_err("diff")),
                FunctionExpr::PctChange => {
                    return Err(PyNotImplementedError::new_err("pct change"))
                },
                FunctionExpr::Interpolate(_) => {
                    return Err(PyNotImplementedError::new_err("interpolate"))
                },
                FunctionExpr::InterpolateBy => {
                    return Err(PyNotImplementedError::new_err("interpolate_by"))
                },
                FunctionExpr::Entropy {
                    base: _,
                    normalize: _,
                } => return Err(PyNotImplementedError::new_err("entropy")),
                FunctionExpr::Log { base: _ } => return Err(PyNotImplementedError::new_err("log")),
                FunctionExpr::Log1p => return Err(PyNotImplementedError::new_err("log1p")),
                FunctionExpr::Exp => return Err(PyNotImplementedError::new_err("exp")),
                FunctionExpr::Unique(maintain_order) => ("unique", maintain_order).to_object(py),
                FunctionExpr::Round { decimals } => ("round", decimals).to_object(py),
                FunctionExpr::RoundSF { digits } => ("round_sig_figs", digits).to_object(py),
                FunctionExpr::Floor => ("floor",).to_object(py),
                FunctionExpr::Ceil => ("ceil",).to_object(py),
                FunctionExpr::UpperBound => ("upper_bound",).to_object(py),
                FunctionExpr::LowerBound => ("lower_bound",).to_object(py),
                FunctionExpr::Fused(_) => return Err(PyNotImplementedError::new_err("fused")),
                FunctionExpr::ConcatExpr(_) => {
                    return Err(PyNotImplementedError::new_err("concat expr"))
                },
                FunctionExpr::Correlation { .. } => {
                    return Err(PyNotImplementedError::new_err("corr"))
                },
                FunctionExpr::PeakMin => return Err(PyNotImplementedError::new_err("peak min")),
                FunctionExpr::PeakMax => return Err(PyNotImplementedError::new_err("peak max")),
                FunctionExpr::Cut { .. } => return Err(PyNotImplementedError::new_err("cut")),
                FunctionExpr::QCut { .. } => return Err(PyNotImplementedError::new_err("qcut")),
                FunctionExpr::RLE => return Err(PyNotImplementedError::new_err("rle")),
                FunctionExpr::RLEID => return Err(PyNotImplementedError::new_err("rleid")),
                FunctionExpr::ToPhysical => {
                    return Err(PyNotImplementedError::new_err("to physical"))
                },
                FunctionExpr::Random { .. } => {
                    return Err(PyNotImplementedError::new_err("random"))
                },
                FunctionExpr::SetSortedFlag(sorted) => (
                    "setsorted",
                    match sorted {
                        IsSorted::Ascending => "ascending",
                        IsSorted::Descending => "descending",
                        IsSorted::Not => "not",
                    },
                )
                    .to_object(py),
                FunctionExpr::FfiPlugin { .. } => {
                    return Err(PyNotImplementedError::new_err("ffi plugin"))
                },
                FunctionExpr::BackwardFill { limit: _ } => {
                    return Err(PyNotImplementedError::new_err("backward fill"))
                },
                FunctionExpr::ForwardFill { limit: _ } => {
                    return Err(PyNotImplementedError::new_err("forward fill"))
                },
                FunctionExpr::SumHorizontal => {
                    return Err(PyNotImplementedError::new_err("sum horizontal"))
                },
                FunctionExpr::MaxHorizontal => {
                    return Err(PyNotImplementedError::new_err("max horizontal"))
                },
                FunctionExpr::MeanHorizontal => {
                    return Err(PyNotImplementedError::new_err("mean horizontal"))
                },
                FunctionExpr::MinHorizontal => {
                    return Err(PyNotImplementedError::new_err("min horizontal"))
                },
                FunctionExpr::EwmMean { options: _ } => {
                    return Err(PyNotImplementedError::new_err("ewm mean"))
                },
                FunctionExpr::EwmStd { options: _ } => {
                    return Err(PyNotImplementedError::new_err("ewm std"))
                },
                FunctionExpr::EwmVar { options: _ } => {
                    return Err(PyNotImplementedError::new_err("ewm var"))
                },
                FunctionExpr::Replace { return_dtype: _ } => {
                    return Err(PyNotImplementedError::new_err("replace"))
                },
                FunctionExpr::Negate => return Err(PyNotImplementedError::new_err("negate")),
                FunctionExpr::FillNullWithStrategy(_) => {
                    return Err(PyNotImplementedError::new_err("fill null with strategy"))
                },
                FunctionExpr::GatherEvery { n, offset } => {
                    ("gather_every", offset, n).to_object(py)
                },
                FunctionExpr::Reinterpret(_) => {
                    return Err(PyNotImplementedError::new_err("reinterpret"))
                },
                FunctionExpr::ExtendConstant => {
                    return Err(PyNotImplementedError::new_err("extend constant"))
                },
                FunctionExpr::Business(_) => {
                    return Err(PyNotImplementedError::new_err("business"))
                },
                FunctionExpr::TopKBy { .. } => {
                    return Err(PyNotImplementedError::new_err("top_k_by"))
                },
                FunctionExpr::EwmMeanBy { half_life: _ } => {
                    return Err(PyNotImplementedError::new_err("ewm_mean_by"))
                },
            },
            options: py.None(),
        }
        .into_py(py),
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
                WindowType::Over(options) => PyWindowMapping { inner: *options }.into_py(py),
                WindowType::Rolling(options) => PyRollingGroupOptions {
                    inner: options.clone(),
                }
                .into_py(py),
            };
            Window {
                function,
                partition_by,
                order_by,
                order_by_descending,
                order_by_nulls_last,
                options,
            }
            .into_py(py)
        },
        AExpr::Wildcard => return Err(PyNotImplementedError::new_err("wildcard")),
        AExpr::Slice {
            input,
            offset,
            length,
        } => Slice {
            input: input.0,
            offset: offset.0,
            length: length.0,
        }
        .into_py(py),
        AExpr::Nth(_) => return Err(PyNotImplementedError::new_err("nth")),
        AExpr::Len => Len {}.into_py(py),
    };
    Ok(result)
}
