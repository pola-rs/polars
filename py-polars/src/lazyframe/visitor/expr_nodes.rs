use polars_core::series::IsSorted;
use polars_plan::dsl::function_expr::rolling::RollingFunction;
use polars_plan::dsl::function_expr::trigonometry::TrigonometricFunction;
use polars_plan::dsl::BooleanFunction;
use polars_plan::prelude::{
    AAggExpr, AExpr, FunctionExpr, GroupbyOptions, LiteralValue, Operator, PowFunction,
    WindowMapping, WindowType,
};
use polars_time::prelude::RollingGroupOptions;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

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

#[pyclass]
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
    fn __hash__(&self) -> u64 {
        use PyOperator::*;
        match self {
            Eq => Eq as u64,
            EqValidity => EqValidity as u64,
            NotEq => NotEq as u64,
            NotEqValidity => NotEqValidity as u64,
            Lt => Lt as u64,
            LtEq => LtEq as u64,
            Gt => Gt as u64,
            GtEq => GtEq as u64,
            Plus => Plus as u64,
            Minus => Minus as u64,
            Multiply => Multiply as u64,
            Divide => Divide as u64,
            TrueDivide => TrueDivide as u64,
            FloorDivide => FloorDivide as u64,
            Modulus => Modulus as u64,
            And => And as u64,
            Or => Or as u64,
            Xor => Xor as u64,
            LogicalAnd => LogicalAnd as u64,
            LogicalOr => LogicalOr as u64,
        }
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
    #[pyo3(get)]
    strict: bool,
}

#[pyclass]
pub struct Sort {
    #[pyo3(get)]
    expr: usize,
    #[pyo3(get)]
    options: PyObject,
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
    /// descending, nulls_last, maintain_order
    sort_options: (Vec<bool>, bool, bool),
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
pub struct Len {}

#[pyclass]
pub struct Window {
    #[pyo3(get)]
    function: usize,
    #[pyo3(get)]
    partition_by: Vec<usize>,
    #[pyo3(get)]
    options: PyObject,
}

#[pyclass]
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

#[pyclass]
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

    #[getter]
    fn check_sorted(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.inner.check_sorted.into_py(py))
    }
}

#[pyclass]
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
                Duration(_, _) => return Err(PyNotImplementedError::new_err("duration literal")),
                Time(_) => return Err(PyNotImplementedError::new_err("time literal")),
                Series(_) => return Err(PyNotImplementedError::new_err("series literal")),
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
            strict,
        } => Cast {
            expr: expr.0,
            dtype: Wrap(data_type.clone()).to_object(py),
            strict: *strict,
        }
        .into_py(py),
        AExpr::Sort { expr, options } => Sort {
            expr: expr.0,
            options: (
                options.maintain_order,
                options.nulls_last,
                options.descending,
            )
                .to_object(py),
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
                sort_options.descending.clone(),
                sort_options.nulls_last,
                sort_options.maintain_order,
            ),
        }
        .into_py(py),
        AExpr::Agg(aggexpr) => match aggexpr {
            AAggExpr::Min {
                input,
                propagate_nans,
            } => Agg {
                name: "min".to_object(py),
                arguments: input.0,
                options: propagate_nans.to_object(py),
            },
            AAggExpr::Max {
                input,
                propagate_nans,
            } => Agg {
                name: "max".to_object(py),
                arguments: input.0,
                options: propagate_nans.to_object(py),
            },
            AAggExpr::Median(n) => Agg {
                name: "median".to_object(py),
                arguments: n.0,
                options: py.None(),
            },
            AAggExpr::NUnique(n) => Agg {
                name: "nunique".to_object(py),
                arguments: n.0,
                options: py.None(),
            },
            AAggExpr::First(n) => Agg {
                name: "first".to_object(py),
                arguments: n.0,
                options: py.None(),
            },
            AAggExpr::Last(n) => Agg {
                name: "last".to_object(py),
                arguments: n.0,
                options: py.None(),
            },
            AAggExpr::Mean(n) => Agg {
                name: "mean".to_object(py),
                arguments: n.0,
                options: py.None(),
            },
            AAggExpr::Implode(_) => return Err(PyNotImplementedError::new_err("implode")),
            AAggExpr::Quantile { .. } => return Err(PyNotImplementedError::new_err("quantile")),
            AAggExpr::Sum(n) => Agg {
                name: "sum".to_object(py),
                arguments: n.0,
                options: py.None(),
            },
            AAggExpr::Count(n, include_null) => Agg {
                name: "count".to_object(py),
                arguments: n.0,
                options: include_null.to_object(py),
            },
            AAggExpr::Std(n, ddof) => Agg {
                name: "std".to_object(py),
                arguments: n.0,
                options: ddof.to_object(py),
            },
            AAggExpr::Var(n, ddof) => Agg {
                name: "var".to_object(py),
                arguments: n.0,
                options: ddof.to_object(py),
            },
            AAggExpr::AggGroups(n) => Agg {
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
                FunctionExpr::StringExpr(_) => {
                    return Err(PyNotImplementedError::new_err("string expr"))
                },
                FunctionExpr::StructExpr(_) => {
                    return Err(PyNotImplementedError::new_err("struct expr"))
                },
                FunctionExpr::TemporalExpr(_) => {
                    return Err(PyNotImplementedError::new_err("temporal expr"))
                },
                FunctionExpr::Boolean(boolfun) => match boolfun {
                    BooleanFunction::IsNull => ("is_null",).to_object(py),
                    BooleanFunction::IsNotNull => ("is_not_null",).to_object(py),
                    _ => return Err(PyNotImplementedError::new_err("boolean expr")),
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
                FunctionExpr::DateOffset => {
                    return Err(PyNotImplementedError::new_err("date offset"))
                },
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
                    RollingFunction::MinBy(_) => {
                        return Err(PyNotImplementedError::new_err("rolling min by"))
                    },
                    RollingFunction::Max(_) => {
                        return Err(PyNotImplementedError::new_err("rolling max"))
                    },
                    RollingFunction::MaxBy(_) => {
                        return Err(PyNotImplementedError::new_err("rolling max by"))
                    },
                    RollingFunction::Mean(_) => {
                        return Err(PyNotImplementedError::new_err("rolling mean"))
                    },
                    RollingFunction::MeanBy(_) => {
                        return Err(PyNotImplementedError::new_err("rolling mean by"))
                    },
                    RollingFunction::Sum(_) => {
                        return Err(PyNotImplementedError::new_err("rolling sum"))
                    },
                    RollingFunction::SumBy(_) => {
                        return Err(PyNotImplementedError::new_err("rolling sum by"))
                    },
                    RollingFunction::Quantile(_) => {
                        return Err(PyNotImplementedError::new_err("rolling quantile"))
                    },
                    RollingFunction::QuantileBy(_) => {
                        return Err(PyNotImplementedError::new_err("rolling quantile by"))
                    },
                    RollingFunction::Var(_) => {
                        return Err(PyNotImplementedError::new_err("rolling var"))
                    },
                    RollingFunction::VarBy(_) => {
                        return Err(PyNotImplementedError::new_err("rolling var by"))
                    },
                    RollingFunction::Std(_) => {
                        return Err(PyNotImplementedError::new_err("rolling std"))
                    },
                    RollingFunction::StdBy(_) => {
                        return Err(PyNotImplementedError::new_err("rolling std by"))
                    },
                    RollingFunction::Skew(_, _) => {
                        return Err(PyNotImplementedError::new_err("rolling skew"))
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
                FunctionExpr::Reshape(_) => return Err(PyNotImplementedError::new_err("reshape")),
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
                FunctionExpr::TopK { sort_options: _ } => {
                    return Err(PyNotImplementedError::new_err("top k"))
                },
                FunctionExpr::CumCount { reverse } => ("cumcount", reverse).to_object(py),
                FunctionExpr::CumSum { reverse } => ("cumsum", reverse).to_object(py),
                FunctionExpr::CumProd { reverse } => ("cumprod", reverse).to_object(py),
                FunctionExpr::CumMin { reverse } => ("cummin", reverse).to_object(py),
                FunctionExpr::CumMax { reverse } => ("cummax", reverse).to_object(py),
                FunctionExpr::Reverse => return Err(PyNotImplementedError::new_err("reverse")),
                FunctionExpr::ValueCounts {
                    sort: _,
                    parallel: _,
                } => return Err(PyNotImplementedError::new_err("value counts")),
                FunctionExpr::UniqueCounts => {
                    return Err(PyNotImplementedError::new_err("unique counts"))
                },
                FunctionExpr::ApproxNUnique => {
                    return Err(PyNotImplementedError::new_err("approx nunique"))
                },
                FunctionExpr::Coalesce => return Err(PyNotImplementedError::new_err("coalesce")),
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
                FunctionExpr::Entropy {
                    base: _,
                    normalize: _,
                } => return Err(PyNotImplementedError::new_err("entropy")),
                FunctionExpr::Log { base: _ } => return Err(PyNotImplementedError::new_err("log")),
                FunctionExpr::Log1p => return Err(PyNotImplementedError::new_err("log1p")),
                FunctionExpr::Exp => return Err(PyNotImplementedError::new_err("exp")),
                FunctionExpr::Unique(_) => return Err(PyNotImplementedError::new_err("unique")),
                FunctionExpr::Round { decimals: _ } => {
                    return Err(PyNotImplementedError::new_err("round"))
                },
                FunctionExpr::RoundSF { digits: _ } => {
                    return Err(PyNotImplementedError::new_err("round sf"))
                },
                FunctionExpr::Floor => ("floor",).to_object(py),
                FunctionExpr::Ceil => ("ceil",).to_object(py),
                FunctionExpr::UpperBound => {
                    return Err(PyNotImplementedError::new_err("upper bound"))
                },
                FunctionExpr::LowerBound => {
                    return Err(PyNotImplementedError::new_err("lower bound"))
                },
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
                    ("strided_slice", offset, n).to_object(py)
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
                FunctionExpr::TopKBy { sort_options: _ } => {
                    return Err(PyNotImplementedError::new_err("top_k_by"))
                },
                FunctionExpr::EwmMeanBy {
                    half_life: _,
                    check_sorted: _,
                } => return Err(PyNotImplementedError::new_err("ewm_mean_by")),
            },
            options: py.None(),
        }
        .into_py(py),
        AExpr::Window {
            function,
            partition_by,
            options,
        } => {
            let function = function.0;
            let partition_by = partition_by.iter().map(|n| n.0).collect();
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
                options,
            }
            .into_py(py)
        },
        AExpr::Wildcard => return Err(PyNotImplementedError::new_err("wildcard")),
        AExpr::Slice { .. } => return Err(PyNotImplementedError::new_err("slice")),
        AExpr::Nth(_) => return Err(PyNotImplementedError::new_err("nth")),
        AExpr::Len => Len {}.into_py(py),
    };
    Ok(result)
}
