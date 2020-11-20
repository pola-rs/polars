use crate::error::PyPolarsEr;
use crate::series::PySeries;
use polars::lazy::dsl;
use polars::lazy::dsl::Operator;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyInt};
use pyo3::{class::basic::CompareOp, PyNumberProtocol, PyObjectProtocol};

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyExpr {
    pub inner: dsl::Expr,
}

#[pyproto]
impl PyNumberProtocol for PyExpr {
    fn __add__(lhs: Self, rhs: Self) -> PyResult<PyExpr> {
        Ok(dsl::binary_expr(lhs.inner, Operator::Plus, rhs.inner).into())
    }
    fn __sub__(lhs: Self, rhs: Self) -> PyResult<PyExpr> {
        Ok(dsl::binary_expr(lhs.inner, Operator::Minus, rhs.inner).into())
    }
    fn __mul__(lhs: Self, rhs: Self) -> PyResult<PyExpr> {
        Ok(dsl::binary_expr(lhs.inner, Operator::Multiply, rhs.inner).into())
    }
    fn __truediv__(lhs: Self, rhs: Self) -> PyResult<PyExpr> {
        Ok(dsl::binary_expr(lhs.inner, Operator::Divide, rhs.inner).into())
    }
}

#[pyproto]
impl<'p> PyObjectProtocol<'p> for PyExpr {
    fn __richcmp__(&'p self, other: PyExpr, op: CompareOp) -> PyExpr {
        match op {
            CompareOp::Eq => self.eq(other),
            CompareOp::Ne => self.neq(other),
            CompareOp::Gt => self.gt(other),
            CompareOp::Lt => self.lt(other),
            CompareOp::Ge => self.gt_eq(other),
            CompareOp::Le => self.lt_eq(other),
        }
    }
}

#[pymethods]
impl PyExpr {
    #[text_signature = "($self, other)"]
    pub fn eq(&self, other: PyExpr) -> PyExpr {
        self.clone().inner.eq(other.inner).into()
    }
    #[text_signature = "($self, other)"]
    pub fn neq(&self, other: PyExpr) -> PyExpr {
        self.clone().inner.neq(other.inner).into()
    }
    #[text_signature = "($self, other)"]
    pub fn gt(&self, other: PyExpr) -> PyExpr {
        self.clone().inner.gt(other.inner).into()
    }
    #[text_signature = "($self, other)"]
    pub fn gt_eq(&self, other: PyExpr) -> PyExpr {
        self.clone().inner.gt_eq(other.inner).into()
    }
    #[text_signature = "($self, other)"]
    pub fn lt_eq(&self, other: PyExpr) -> PyExpr {
        self.clone().inner.lt_eq(other.inner).into()
    }
    #[text_signature = "($self, other)"]
    pub fn lt(&self, other: PyExpr) -> PyExpr {
        self.clone().inner.lt(other.inner).into()
    }
    #[text_signature = "($self, name)"]
    pub fn alias(&self, name: &str) -> PyExpr {
        self.clone().inner.alias(name).into()
    }
    #[text_signature = "($self)"]
    pub fn is_not(&self) -> PyExpr {
        self.clone().inner.not().into()
    }
    #[text_signature = "($self)"]
    pub fn is_null(&self) -> PyExpr {
        self.clone().inner.is_null().into()
    }
    #[text_signature = "($self)"]
    pub fn is_not_null(&self) -> PyExpr {
        self.clone().inner.is_not_null().into()
    }
    #[text_signature = "($self)"]
    pub fn agg_min(&self) -> PyExpr {
        self.clone().inner.agg_min().into()
    }
    #[text_signature = "($self)"]
    pub fn agg_max(&self) -> PyExpr {
        self.clone().inner.agg_max().into()
    }
    #[text_signature = "($self)"]
    pub fn agg_mean(&self) -> PyExpr {
        self.clone().inner.agg_mean().into()
    }
    #[text_signature = "($self)"]
    pub fn agg_median(&self) -> PyExpr {
        self.clone().inner.agg_median().into()
    }
    #[text_signature = "($self)"]
    pub fn agg_sum(&self) -> PyExpr {
        self.clone().inner.agg_sum().into()
    }
    #[text_signature = "($self)"]
    pub fn agg_n_unique(&self) -> PyExpr {
        self.clone().inner.agg_n_unique().into()
    }
    #[text_signature = "($self)"]
    pub fn agg_first(&self) -> PyExpr {
        self.clone().inner.agg_first().into()
    }
    #[text_signature = "($self)"]
    pub fn agg_last(&self) -> PyExpr {
        self.clone().inner.agg_last().into()
    }
    pub fn agg_list(&self) -> PyExpr {
        self.clone().inner.agg_list().into()
    }
    #[text_signature = "($self, quantile)"]
    pub fn agg_quantile(&self, quantile: f64) -> PyExpr {
        self.clone().inner.agg_quantile(quantile).into()
    }
    #[text_signature = "($self)"]
    pub fn agg_groups(&self) -> PyExpr {
        self.clone().inner.agg_groups().into()
    }

    #[text_signature = "($self, data_type)"]
    pub fn cast(&self, data_type: &PyAny) -> PyExpr {
        // TODO! accept the DataType objects.

        let str_repr = data_type.str().unwrap().to_str().unwrap();
        let dt = str_to_arrow_type(str_repr);
        let expr = self.inner.clone().cast(dt);
        expr.into()
    }
    #[text_signature = "($self, reverse)"]
    pub fn sort(&self, reverse: bool) -> PyExpr {
        self.clone().inner.sort(reverse).into()
    }
    pub fn shift(&self, periods: i32) -> PyExpr {
        self.clone().inner.shift(periods).into()
    }
    pub fn fill_none(&self, strategy: &str) -> PyResult<PyExpr> {
        let strat = match strategy {
            "backward" => FillNoneStrategy::Backward,
            "forward" => FillNoneStrategy::Forward,
            "min" => FillNoneStrategy::Min,
            "max" => FillNoneStrategy::Max,
            "mean" => FillNoneStrategy::Mean,
            s => return Err(PyPolarsEr::Other(format!("Strategy {} not supported", s)).into()),
        };
        Ok(self.clone().inner.fill_none(strat).into())
    }
    pub fn max(&self) -> PyExpr {
        self.clone().inner.max().into()
    }
    pub fn min(&self) -> PyExpr {
        self.clone().inner.min().into()
    }
    pub fn sum(&self) -> PyExpr {
        self.clone().inner.sum().into()
    }
    pub fn mean(&self) -> PyExpr {
        self.clone().inner.mean().into()
    }
    pub fn median(&self) -> PyExpr {
        self.clone().inner.median().into()
    }
    pub fn is_unique(&self) -> PyExpr {
        self.clone().inner.is_unique().into()
    }
    pub fn is_duplicated(&self) -> PyExpr {
        self.clone().inner.is_duplicated().into()
    }
    pub fn quantile(&self, quantile: f64) -> PyExpr {
        self.clone().inner.quantile(quantile).into()
    }
    pub fn str_lengths(&self) -> PyExpr {
        let function = |s: Series| {
            let ca = s.utf8()?;
            Ok(ca.str_lengths().into_series())
        };
        self.clone().inner.apply(function, None).into()
    }

    pub fn str_replace(&self, pat: String, val: String) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            match ca.replace(&pat, &val) {
                Ok(ca) => Ok(ca.into_series()),
                Err(e) => Err(PolarsError::Other(format!("{:?}", e).into())),
            }
        };
        self.clone().inner.apply(function, None).into()
    }

    pub fn str_replace_all(&self, pat: String, val: String) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            match ca.replace_all(&pat, &val) {
                Ok(ca) => Ok(ca.into_series()),
                Err(e) => Err(PolarsError::Other(format!("{:?}", e).into())),
            }
        };
        self.clone().inner.apply(function, None).into()
    }

    pub fn str_contains(&self, pat: String) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            match ca.contains(&pat) {
                Ok(ca) => Ok(ca.into_series()),
                Err(e) => Err(PolarsError::Other(format!("{:?}", e).into())),
            }
        };
        self.clone().inner.apply(function, None).into()
    }

    pub fn apply(&self, lambda: PyObject, output_type: &PyAny) -> PyExpr {
        let output_type = match output_type.is_none() {
            true => None,
            false => {
                let str_repr = output_type.str().unwrap().to_str().unwrap();
                Some(str_to_arrow_type(str_repr))
            }
        };

        let function = move |s: Series| {
            let gil = Python::acquire_gil();
            let py = gil.python();
            // get the pypolars module
            let pypolars = PyModule::import(py, "pypolars").unwrap();
            // create a PySeries struct/object for Python
            let pyseries = PySeries::new(s);
            // Wrap this PySeries object in the python side Series wrapper
            let python_series_wrapper = pypolars.call1("wrap_s", (pyseries,)).unwrap();
            // call the lambda en get a python side Series wrapper
            let result_series_wrapper = lambda.call1(py, (python_series_wrapper,)).unwrap();
            // unpack the wrapper in a PySeries
            let py_pyseries = result_series_wrapper
                .getattr(py, "_s")
                .expect("could net get series attribute '_s'");
            // Downcast to Rust
            let pyseries = py_pyseries.extract::<PySeries>(py).unwrap();
            // Finally get the actual Series
            Ok(pyseries.series)
        };

        self.clone().inner.apply(function, output_type).into()
    }
}

fn str_to_arrow_type(s: &str) -> ArrowDataType {
    match s {
        "<class 'pypolars.datatypes.UInt8'>" => ArrowDataType::UInt8,
        "<class 'pypolars.datatypes.UInt16'>" => ArrowDataType::UInt16,
        "<class 'pypolars.datatypes.UInt32'>" => ArrowDataType::UInt32,
        "<class 'pypolars.datatypes.UInt64'>" => ArrowDataType::UInt64,
        "<class 'pypolars.datatypes.Int8'>" => ArrowDataType::Int8,
        "<class 'pypolars.datatypes.Int16'>" => ArrowDataType::Int16,
        "<class 'pypolars.datatypes.Int32'>" => ArrowDataType::Int32,
        "<class 'pypolars.datatypes.Int64'>" => ArrowDataType::Int64,
        "<class 'pypolars.datatypes.Float32'>" => ArrowDataType::Float32,
        "<class 'pypolars.datatypes.Float64'>" => ArrowDataType::Float64,
        "<class 'pypolars.datatypes.Boolean'>" => ArrowDataType::Boolean,
        "<class 'pypolars.datatypes.Utf8'>" => ArrowDataType::Utf8,
        "<class 'pypolars.datatypes.Date32'>" => ArrowDataType::Date32(DateUnit::Day),
        "<class 'pypolars.datatypes.Date64'>" => ArrowDataType::Date64(DateUnit::Millisecond),
        tp => panic!(format!("Type {} not implemented in str_to_arrow_type", tp)),
    }
}

impl From<dsl::Expr> for PyExpr {
    fn from(expr: dsl::Expr) -> Self {
        PyExpr { inner: expr }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct When {
    predicate: PyExpr,
}

#[pyclass]
#[derive(Clone)]
pub struct WhenThen {
    predicate: PyExpr,
    then: PyExpr,
}

#[pymethods]
impl When {
    pub fn then(&self, expr: PyExpr) -> WhenThen {
        WhenThen {
            predicate: self.predicate.clone(),
            then: expr,
        }
    }
}

#[pymethods]
impl WhenThen {
    pub fn otherwise(&self, expr: PyExpr) -> PyExpr {
        dsl::ternary_expr(
            self.predicate.inner.clone(),
            self.then.inner.clone(),
            expr.inner,
        )
        .into()
    }
}

pub fn when(predicate: PyExpr) -> When {
    When { predicate }
}

pub fn col(name: &str) -> PyExpr {
    dsl::col(name).into()
}

pub fn binary_expr(l: PyExpr, op: u8, r: PyExpr) -> PyExpr {
    let left = l.inner;
    let right = r.inner;

    let op = dsl::Operator::from(op);
    dsl::binary_expr(left, op, right).into()
}

pub fn lit(value: &PyAny) -> PyExpr {
    if let Ok(int) = value.downcast::<PyInt>() {
        let val = int.extract::<i64>().unwrap();
        dsl::lit(val).into()
    } else if let Ok(float) = value.downcast::<PyFloat>() {
        let val = float.extract::<f64>().unwrap();
        dsl::lit(val).into()
    } else {
        panic!("could not convert type")
    }
}
