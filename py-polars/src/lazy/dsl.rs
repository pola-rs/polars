use crate::series::PySeries;
use crate::utils::str_to_arrow_type;
use polars::lazy::dsl;
use polars::lazy::dsl::Operator;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyInt, PyString};
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
    pub fn min(&self) -> PyExpr {
        self.clone().inner.min().into()
    }
    #[text_signature = "($self)"]
    pub fn max(&self) -> PyExpr {
        self.clone().inner.max().into()
    }
    #[text_signature = "($self)"]
    pub fn mean(&self) -> PyExpr {
        self.clone().inner.mean().into()
    }
    #[text_signature = "($self)"]
    pub fn median(&self) -> PyExpr {
        self.clone().inner.median().into()
    }
    #[text_signature = "($self)"]
    pub fn sum(&self) -> PyExpr {
        self.clone().inner.sum().into()
    }
    #[text_signature = "($self)"]
    pub fn n_unique(&self) -> PyExpr {
        self.clone().inner.n_unique().into()
    }
    #[text_signature = "($self)"]
    pub fn first(&self) -> PyExpr {
        self.clone().inner.first().into()
    }
    #[text_signature = "($self)"]
    pub fn last(&self) -> PyExpr {
        self.clone().inner.last().into()
    }
    pub fn list(&self) -> PyExpr {
        self.clone().inner.list().into()
    }
    #[text_signature = "($self, quantile)"]
    pub fn quantile(&self, quantile: f64) -> PyExpr {
        self.clone().inner.quantile(quantile).into()
    }
    pub fn agg_groups(&self) -> PyExpr {
        self.clone().inner.agg_groups().into()
    }
    pub fn count(&self) -> PyExpr {
        self.clone().inner.count().into()
    }
    #[text_signature = "($self, data_type)"]
    pub fn cast(&self, data_type: &PyAny) -> PyExpr {
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
    pub fn fill_none(&self, expr: PyExpr) -> PyResult<PyExpr> {
        Ok(self.clone().inner.fill_none(expr.inner).into())
    }
    pub fn reverse(&self) -> PyExpr {
        self.clone().inner.reverse().into()
    }
    pub fn std(&self) -> PyExpr {
        self.clone().inner.std().into()
    }
    pub fn var(&self) -> PyExpr {
        self.clone().inner.var().into()
    }
    pub fn is_unique(&self) -> PyExpr {
        self.clone().inner.is_unique().into()
    }

    pub fn explode(&self) -> PyExpr {
        self.clone().inner.explode().into()
    }

    pub fn take_every(&self, n: usize) -> PyExpr {
        self.clone()
            .inner
            .map(move |s: Series| Ok(s.take_every(n)), None)
            .into()
    }
    pub fn tail(&self, n: Option<usize>) -> PyExpr {
        self.clone()
            .inner
            .map(move |s: Series| Ok(s.tail(n)), None)
            .into()
    }
    pub fn head(&self, n: Option<usize>) -> PyExpr {
        self.clone()
            .inner
            .map(move |s: Series| Ok(s.head(n)), None)
            .into()
    }

    pub fn is_duplicated(&self) -> PyExpr {
        self.clone().inner.is_duplicated().into()
    }

    pub fn over(&self, partition_by: PyExpr) -> PyExpr {
        self.clone().inner.over(partition_by.inner).into()
    }

    pub fn _and(&self, expr: PyExpr) -> PyExpr {
        self.clone().inner.and(expr.inner).into()
    }

    pub fn str_parse_date32(&self, fmt: Option<String>) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            ca.as_date32(fmt.as_ref().map(|s| s.as_str()))
                .map(|ca| ca.into_series())
        };
        self.clone()
            .inner
            .map(function, Some(ArrowDataType::Date32(DateUnit::Day)))
            .into()
    }

    pub fn str_parse_date64(&self, fmt: Option<String>) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            ca.as_date64(fmt.as_ref().map(|s| s.as_str()))
                .map(|ca| ca.into_series())
        };
        self.clone()
            .inner
            .map(function, Some(ArrowDataType::Date64(DateUnit::Millisecond)))
            .into()
    }

    pub fn str_to_uppercase(&self) -> PyExpr {
        let function = |s: Series| {
            let ca = s.utf8()?;
            Ok(ca.to_uppercase().into_series())
        };
        self.clone()
            .inner
            .map(function, Some(ArrowDataType::UInt32))
            .into()
    }

    pub fn str_to_lowercase(&self) -> PyExpr {
        let function = |s: Series| {
            let ca = s.utf8()?;
            Ok(ca.to_lowercase().into_series())
        };
        self.clone()
            .inner
            .map(function, Some(ArrowDataType::UInt32))
            .into()
    }

    pub fn str_lengths(&self) -> PyExpr {
        let function = |s: Series| {
            let ca = s.utf8()?;
            Ok(ca.str_lengths().into_series())
        };
        self.clone()
            .inner
            .map(function, Some(ArrowDataType::UInt32))
            .into()
    }

    pub fn str_replace(&self, pat: String, val: String) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            match ca.replace(&pat, &val) {
                Ok(ca) => Ok(ca.into_series()),
                Err(e) => Err(PolarsError::Other(format!("{:?}", e).into())),
            }
        };
        self.clone().inner.map(function, None).into()
    }

    pub fn str_replace_all(&self, pat: String, val: String) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            match ca.replace_all(&pat, &val) {
                Ok(ca) => Ok(ca.into_series()),
                Err(e) => Err(PolarsError::Other(format!("{:?}", e).into())),
            }
        };
        self.clone().inner.map(function, None).into()
    }

    pub fn str_contains(&self, pat: String) -> PyExpr {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            match ca.contains(&pat) {
                Ok(ca) => Ok(ca.into_series()),
                Err(e) => Err(PolarsError::Other(format!("{:?}", e).into())),
            }
        };
        self.clone()
            .inner
            .map(function, Some(ArrowDataType::Boolean))
            .into()
    }

    pub fn datetime_str_fmt(&self, fmt: String) -> PyExpr {
        let function = move |s: Series| s.datetime_str_fmt(&fmt);
        self.clone()
            .inner
            .map(function, Some(ArrowDataType::Utf8))
            .into()
    }

    pub fn year(&self) -> PyExpr {
        let function = move |s: Series| s.year();
        self.clone()
            .inner
            .map(function, Some(ArrowDataType::UInt32))
            .into()
    }
    pub fn month(&self) -> PyExpr {
        let function = move |s: Series| s.month();
        self.clone()
            .inner
            .map(function, Some(ArrowDataType::UInt32))
            .into()
    }
    pub fn day(&self) -> PyExpr {
        let function = move |s: Series| s.day();
        self.clone()
            .inner
            .map(function, Some(ArrowDataType::UInt32))
            .into()
    }
    pub fn ordinal_day(&self) -> PyExpr {
        let function = move |s: Series| s.ordinal_day();
        self.clone()
            .inner
            .map(function, Some(ArrowDataType::UInt32))
            .into()
    }
    pub fn hour(&self) -> PyExpr {
        let function = move |s: Series| s.hour();
        self.clone()
            .inner
            .map(function, Some(ArrowDataType::UInt32))
            .into()
    }
    pub fn minute(&self) -> PyExpr {
        let function = move |s: Series| s.minute();
        self.clone()
            .inner
            .map(function, Some(ArrowDataType::UInt32))
            .into()
    }
    pub fn second(&self) -> PyExpr {
        let function = move |s: Series| s.second();
        self.clone()
            .inner
            .map(function, Some(ArrowDataType::UInt32))
            .into()
    }
    pub fn nanosecond(&self) -> PyExpr {
        let function = move |s: Series| s.nanosecond();
        self.clone()
            .inner
            .map(function, Some(ArrowDataType::UInt32))
            .into()
    }

    pub fn map(&self, lambda: PyObject, output_type: &PyAny) -> PyExpr {
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
            // call the lambda and get a python side Series wrapper
            let result_series_wrapper = match lambda.call1(py, (python_series_wrapper,)) {
                Ok(pyobj) => pyobj,
                Err(e) => panic!(format!("UDF failed: {}", e.pvalue(py).to_string())),
            };
            // unpack the wrapper in a PySeries
            let py_pyseries = result_series_wrapper.getattr(py, "_s").expect(
                "Could net get series attribute '_s'. Make sure that you return a Series object.",
            );
            // Downcast to Rust
            let pyseries = py_pyseries.extract::<PySeries>(py).unwrap();
            // Finally get the actual Series
            Ok(pyseries.series)
        };

        self.clone().inner.map(function, output_type).into()
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
    } else if let Ok(pystr) = value.downcast::<PyString>() {
        dsl::lit(
            pystr
                .to_str()
                .expect("could not transform Python string to Rust Unicode"),
        )
        .into()
    } else {
        panic!(format!("could not convert value {:?} as a Literal", value))
    }
}
