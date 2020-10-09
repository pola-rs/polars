#[macro_use]
extern crate polars;
use crate::{
    dataframe::PyDataFrame,
    lazy::dataframe::{PyLazyFrame, PyLazyGroupBy},
    series::PySeries,
};
use polars::datatypes::ArrowDataType;
use polars::lazy::dsl;
use polars::lazy::dsl::Operator;
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyInt};
use pyo3::{wrap_pyfunction, PyNumberProtocol};

pub mod dataframe;
pub mod datatypes;
pub mod dispatch;
pub mod error;
pub mod file;
pub mod lazy;
pub mod npy;
pub mod series;

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
    pub fn not(&self) -> PyExpr {
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
    #[text_signature = "($self, quantile)"]
    pub fn agg_quantile(&self, quantile: f64) -> PyExpr {
        self.clone().inner.agg_quantile(quantile).into()
    }
    #[text_signature = "($self)"]
    pub fn agg_groups(&self) -> PyExpr {
        self.clone().inner.agg_groups().into()
    }

    #[text_signature = "($self, data_type)"]
    pub fn cast(&self, data_type: &str) -> PyExpr {
        // TODO! accept the DataType objects.
        let expr = match data_type {
            "u8" => self.inner.clone().cast(ArrowDataType::UInt8),
            "u16" => self.inner.clone().cast(ArrowDataType::UInt16),
            "u32" => self.inner.clone().cast(ArrowDataType::UInt32),
            "u64" => self.inner.clone().cast(ArrowDataType::UInt64),
            "i8" => self.inner.clone().cast(ArrowDataType::Int8),
            "i16" => self.inner.clone().cast(ArrowDataType::Int16),
            "i32" => self.inner.clone().cast(ArrowDataType::Int32),
            "i64" => self.inner.clone().cast(ArrowDataType::Int64),
            "f32" => self.inner.clone().cast(ArrowDataType::Float32),
            "f64" => self.inner.clone().cast(ArrowDataType::Float64),
            "bool" => self.inner.clone().cast(ArrowDataType::Boolean),
            "utf8" => self.inner.clone().cast(ArrowDataType::Utf8),
            _ => todo!(),
        };
        expr.into()
    }
    #[text_signature = "($self, reverse)"]
    pub fn sort(&self, reverse: bool) -> PyExpr {
        self.clone().inner.sort(reverse).into()
    }
}

impl From<dsl::Expr> for PyExpr {
    fn from(expr: dsl::Expr) -> Self {
        PyExpr { inner: expr }
    }
}

#[pyfunction]
pub fn col(name: &str) -> PyExpr {
    dsl::col(name).into()
}

#[pyfunction]
pub fn binary_expr(l: PyExpr, op: u8, r: PyExpr) -> PyExpr {
    let left = l.inner;
    let right = r.inner;

    let op = dsl::Operator::from(op);
    dsl::binary_expr(left, op, right).into()
}

#[pyfunction]
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

#[pymodule]
fn pypolars(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySeries>().unwrap();
    m.add_class::<PyDataFrame>().unwrap();
    m.add_class::<PyLazyFrame>().unwrap();
    m.add_class::<PyLazyGroupBy>().unwrap();
    m.add_class::<PyExpr>().unwrap();
    m.add_wrapped(wrap_pyfunction!(col)).unwrap();
    m.add_wrapped(wrap_pyfunction!(lit)).unwrap();
    m.add_wrapped(wrap_pyfunction!(binary_expr)).unwrap();
    Ok(())
}
