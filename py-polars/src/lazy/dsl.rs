mod binary;
mod categorical;
mod datetime;
mod general;
mod list;
#[cfg(feature = "meta")]
mod meta;
mod string;
mod r#struct;

use polars::lazy::dsl;
use polars::lazy::dsl::Operator;
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyFloat, PyInt, PyString};

use super::apply::*;
use crate::conversion::Wrap;
use crate::lazy::utils::py_exprs_to_exprs;
use crate::prelude::ObjectValue;
use crate::series::PySeries;

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyExpr {
    pub inner: dsl::Expr,
}

impl From<dsl::Expr> for PyExpr {
    fn from(expr: dsl::Expr) -> Self {
        PyExpr { inner: expr }
    }
}

pub fn binary_expr(l: PyExpr, op: u8, r: PyExpr) -> PyExpr {
    let left = l.inner;
    let right = r.inner;

    let op = match op {
        0 => Operator::Eq,
        1 => Operator::NotEq,
        2 => Operator::Lt,
        3 => Operator::LtEq,
        4 => Operator::Gt,
        5 => Operator::GtEq,
        6 => Operator::Plus,
        7 => Operator::Minus,
        8 => Operator::Multiply,
        9 => Operator::Divide,
        10 => Operator::Modulus,
        11 => Operator::And,
        12 => Operator::Or,
        _ => panic!("not an operator"),
    };

    dsl::binary_expr(left, op, right).into()
}

pub fn fold(acc: PyExpr, lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = py_exprs_to_exprs(exprs);

    let func = move |a: Series, b: Series| binary_lambda(&lambda, a, b);
    polars::lazy::dsl::fold_exprs(acc.inner, func, exprs).into()
}

pub fn reduce(lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = py_exprs_to_exprs(exprs);

    let func = move |a: Series, b: Series| binary_lambda(&lambda, a, b);
    polars::lazy::dsl::reduce_exprs(func, exprs).into()
}

pub fn lit(value: &PyAny, allow_object: bool) -> PyResult<PyExpr> {
    if let Ok(true) = value.is_instance_of::<PyBool>() {
        let val = value.extract::<bool>().unwrap();
        Ok(dsl::lit(val).into())
    } else if let Ok(int) = value.downcast::<PyInt>() {
        match int.extract::<i64>() {
            Ok(val) => {
                if val >= 0 && val < i32::MAX as i64 || val <= 0 && val > i32::MIN as i64 {
                    Ok(dsl::lit(val as i32).into())
                } else {
                    Ok(dsl::lit(val).into())
                }
            }
            _ => {
                let val = int.extract::<u64>().unwrap();
                Ok(dsl::lit(val).into())
            }
        }
    } else if let Ok(float) = value.downcast::<PyFloat>() {
        let val = float.extract::<f64>().unwrap();
        Ok(dsl::lit(val).into())
    } else if let Ok(pystr) = value.downcast::<PyString>() {
        Ok(dsl::lit(
            pystr
                .to_str()
                .expect("could not transform Python string to Rust Unicode"),
        )
        .into())
    } else if let Ok(series) = value.extract::<PySeries>() {
        Ok(dsl::lit(series.series).into())
    } else if value.is_none() {
        Ok(dsl::lit(Null {}).into())
    } else if let Ok(value) = value.downcast::<PyBytes>() {
        Ok(dsl::lit(value.as_bytes()).into())
    } else if allow_object {
        let s = Python::with_gil(|py| {
            PySeries::new_object("", vec![ObjectValue::from(value.into_py(py))], false).series
        });
        Ok(dsl::lit(s).into())
    } else {
        Err(PyValueError::new_err(format!(
            "could not convert value {:?} as a Literal",
            value.str()?
        )))
    }
}

pub fn range(low: i64, high: i64, dtype: Wrap<DataType>) -> PyExpr {
    match dtype.0 {
        DataType::Int32 => dsl::range(low as i32, high as i32).into(),
        DataType::UInt32 => dsl::range(low as u32, high as u32).into(),
        _ => dsl::range(low, high).into(),
    }
}
