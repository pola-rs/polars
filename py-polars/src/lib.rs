#[macro_use]
extern crate polars;
use crate::lazy::dsl::PyExpr;
use crate::{
    dataframe::PyDataFrame,
    lazy::{
        dataframe::{PyLazyFrame, PyLazyGroupBy},
        dsl,
    },
    series::PySeries,
};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub mod dataframe;
pub mod datatypes;
pub mod dispatch;
pub mod error;
pub mod file;
pub mod lazy;
pub mod npy;
pub mod series;

#[pyfunction]
fn col(name: &str) -> dsl::PyExpr {
    dsl::col(name)
}

#[pyfunction]
fn lit(value: &PyAny) -> dsl::PyExpr {
    dsl::lit(value)
}

#[pyfunction]
fn binary_expr(l: dsl::PyExpr, op: u8, r: dsl::PyExpr) -> dsl::PyExpr {
    dsl::binary_expr(l, op, r)
}

#[pyfunction]
fn when(predicate: PyExpr) -> dsl::When {
    dsl::when(predicate)
}

#[pymodule]
fn pypolars(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySeries>().unwrap();
    m.add_class::<PyDataFrame>().unwrap();
    m.add_class::<PyLazyFrame>().unwrap();
    m.add_class::<PyLazyGroupBy>().unwrap();
    m.add_class::<dsl::PyExpr>().unwrap();
    m.add_wrapped(wrap_pyfunction!(col)).unwrap();
    m.add_wrapped(wrap_pyfunction!(lit)).unwrap();
    m.add_wrapped(wrap_pyfunction!(binary_expr)).unwrap();
    m.add_wrapped(wrap_pyfunction!(when)).unwrap();
    Ok(())
}
