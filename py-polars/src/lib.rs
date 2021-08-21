#![allow(clippy::nonstandard_macro_braces)] // needed because clippy does not understand proc macro of pyo3
#[macro_use]
extern crate polars;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::lazy::dsl::PyExpr;
use crate::{
    dataframe::PyDataFrame,
    lazy::{
        dataframe::{PyLazyFrame, PyLazyGroupBy},
        dsl,
    },
    series::PySeries,
};

pub mod apply;
pub mod arrow_interop;
pub mod conversion;
pub mod dataframe;
pub mod datatypes;
pub mod error;
pub mod file;
pub mod lazy;
pub mod npy;
pub mod prelude;
pub mod series;
pub mod utils;

use crate::conversion::{get_df, get_pyseq};
use crate::error::PyPolarsEr;
use crate::utils::str_to_polarstype;
use mimalloc::MiMalloc;
use polars::prelude::*;
use std::iter::FromIterator;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[pyfunction]
fn col(name: &str) -> dsl::PyExpr {
    dsl::col(name)
}

#[pyfunction]
fn cols(names: Vec<String>) -> dsl::PyExpr {
    dsl::cols(names)
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
pub fn fold(acc: PyExpr, lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    dsl::fold(acc, lambda, exprs)
}

#[pyfunction]
fn binary_function(
    a: dsl::PyExpr,
    b: dsl::PyExpr,
    lambda: PyObject,
    output_type: &PyAny,
) -> dsl::PyExpr {
    dsl::binary_function(a, b, lambda, output_type)
}

#[pyfunction]
fn pearson_corr(a: dsl::PyExpr, b: dsl::PyExpr) -> dsl::PyExpr {
    polars::lazy::functions::pearson_corr(a.inner, b.inner).into()
}

#[pyfunction]
fn cov(a: dsl::PyExpr, b: dsl::PyExpr) -> dsl::PyExpr {
    polars::lazy::functions::cov(a.inner, b.inner).into()
}

#[pyfunction]
fn argsort_by(by: Vec<dsl::PyExpr>, reverse: Vec<bool>) -> dsl::PyExpr {
    let by = by.into_iter().map(|e| e.inner).collect();
    polars::lazy::functions::argsort_by(by, &reverse).into()
}

#[pyfunction]
fn when(predicate: PyExpr) -> dsl::When {
    dsl::when(predicate)
}

const VERSION: &str = env!("CARGO_PKG_VERSION");
#[pyfunction]
fn version() -> &'static str {
    VERSION
}

#[pyfunction]
fn toggle_string_cache(toggle: bool) {
    polars::toggle_string_cache(toggle)
}

#[pyfunction]
fn concat_str(s: Vec<dsl::PyExpr>, delimiter: &str) -> dsl::PyExpr {
    let s = s.into_iter().map(|e| e.inner).collect();
    polars::lazy::functions::concat_str(s, delimiter).into()
}

#[pyfunction]
fn concat_df(dfs: &PyAny) -> PyResult<PyDataFrame> {
    let (seq, _len) = get_pyseq(dfs)?;
    let mut iter = seq.iter()?;
    let first = iter.next().unwrap()?;

    let mut df = get_df(first)?;

    for res in iter {
        let item = res?;
        let other = get_df(item)?;
        df.vstack_mut(&other).map_err(PyPolarsEr::from)?;
    }
    Ok(df.into())
}

#[pyfunction]
fn series_from_range(low: i64, high: i64, step_by: usize, dtype: &PyAny) -> PySeries {
    let str_repr = dtype.str().unwrap().to_str().unwrap();
    let dtype = str_to_polarstype(str_repr);

    if step_by == 1 {
        match dtype {
            DataType::UInt32 => Series::from_iter((low as u32)..(high as u32)).into(),
            DataType::Int32 => Series::from_iter((low as i32)..(high as i32)).into(),
            DataType::Int64 => Series::from_iter((low as i64)..(high as i64)).into(),
            _ => unimplemented!(),
        }
    } else {
        match dtype {
            DataType::UInt32 => {
                Series::from_iter(((low as u32)..(high as u32)).step_by(step_by)).into()
            }
            DataType::Int32 => {
                Series::from_iter(((low as i32)..(high as i32)).step_by(step_by)).into()
            }
            DataType::Int64 => {
                Series::from_iter(((low as i64)..(high as i64)).step_by(step_by)).into()
            }
            _ => unimplemented!(),
        }
    }
}

#[pymodule]
fn polars(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySeries>().unwrap();
    m.add_class::<PyDataFrame>().unwrap();
    m.add_class::<PyLazyFrame>().unwrap();
    m.add_class::<PyLazyGroupBy>().unwrap();
    m.add_class::<dsl::PyExpr>().unwrap();
    m.add_wrapped(wrap_pyfunction!(col)).unwrap();
    m.add_wrapped(wrap_pyfunction!(cols)).unwrap();
    m.add_wrapped(wrap_pyfunction!(lit)).unwrap();
    m.add_wrapped(wrap_pyfunction!(fold)).unwrap();
    m.add_wrapped(wrap_pyfunction!(binary_expr)).unwrap();
    m.add_wrapped(wrap_pyfunction!(binary_function)).unwrap();
    m.add_wrapped(wrap_pyfunction!(pearson_corr)).unwrap();
    m.add_wrapped(wrap_pyfunction!(cov)).unwrap();
    m.add_wrapped(wrap_pyfunction!(argsort_by)).unwrap();
    m.add_wrapped(wrap_pyfunction!(when)).unwrap();
    m.add_wrapped(wrap_pyfunction!(version)).unwrap();
    m.add_wrapped(wrap_pyfunction!(toggle_string_cache))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(series_from_range)).unwrap();
    m.add_wrapped(wrap_pyfunction!(concat_str)).unwrap();
    m.add_wrapped(wrap_pyfunction!(concat_df)).unwrap();
    Ok(())
}
