#![allow(clippy::nonstandard_macro_braces)] // needed because clippy does not understand proc macro of pyo3
#![allow(clippy::transmute_undefined_repr)]
#[macro_use]
extern crate polars;
extern crate core;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::lazy::dsl::PyExpr;
use crate::{
    dataframe::PyDataFrame,
    file::EitherRustPythonFile,
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
mod list_construction;
pub mod npy;
pub mod prelude;
pub mod series;
pub mod utils;

use crate::conversion::{get_df, get_lf, get_pyseq, get_series, Wrap};
use crate::error::{
    ArrowErrorException, ComputeError, NoDataError, NotFoundError, PyPolarsErr, SchemaError,
};
use crate::file::get_either_file;
use crate::prelude::{ClosedWindow, DataType, Duration, PyDataType};
use dsl::ToExprs;
use mimalloc::MiMalloc;
use polars::functions::{diag_concat_df, hor_concat_df};
use polars_core::datatypes::TimeUnit;
use polars_core::export::arrow::io::ipc::read::read_file_metadata;
use polars_core::prelude::IntoSeries;
use pyo3::types::PyDict;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[pyfunction]
fn col(name: &str) -> dsl::PyExpr {
    dsl::col(name)
}

#[pyfunction]
fn count() -> dsl::PyExpr {
    dsl::count()
}

#[pyfunction]
fn first() -> dsl::PyExpr {
    dsl::first()
}

#[pyfunction]
fn last() -> dsl::PyExpr {
    dsl::last()
}

#[pyfunction]
fn cols(names: Vec<String>) -> dsl::PyExpr {
    dsl::cols(names)
}

#[pyfunction]
fn dtype_cols(dtypes: &PyAny) -> PyResult<dsl::PyExpr> {
    let (seq, len) = get_pyseq(dtypes)?;
    let iter = seq.iter()?;

    let mut dtypes = Vec::with_capacity(len);

    for res in iter {
        let item = res?;
        let pydt = item.extract::<PyDataType>()?;
        let dt: DataType = pydt.into();
        dtypes.push(dt)
    }
    Ok(dsl::dtype_cols(dtypes))
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
pub fn arange(low: PyExpr, high: PyExpr, step: usize) -> PyExpr {
    polars::lazy::dsl::arange(low.inner, high.inner, step).into()
}

#[pyfunction]
fn binary_function(
    a: dsl::PyExpr,
    b: dsl::PyExpr,
    lambda: PyObject,
    output_type: Option<Wrap<DataType>>,
) -> dsl::PyExpr {
    lazy::binary_function(a, b, lambda, output_type.map(|dt| dt.0))
}

#[pyfunction]
fn pearson_corr(a: dsl::PyExpr, b: dsl::PyExpr) -> dsl::PyExpr {
    polars::lazy::dsl::pearson_corr(a.inner, b.inner).into()
}

#[pyfunction]
fn spearman_rank_corr(a: dsl::PyExpr, b: dsl::PyExpr) -> dsl::PyExpr {
    polars::lazy::dsl::spearman_rank_corr(a.inner, b.inner).into()
}

#[pyfunction]
fn cov(a: dsl::PyExpr, b: dsl::PyExpr) -> dsl::PyExpr {
    polars::lazy::dsl::cov(a.inner, b.inner).into()
}

#[pyfunction]
fn argsort_by(by: Vec<dsl::PyExpr>, reverse: Vec<bool>) -> dsl::PyExpr {
    let by = by
        .into_iter()
        .map(|e| e.inner)
        .collect::<Vec<polars::lazy::dsl::Expr>>();
    polars::lazy::dsl::argsort_by(by, &reverse).into()
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
fn concat_str(s: Vec<dsl::PyExpr>, sep: &str) -> dsl::PyExpr {
    let s = s.into_iter().map(|e| e.inner).collect();
    polars::lazy::dsl::concat_str(s, sep).into()
}

#[pyfunction]
fn concat_lst(s: Vec<dsl::PyExpr>) -> dsl::PyExpr {
    let s = s.into_iter().map(|e| e.inner).collect();
    polars::lazy::dsl::concat_lst(s).into()
}

#[pyfunction]
fn py_datetime(
    year: dsl::PyExpr,
    month: dsl::PyExpr,
    day: dsl::PyExpr,
    hour: Option<dsl::PyExpr>,
    minute: Option<dsl::PyExpr>,
    second: Option<dsl::PyExpr>,
    millisecond: Option<dsl::PyExpr>,
) -> dsl::PyExpr {
    let hour = hour.map(|e| e.inner);
    let minute = minute.map(|e| e.inner);
    let second = second.map(|e| e.inner);
    let millisecond = millisecond.map(|e| e.inner);
    polars::lazy::dsl::datetime(
        year.inner,
        month.inner,
        day.inner,
        hour,
        minute,
        second,
        millisecond,
    )
    .into()
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
        df.vstack_mut(&other).map_err(PyPolarsErr::from)?;
    }
    Ok(df.into())
}

#[pyfunction]
fn concat_lf(lfs: &PyAny, rechunk: bool) -> PyResult<PyLazyFrame> {
    let (seq, len) = get_pyseq(lfs)?;
    let mut lfs = Vec::with_capacity(len);

    for res in seq.iter()? {
        let item = res?;
        let lf = get_lf(item)?;
        lfs.push(lf);
    }

    let lf = polars::lazy::dsl::concat(lfs, rechunk).map_err(PyPolarsErr::from)?;
    Ok(lf.into())
}

#[pyfunction]
fn py_diag_concat_df(dfs: &PyAny) -> PyResult<PyDataFrame> {
    let (seq, _len) = get_pyseq(dfs)?;
    let iter = seq.iter()?;

    let dfs = iter
        .map(|item| {
            let item = item?;
            get_df(item)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let df = diag_concat_df(&dfs).map_err(PyPolarsErr::from)?;
    Ok(df.into())
}

#[pyfunction]
fn py_hor_concat_df(dfs: &PyAny) -> PyResult<PyDataFrame> {
    let (seq, _len) = get_pyseq(dfs)?;
    let iter = seq.iter()?;

    let dfs = iter
        .map(|item| {
            let item = item?;
            get_df(item)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let df = hor_concat_df(&dfs).map_err(PyPolarsErr::from)?;
    Ok(df.into())
}

#[pyfunction]
fn concat_series(series: &PyAny) -> PyResult<PySeries> {
    let (seq, _len) = get_pyseq(series)?;
    let mut iter = seq.iter()?;
    let first = iter.next().unwrap()?;

    let mut s = get_series(first)?;

    for res in iter {
        let item = res?;
        let item = get_series(item)?;
        s.append(&item).map_err(PyPolarsErr::from)?;
    }
    Ok(s.into())
}

#[pyfunction]
fn ipc_schema(py: Python, py_f: PyObject) -> PyResult<PyObject> {
    let metadata = match get_either_file(py_f, false)? {
        EitherRustPythonFile::Rust(mut r) => {
            read_file_metadata(&mut r).map_err(PyPolarsErr::from)?
        }
        EitherRustPythonFile::Py(mut r) => read_file_metadata(&mut r).map_err(PyPolarsErr::from)?,
    };

    let dict = PyDict::new(py);
    for field in metadata.schema.fields {
        let dt: Wrap<DataType> = Wrap((&field.data_type).into());
        dict.set_item(field.name, dt.to_object(py))?;
    }
    Ok(dict.to_object(py))
}

#[pyfunction]
fn collect_all(lfs: Vec<PyLazyFrame>, py: Python) -> PyResult<Vec<PyDataFrame>> {
    use polars_core::utils::rayon::prelude::*;

    let out = py.allow_threads(|| {
        polars_core::POOL.install(|| {
            lfs.par_iter()
                .map(|lf| {
                    let df = lf.ldf.clone().collect()?;
                    Ok(PyDataFrame::new(df))
                })
                .collect::<polars_core::error::Result<Vec<_>>>()
                .map_err(PyPolarsErr::from)
        })
    });

    Ok(out?)
}

#[pyfunction]
pub fn map_mul(
    py: Python,
    pyexpr: Vec<PyExpr>,
    lambda: PyObject,
    output_type: &PyAny,
    apply_groups: bool,
) -> PyExpr {
    lazy::map_mul(&pyexpr, py, lambda, output_type, apply_groups)
}

#[pyfunction]
fn py_date_range(
    start: i64,
    stop: i64,
    every: &str,
    closed: Wrap<ClosedWindow>,
    name: &str,
    tu: &str,
) -> PySeries {
    let tu = match tu {
        "ns" => TimeUnit::Nanoseconds,
        "ms" => TimeUnit::Milliseconds,
        _ => panic!("{}", "expected one of {'ns', 'ms'}"),
    };
    polars::time::date_range_impl(name, start, stop, Duration::parse(every), closed.0, tu)
        .into_series()
        .into()
}

#[pyfunction]
fn min_exprs(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    polars::lazy::dsl::min_exprs(exprs).into()
}

#[pyfunction]
fn max_exprs(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    polars::lazy::dsl::max_exprs(exprs).into()
}

#[pyfunction]
fn as_struct(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    polars::lazy::dsl::as_struct(&exprs).into()
}

#[pymodule]
fn polars(py: Python, m: &PyModule) -> PyResult<()> {
    m.add("NotFoundError", py.get_type::<NotFoundError>())
        .unwrap();
    m.add("NoDataError", py.get_type::<NoDataError>()).unwrap();
    m.add("ComputeError", py.get_type::<ComputeError>())
        .unwrap();
    m.add("ShapeError", py.get_type::<crate::error::ShapeError>())
        .unwrap();
    m.add("SchemaError", py.get_type::<SchemaError>()).unwrap();
    m.add("ArrowError", py.get_type::<ArrowErrorException>())
        .unwrap();
    m.add_class::<PySeries>().unwrap();
    m.add_class::<PyDataFrame>().unwrap();
    m.add_class::<PyLazyFrame>().unwrap();
    m.add_class::<PyLazyGroupBy>().unwrap();
    m.add_class::<dsl::PyExpr>().unwrap();
    m.add_wrapped(wrap_pyfunction!(col)).unwrap();
    m.add_wrapped(wrap_pyfunction!(count)).unwrap();
    m.add_wrapped(wrap_pyfunction!(first)).unwrap();
    m.add_wrapped(wrap_pyfunction!(last)).unwrap();
    m.add_wrapped(wrap_pyfunction!(cols)).unwrap();
    m.add_wrapped(wrap_pyfunction!(dtype_cols)).unwrap();
    m.add_wrapped(wrap_pyfunction!(lit)).unwrap();
    m.add_wrapped(wrap_pyfunction!(fold)).unwrap();
    m.add_wrapped(wrap_pyfunction!(binary_expr)).unwrap();
    m.add_wrapped(wrap_pyfunction!(arange)).unwrap();
    m.add_wrapped(wrap_pyfunction!(binary_function)).unwrap();
    m.add_wrapped(wrap_pyfunction!(pearson_corr)).unwrap();
    m.add_wrapped(wrap_pyfunction!(cov)).unwrap();
    m.add_wrapped(wrap_pyfunction!(argsort_by)).unwrap();
    m.add_wrapped(wrap_pyfunction!(when)).unwrap();
    m.add_wrapped(wrap_pyfunction!(version)).unwrap();
    m.add_wrapped(wrap_pyfunction!(toggle_string_cache))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(concat_str)).unwrap();
    m.add_wrapped(wrap_pyfunction!(concat_lst)).unwrap();
    m.add_wrapped(wrap_pyfunction!(concat_df)).unwrap();
    m.add_wrapped(wrap_pyfunction!(concat_lf)).unwrap();
    m.add_wrapped(wrap_pyfunction!(concat_series)).unwrap();
    m.add_wrapped(wrap_pyfunction!(ipc_schema)).unwrap();
    m.add_wrapped(wrap_pyfunction!(collect_all)).unwrap();
    m.add_wrapped(wrap_pyfunction!(spearman_rank_corr)).unwrap();
    m.add_wrapped(wrap_pyfunction!(map_mul)).unwrap();
    m.add_wrapped(wrap_pyfunction!(py_diag_concat_df)).unwrap();
    m.add_wrapped(wrap_pyfunction!(py_hor_concat_df)).unwrap();
    m.add_wrapped(wrap_pyfunction!(py_datetime)).unwrap();
    m.add_wrapped(wrap_pyfunction!(py_date_range)).unwrap();
    m.add_wrapped(wrap_pyfunction!(min_exprs)).unwrap();
    m.add_wrapped(wrap_pyfunction!(max_exprs)).unwrap();
    m.add_wrapped(wrap_pyfunction!(as_struct)).unwrap();
    Ok(())
}
