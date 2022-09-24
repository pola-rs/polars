#![feature(vec_into_raw_parts)]
#![allow(clippy::nonstandard_macro_braces)] // needed because clippy does not understand proc macro of pyo3
#![allow(clippy::transmute_undefined_repr)]
extern crate core;
extern crate polars;

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
pub(crate) mod py_modules;
pub mod series;
mod set;
#[cfg(feature = "polars-sql")]
mod sql;
pub mod utils;

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;
use lazy::ToExprs;
#[cfg(not(target_os = "linux"))]
use mimalloc::MiMalloc;
use polars::functions::{diag_concat_df, hor_concat_df};
use polars::prelude::Null;
use polars_core::datatypes::TimeUnit;
use polars_core::prelude::{DataFrame, IntoSeries, IDX_DTYPE};
use polars_core::POOL;
use pyo3::exceptions::PyValueError;
use pyo3::panic::PanicException;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyString};
use pyo3::wrap_pyfunction;

use crate::conversion::{get_df, get_lf, get_pyseq, get_series, Wrap};
use crate::dataframe::PyDataFrame;
use crate::error::{
    ArrowErrorException, ComputeError, DuplicateError, InvalidOperationError, NoDataError,
    NotFoundError, PyPolarsErr, SchemaError,
};
use crate::file::{get_either_file, EitherRustPythonFile};
use crate::lazy::dataframe::{PyLazyFrame, PyLazyGroupBy};
use crate::lazy::dsl;
use crate::lazy::dsl::PyExpr;
use crate::prelude::{
    vec_extract_wrapped, ClosedWindow, DataType, DatetimeArgs, Duration, DurationArgs,
};
use crate::series::PySeries;

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;

#[global_allocator]
#[cfg(not(target_os = "linux"))]
static ALLOC: MiMalloc = MiMalloc;

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
fn dtype_cols(dtypes: Vec<Wrap<DataType>>) -> PyResult<dsl::PyExpr> {
    let dtypes = vec_extract_wrapped(dtypes);
    Ok(dsl::dtype_cols(dtypes))
}

#[pyfunction]
fn dtype_str_repr(dtype: Wrap<DataType>) -> PyResult<String> {
    let dtype = dtype.0;
    Ok(dtype.to_string())
}

#[pyfunction]
fn lit(value: &PyAny, allow_object: bool) -> PyResult<dsl::PyExpr> {
    dsl::lit(value, allow_object)
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
fn arange(low: PyExpr, high: PyExpr, step: usize) -> PyExpr {
    polars::lazy::dsl::arange(low.inner, high.inner, step).into()
}

#[pyfunction]
fn repeat(value: &PyAny, n_times: PyExpr) -> PyResult<PyExpr> {
    if let Ok(true) = value.is_instance_of::<PyBool>() {
        let val = value.extract::<bool>().unwrap();
        Ok(polars::lazy::dsl::repeat(val, n_times.inner).into())
    } else if let Ok(int) = value.downcast::<PyInt>() {
        let val = int.extract::<i64>().unwrap();

        if val > 0 && val < i32::MAX as i64 || val < 0 && val > i32::MIN as i64 {
            Ok(polars::lazy::dsl::repeat(val as i32, n_times.inner).into())
        } else {
            Ok(polars::lazy::dsl::repeat(val, n_times.inner).into())
        }
    } else if let Ok(float) = value.downcast::<PyFloat>() {
        let val = float.extract::<f64>().unwrap();
        Ok(polars::lazy::dsl::repeat(val, n_times.inner).into())
    } else if let Ok(pystr) = value.downcast::<PyString>() {
        let val = pystr
            .to_str()
            .expect("could not transform Python string to Rust Unicode");
        Ok(polars::lazy::dsl::repeat(val, n_times.inner).into())
    } else if value.is_none() {
        Ok(polars::lazy::dsl::repeat(Null {}, n_times.inner).into())
    } else {
        Err(PyValueError::new_err(format!(
            "could not convert value {:?} as a Literal",
            value.str()?
        )))
    }
}

#[pyfunction]
fn pearson_corr(a: dsl::PyExpr, b: dsl::PyExpr, ddof: u8) -> dsl::PyExpr {
    polars::lazy::dsl::pearson_corr(a.inner, b.inner, ddof).into()
}

#[pyfunction]
fn spearman_rank_corr(
    a: dsl::PyExpr,
    b: dsl::PyExpr,
    ddof: u8,
    propagate_nans: bool,
) -> dsl::PyExpr {
    polars::lazy::dsl::spearman_rank_corr(a.inner, b.inner, ddof, propagate_nans).into()
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
fn using_string_cache() -> bool {
    polars::using_string_cache()
}

#[pyfunction]
fn concat_str(s: Vec<dsl::PyExpr>, sep: &str) -> dsl::PyExpr {
    let s = s.into_iter().map(|e| e.inner).collect::<Vec<_>>();
    polars::lazy::dsl::concat_str(s, sep).into()
}

#[pyfunction]
fn concat_lst(s: Vec<dsl::PyExpr>) -> dsl::PyExpr {
    let s = s.into_iter().map(|e| e.inner).collect::<Vec<_>>();
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

    let args = DatetimeArgs {
        year: year.inner,
        month: month.inner,
        day: day.inner,
        hour,
        minute,
        second,
        millisecond,
    };

    polars::lazy::dsl::datetime(args).into()
}

#[pyfunction]
fn py_duration(
    days: Option<PyExpr>,
    seconds: Option<PyExpr>,
    nanoseconds: Option<PyExpr>,
    milliseconds: Option<PyExpr>,
    minutes: Option<PyExpr>,
    hours: Option<PyExpr>,
    weeks: Option<PyExpr>,
) -> dsl::PyExpr {
    let args = DurationArgs {
        days: days.map(|e| e.inner),
        seconds: seconds.map(|e| e.inner),
        nanoseconds: nanoseconds.map(|e| e.inner),
        milliseconds: milliseconds.map(|e| e.inner),
        minutes: minutes.map(|e| e.inner),
        hours: hours.map(|e| e.inner),
        weeks: weeks.map(|e| e.inner),
    };

    polars::lazy::dsl::duration(args).into()
}

#[pyfunction]
fn concat_df(dfs: &PyAny, py: Python) -> PyResult<PyDataFrame> {
    use polars_core::error::PolarsResult;
    use polars_core::utils::rayon::prelude::*;

    let (seq, _len) = get_pyseq(dfs)?;
    let mut iter = seq.iter()?;
    let first = iter.next().unwrap()?;

    let first_rdf = get_df(first)?;
    let identity_df = first_rdf.slice(0, 0);

    let mut rdfs: Vec<PolarsResult<DataFrame>> = vec![Ok(first_rdf)];

    for item in iter {
        let rdf = get_df(item?)?;
        rdfs.push(Ok(rdf));
    }

    let identity = || Ok(identity_df.clone());

    let df = py
        .allow_threads(|| {
            polars_core::POOL.install(|| {
                rdfs.into_par_iter()
                    .fold(identity, |acc: PolarsResult<DataFrame>, df| {
                        let mut acc = acc?;
                        acc.vstack_mut(&df?)?;
                        Ok(acc)
                    })
                    .reduce(identity, |acc, df| {
                        let mut acc = acc?;
                        acc.vstack_mut(&df?)?;
                        Ok(acc)
                    })
            })
        })
        .map_err(PyPolarsErr::from)?;

    Ok(df.into())
}

#[pyfunction]
fn concat_lf(lfs: &PyAny, rechunk: bool, parallel: bool) -> PyResult<PyLazyFrame> {
    let (seq, len) = get_pyseq(lfs)?;
    let mut lfs = Vec::with_capacity(len);

    for res in seq.iter()? {
        let item = res?;
        let lf = get_lf(item)?;
        lfs.push(lf);
    }

    let lf = polars::lazy::dsl::concat(lfs, rechunk, parallel).map_err(PyPolarsErr::from)?;
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

#[cfg(feature = "ipc")]
#[pyfunction]
fn ipc_schema(py: Python, py_f: PyObject) -> PyResult<PyObject> {
    use polars_core::export::arrow::io::ipc::read::read_file_metadata;
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

#[cfg(feature = "parquet")]
#[pyfunction]
fn parquet_schema(py: Python, py_f: PyObject) -> PyResult<PyObject> {
    use polars_core::export::arrow::io::parquet::read::{infer_schema, read_metadata};

    let metadata = match get_either_file(py_f, false)? {
        EitherRustPythonFile::Rust(mut r) => read_metadata(&mut r).map_err(PyPolarsErr::from)?,
        EitherRustPythonFile::Py(mut r) => read_metadata(&mut r).map_err(PyPolarsErr::from)?,
    };
    let arrow_schema = infer_schema(&metadata).map_err(PyPolarsErr::from)?;

    let dict = PyDict::new(py);
    for field in arrow_schema.fields {
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
                .collect::<polars_core::error::PolarsResult<Vec<_>>>()
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
    tu: Wrap<TimeUnit>,
) -> PySeries {
    polars::time::date_range_impl(name, start, stop, Duration::parse(every), closed.0, tu.0)
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
fn coalesce_exprs(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    polars::lazy::dsl::coalesce(&exprs).into()
}

#[pyfunction]
fn sum_exprs(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    polars::lazy::dsl::sum_exprs(exprs).into()
}

#[pyfunction]
fn as_struct(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    polars::lazy::dsl::as_struct(&exprs).into()
}

#[pyfunction]
fn pool_size() -> usize {
    POOL.current_num_threads()
}

#[pyfunction]
fn arg_where(condition: PyExpr) -> PyExpr {
    polars::lazy::dsl::arg_where(condition.inner).into()
}

#[pyfunction]
fn get_idx_type(py: Python) -> PyObject {
    Wrap(IDX_DTYPE).to_object(py)
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
    m.add("DuplicateError", py.get_type::<DuplicateError>())
        .unwrap();
    m.add("PanicException", py.get_type::<PanicException>())
        .unwrap();
    m.add(
        "InvalidOperationError",
        py.get_type::<InvalidOperationError>(),
    )
    .unwrap();
    m.add_class::<PySeries>().unwrap();
    m.add_class::<PyDataFrame>().unwrap();
    m.add_class::<PyLazyFrame>().unwrap();
    m.add_class::<PyLazyGroupBy>().unwrap();
    m.add_class::<dsl::PyExpr>().unwrap();
    #[cfg(feature = "polars-sql")]
    m.add_class::<sql::PySQLContext>().unwrap();
    m.add_wrapped(wrap_pyfunction!(col)).unwrap();
    m.add_wrapped(wrap_pyfunction!(count)).unwrap();
    m.add_wrapped(wrap_pyfunction!(first)).unwrap();
    m.add_wrapped(wrap_pyfunction!(last)).unwrap();
    m.add_wrapped(wrap_pyfunction!(cols)).unwrap();
    m.add_wrapped(wrap_pyfunction!(dtype_cols)).unwrap();
    m.add_wrapped(wrap_pyfunction!(dtype_str_repr)).unwrap();
    m.add_wrapped(wrap_pyfunction!(lit)).unwrap();
    m.add_wrapped(wrap_pyfunction!(fold)).unwrap();
    m.add_wrapped(wrap_pyfunction!(binary_expr)).unwrap();
    m.add_wrapped(wrap_pyfunction!(arange)).unwrap();
    m.add_wrapped(wrap_pyfunction!(pearson_corr)).unwrap();
    m.add_wrapped(wrap_pyfunction!(cov)).unwrap();
    m.add_wrapped(wrap_pyfunction!(argsort_by)).unwrap();
    m.add_wrapped(wrap_pyfunction!(when)).unwrap();
    m.add_wrapped(wrap_pyfunction!(version)).unwrap();
    m.add_wrapped(wrap_pyfunction!(toggle_string_cache))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(using_string_cache)).unwrap();
    m.add_wrapped(wrap_pyfunction!(concat_str)).unwrap();
    m.add_wrapped(wrap_pyfunction!(concat_lst)).unwrap();
    m.add_wrapped(wrap_pyfunction!(concat_df)).unwrap();
    m.add_wrapped(wrap_pyfunction!(concat_lf)).unwrap();
    m.add_wrapped(wrap_pyfunction!(concat_series)).unwrap();
    #[cfg(feature = "ipc")]
    m.add_wrapped(wrap_pyfunction!(ipc_schema)).unwrap();
    #[cfg(feature = "parquet")]
    m.add_wrapped(wrap_pyfunction!(parquet_schema)).unwrap();
    m.add_wrapped(wrap_pyfunction!(collect_all)).unwrap();
    m.add_wrapped(wrap_pyfunction!(spearman_rank_corr)).unwrap();
    m.add_wrapped(wrap_pyfunction!(map_mul)).unwrap();
    m.add_wrapped(wrap_pyfunction!(py_diag_concat_df)).unwrap();
    m.add_wrapped(wrap_pyfunction!(py_hor_concat_df)).unwrap();
    m.add_wrapped(wrap_pyfunction!(py_datetime)).unwrap();
    m.add_wrapped(wrap_pyfunction!(py_duration)).unwrap();
    m.add_wrapped(wrap_pyfunction!(py_date_range)).unwrap();
    m.add_wrapped(wrap_pyfunction!(sum_exprs)).unwrap();
    m.add_wrapped(wrap_pyfunction!(min_exprs)).unwrap();
    m.add_wrapped(wrap_pyfunction!(max_exprs)).unwrap();
    m.add_wrapped(wrap_pyfunction!(as_struct)).unwrap();
    m.add_wrapped(wrap_pyfunction!(repeat)).unwrap();
    m.add_wrapped(wrap_pyfunction!(pool_size)).unwrap();
    m.add_wrapped(wrap_pyfunction!(arg_where)).unwrap();
    m.add_wrapped(wrap_pyfunction!(get_idx_type)).unwrap();
    m.add_wrapped(wrap_pyfunction!(coalesce_exprs)).unwrap();
    Ok(())
}
