#![feature(vec_into_raw_parts)]
#![allow(clippy::nonstandard_macro_braces)] // needed because clippy does not understand proc macro of pyo3
#![allow(clippy::transmute_undefined_repr)]
extern crate polars as polars_rs;

#[cfg(feature = "build_info")]
#[macro_use]
extern crate pyo3_built;

#[cfg(feature = "build_info")]
#[allow(dead_code)]
mod build {
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

pub mod apply;
pub mod arrow_interop;
#[cfg(feature = "csv-file")]
mod batched_csv;
pub mod conversion;
pub mod dataframe;
pub mod datatypes;
pub mod error;
pub mod file;
pub mod lazy;
mod list_construction;
pub mod npy;
#[cfg(feature = "object")]
mod object;
pub mod prelude;
pub(crate) mod py_modules;
pub mod series;
mod set;
#[cfg(feature = "sql")]
mod sql;
pub mod utils;

#[cfg(all(target_os = "linux", not(use_mimalloc)))]
use jemallocator::Jemalloc;
use lazy::ToExprs;
#[cfg(any(not(target_os = "linux"), use_mimalloc))]
use mimalloc::MiMalloc;
use polars_core::datatypes::{TimeUnit, TimeZone};
use polars_core::prelude::{DataFrame, IntoSeries, IDX_DTYPE};
use polars_core::POOL;
use polars_rs::functions::{diag_concat_df, hor_concat_df};
use polars_rs::prelude::Null;
use pyo3::exceptions::PyValueError;
use pyo3::panic::PanicException;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyString};
use pyo3::wrap_pyfunction;

use crate::conversion::{get_df, get_lf, get_series, Wrap};
use crate::dataframe::PyDataFrame;
use crate::error::{
    ArrowErrorException, ColumnNotFoundError, ComputeError, DuplicateError, InvalidOperationError,
    NoDataError, PyPolarsErr, SchemaError, SchemaFieldNotFoundError, StructFieldNotFoundError,
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
#[cfg(all(target_os = "linux", not(use_mimalloc)))]
static ALLOC: Jemalloc = Jemalloc;

#[global_allocator]
#[cfg(any(not(target_os = "linux"), use_mimalloc))]
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
fn fold(acc: PyExpr, lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    dsl::fold(acc, lambda, exprs)
}

#[pyfunction]
fn reduce(lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    dsl::reduce(lambda, exprs)
}

#[pyfunction]
fn cumfold(acc: PyExpr, lambda: PyObject, exprs: Vec<PyExpr>, include_init: bool) -> PyExpr {
    dsl::cumfold(acc, lambda, exprs, include_init)
}

#[pyfunction]
fn cumreduce(lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    dsl::cumreduce(lambda, exprs)
}

#[pyfunction]
fn arange(low: PyExpr, high: PyExpr, step: usize) -> PyExpr {
    polars_rs::lazy::dsl::arange(low.inner, high.inner, step).into()
}

#[pyfunction]
fn repeat(value: &PyAny, n_times: PyExpr) -> PyResult<PyExpr> {
    if let Ok(true) = value.is_instance_of::<PyBool>() {
        let val = value.extract::<bool>().unwrap();
        Ok(polars_rs::lazy::dsl::repeat(val, n_times.inner).into())
    } else if let Ok(int) = value.downcast::<PyInt>() {
        let val = int.extract::<i64>().unwrap();

        if val > 0 && val < i32::MAX as i64 || val < 0 && val > i32::MIN as i64 {
            Ok(polars_rs::lazy::dsl::repeat(val as i32, n_times.inner).into())
        } else {
            Ok(polars_rs::lazy::dsl::repeat(val, n_times.inner).into())
        }
    } else if let Ok(float) = value.downcast::<PyFloat>() {
        let val = float.extract::<f64>().unwrap();
        Ok(polars_rs::lazy::dsl::repeat(val, n_times.inner).into())
    } else if let Ok(pystr) = value.downcast::<PyString>() {
        let val = pystr
            .to_str()
            .expect("could not transform Python string to Rust Unicode");
        Ok(polars_rs::lazy::dsl::repeat(val, n_times.inner).into())
    } else if value.is_none() {
        Ok(polars_rs::lazy::dsl::repeat(Null {}, n_times.inner).into())
    } else {
        Err(PyValueError::new_err(format!(
            "could not convert value {:?} as a Literal",
            value.str()?
        )))
    }
}

#[pyfunction]
fn pearson_corr(a: dsl::PyExpr, b: dsl::PyExpr, ddof: u8) -> dsl::PyExpr {
    polars_rs::lazy::dsl::pearson_corr(a.inner, b.inner, ddof).into()
}

#[pyfunction]
fn spearman_rank_corr(
    a: dsl::PyExpr,
    b: dsl::PyExpr,
    ddof: u8,
    propagate_nans: bool,
) -> dsl::PyExpr {
    #[cfg(feature = "propagate_nans")]
    {
        polars_rs::lazy::dsl::spearman_rank_corr(a.inner, b.inner, ddof, propagate_nans).into()
    }
    #[cfg(not(feature = "propagate_nans"))]
    {
        panic!("activate 'popagate_nans'")
    }
}

#[pyfunction]
fn cov(a: dsl::PyExpr, b: dsl::PyExpr) -> dsl::PyExpr {
    polars_rs::lazy::dsl::cov(a.inner, b.inner).into()
}

#[pyfunction]
fn arg_sort_by(by: Vec<dsl::PyExpr>, reverse: Vec<bool>) -> dsl::PyExpr {
    let by = by
        .into_iter()
        .map(|e| e.inner)
        .collect::<Vec<polars_rs::lazy::dsl::Expr>>();
    polars_rs::lazy::dsl::arg_sort_by(by, &reverse).into()
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
    polars_rs::toggle_string_cache(toggle)
}

#[pyfunction]
fn using_string_cache() -> bool {
    polars_rs::using_string_cache()
}

#[pyfunction]
fn concat_str(s: Vec<dsl::PyExpr>, sep: &str) -> dsl::PyExpr {
    let s = s.into_iter().map(|e| e.inner).collect::<Vec<_>>();
    polars_rs::lazy::dsl::concat_str(s, sep).into()
}

#[pyfunction]
fn concat_lst(s: Vec<dsl::PyExpr>) -> dsl::PyExpr {
    let s = s.into_iter().map(|e| e.inner).collect::<Vec<_>>();
    polars_rs::lazy::dsl::concat_lst(s).into()
}

#[pyfunction]
fn py_datetime(
    year: dsl::PyExpr,
    month: dsl::PyExpr,
    day: dsl::PyExpr,
    hour: Option<dsl::PyExpr>,
    minute: Option<dsl::PyExpr>,
    second: Option<dsl::PyExpr>,
    microsecond: Option<dsl::PyExpr>,
) -> dsl::PyExpr {
    let hour = hour.map(|e| e.inner);
    let minute = minute.map(|e| e.inner);
    let second = second.map(|e| e.inner);
    let microsecond = microsecond.map(|e| e.inner);

    let args = DatetimeArgs {
        year: year.inner,
        month: month.inner,
        day: day.inner,
        hour,
        minute,
        second,
        microsecond,
    };

    polars_rs::lazy::dsl::datetime(args).into()
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn py_duration(
    days: Option<PyExpr>,
    seconds: Option<PyExpr>,
    nanoseconds: Option<PyExpr>,
    microseconds: Option<PyExpr>,
    milliseconds: Option<PyExpr>,
    minutes: Option<PyExpr>,
    hours: Option<PyExpr>,
    weeks: Option<PyExpr>,
) -> dsl::PyExpr {
    let args = DurationArgs {
        days: days.map(|e| e.inner),
        seconds: seconds.map(|e| e.inner),
        nanoseconds: nanoseconds.map(|e| e.inner),
        microseconds: microseconds.map(|e| e.inner),
        milliseconds: milliseconds.map(|e| e.inner),
        minutes: minutes.map(|e| e.inner),
        hours: hours.map(|e| e.inner),
        weeks: weeks.map(|e| e.inner),
    };

    polars_rs::lazy::dsl::duration(args).into()
}

#[pyfunction]
fn concat_df(dfs: &PyAny, py: Python) -> PyResult<PyDataFrame> {
    use polars_core::error::PolarsResult;
    use polars_core::utils::rayon::prelude::*;

    let mut iter = dfs.iter()?;
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
fn concat_lf(seq: &PyAny, rechunk: bool, parallel: bool) -> PyResult<PyLazyFrame> {
    let len = seq.len()?;
    let mut lfs = Vec::with_capacity(len);

    for res in seq.iter()? {
        let item = res?;
        let lf = get_lf(item)?;
        lfs.push(lf);
    }

    let lf = polars_rs::lazy::dsl::concat(lfs, rechunk, parallel).map_err(PyPolarsErr::from)?;
    Ok(lf.into())
}

#[pyfunction]
fn py_diag_concat_df(dfs: &PyAny) -> PyResult<PyDataFrame> {
    let iter = dfs.iter()?;

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
fn py_diag_concat_lf(lfs: &PyAny, rechunk: bool, parallel: bool) -> PyResult<PyLazyFrame> {
    let iter = lfs.iter()?;

    let lfs = iter
        .map(|item| {
            let item = item?;
            get_lf(item)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let lf = polars_rs::lazy::dsl::functions::diag_concat_lf(lfs, rechunk, parallel)
        .map_err(PyPolarsErr::from)?;
    Ok(lf.into())
}

#[pyfunction]
fn py_hor_concat_df(dfs: &PyAny) -> PyResult<PyDataFrame> {
    let iter = dfs.iter()?;

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
    let mut iter = series.iter()?;
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
#[pyo3(signature = (pyexpr, lambda, output_type, apply_groups, returns_scalar))]
pub fn map_mul(
    py: Python,
    pyexpr: Vec<PyExpr>,
    lambda: PyObject,
    output_type: Option<Wrap<DataType>>,
    apply_groups: bool,
    returns_scalar: bool,
) -> PyExpr {
    lazy::map_mul(
        &pyexpr,
        py,
        lambda,
        output_type,
        apply_groups,
        returns_scalar,
    )
}

#[pyfunction]
fn py_date_range(
    start: i64,
    stop: i64,
    every: &str,
    closed: Wrap<ClosedWindow>,
    name: &str,
    tu: Wrap<TimeUnit>,
    tz: Option<TimeZone>,
) -> PyResult<PySeries> {
    let date_range = polars_rs::time::date_range_impl(
        name,
        start,
        stop,
        Duration::parse(every),
        closed.0,
        tu.0,
        tz.as_ref(),
    )
    .map_err(PyPolarsErr::from)?;
    Ok(date_range.into_series().into())
}

#[pyfunction]
fn py_date_range_lazy(
    start: PyExpr,
    end: PyExpr,
    every: &str,
    closed: Wrap<ClosedWindow>,
    name: String,
    tz: Option<TimeZone>,
) -> PyExpr {
    let start = start.inner;
    let end = end.inner;
    let every = Duration::parse(every);
    polars_rs::lazy::dsl::functions::date_range(name, start, end, every, closed.0, tz).into()
}

#[pyfunction]
fn min_exprs(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    polars_rs::lazy::dsl::min_exprs(exprs).into()
}

#[pyfunction]
fn max_exprs(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    polars_rs::lazy::dsl::max_exprs(exprs).into()
}

#[pyfunction]
fn coalesce_exprs(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    polars_rs::lazy::dsl::coalesce(&exprs).into()
}

#[pyfunction]
fn sum_exprs(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    polars_rs::lazy::dsl::sum_exprs(exprs).into()
}

#[pyfunction]
fn as_struct(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    polars_rs::lazy::dsl::as_struct(&exprs).into()
}

#[pyfunction]
fn pool_size() -> usize {
    POOL.current_num_threads()
}

#[pyfunction]
fn arg_where(condition: PyExpr) -> PyExpr {
    polars_rs::lazy::dsl::arg_where(condition.inner).into()
}

#[pyfunction]
fn get_idx_type(py: Python) -> PyObject {
    Wrap(IDX_DTYPE).to_object(py)
}

#[pyfunction]
fn set_float_fmt(fmt: &str) -> PyResult<()> {
    use polars_core::fmt::{set_float_fmt, FloatFmt};
    let fmt = match fmt {
        "full" => FloatFmt::Full,
        "mixed" => FloatFmt::Mixed,
        e => {
            return Err(PyValueError::new_err(format!(
                "fmt must be one of {{'full', 'mixed'}}, got {e}",
            )))
        }
    };
    set_float_fmt(fmt);
    Ok(())
}

#[pymodule]
fn polars(py: Python, m: &PyModule) -> PyResult<()> {
    m.add("ArrowError", py.get_type::<ArrowErrorException>())
        .unwrap();
    m.add("ColumnNotFoundError", py.get_type::<ColumnNotFoundError>())
        .unwrap();
    m.add("ComputeError", py.get_type::<ComputeError>())
        .unwrap();
    m.add("DuplicateError", py.get_type::<DuplicateError>())
        .unwrap();
    m.add(
        "InvalidOperationError",
        py.get_type::<InvalidOperationError>(),
    )
    .unwrap();
    m.add("NoDataError", py.get_type::<NoDataError>()).unwrap();
    m.add("PanicException", py.get_type::<PanicException>())
        .unwrap();
    m.add("SchemaError", py.get_type::<SchemaError>()).unwrap();
    m.add(
        "SchemaFieldNotFoundError",
        py.get_type::<SchemaFieldNotFoundError>(),
    )
    .unwrap();
    m.add("ShapeError", py.get_type::<crate::error::ShapeError>())
        .unwrap();
    m.add(
        "StructFieldNotFoundError",
        py.get_type::<StructFieldNotFoundError>(),
    )
    .unwrap();

    #[cfg(feature = "build_info")]
    m.add(
        "_build_info_",
        pyo3_built!(py, build, "build", "time", "deps", "features", "host", "target", "git"),
    )?;

    m.add_class::<PySeries>().unwrap();
    m.add_class::<PyDataFrame>().unwrap();
    m.add_class::<PyLazyFrame>().unwrap();
    m.add_class::<PyLazyGroupBy>().unwrap();
    m.add_class::<dsl::PyExpr>().unwrap();
    #[cfg(feature = "csv-file")]
    m.add_class::<batched_csv::PyBatchedCsv>().unwrap();
    #[cfg(feature = "sql")]
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
    m.add_wrapped(wrap_pyfunction!(cumfold)).unwrap();
    m.add_wrapped(wrap_pyfunction!(reduce)).unwrap();
    m.add_wrapped(wrap_pyfunction!(cumreduce)).unwrap();
    m.add_wrapped(wrap_pyfunction!(binary_expr)).unwrap();
    m.add_wrapped(wrap_pyfunction!(arange)).unwrap();
    m.add_wrapped(wrap_pyfunction!(pearson_corr)).unwrap();
    m.add_wrapped(wrap_pyfunction!(cov)).unwrap();
    m.add_wrapped(wrap_pyfunction!(arg_sort_by)).unwrap();
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
    m.add_wrapped(wrap_pyfunction!(py_diag_concat_lf)).unwrap();
    m.add_wrapped(wrap_pyfunction!(py_hor_concat_df)).unwrap();
    m.add_wrapped(wrap_pyfunction!(py_datetime)).unwrap();
    m.add_wrapped(wrap_pyfunction!(py_duration)).unwrap();
    m.add_wrapped(wrap_pyfunction!(py_date_range)).unwrap();
    m.add_wrapped(wrap_pyfunction!(py_date_range_lazy)).unwrap();
    m.add_wrapped(wrap_pyfunction!(sum_exprs)).unwrap();
    m.add_wrapped(wrap_pyfunction!(min_exprs)).unwrap();
    m.add_wrapped(wrap_pyfunction!(max_exprs)).unwrap();
    m.add_wrapped(wrap_pyfunction!(as_struct)).unwrap();
    m.add_wrapped(wrap_pyfunction!(repeat)).unwrap();
    m.add_wrapped(wrap_pyfunction!(pool_size)).unwrap();
    m.add_wrapped(wrap_pyfunction!(arg_where)).unwrap();
    m.add_wrapped(wrap_pyfunction!(get_idx_type)).unwrap();
    m.add_wrapped(wrap_pyfunction!(coalesce_exprs)).unwrap();
    m.add_wrapped(wrap_pyfunction!(set_float_fmt)).unwrap();
    Ok(())
}
