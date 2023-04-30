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
#[cfg(feature = "csv")]
mod batched_csv;
pub mod conversion;
pub mod dataframe;
pub mod datatypes;
pub mod error;
pub mod file;
mod functions;
pub mod lazy;
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
#[cfg(any(not(target_os = "linux"), use_mimalloc))]
use mimalloc::MiMalloc;
#[cfg(feature = "object")]
pub use object::register_object_builder;
use polars_core::prelude::{DataFrame, IDX_DTYPE};
use polars_core::POOL;
use pyo3::exceptions::PyValueError;
use pyo3::panic::PanicException;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::conversion::{get_df, get_series, Wrap};
use crate::dataframe::PyDataFrame;
use crate::error::{
    ArrowErrorException, ColumnNotFoundError, ComputeError, DuplicateError, InvalidOperationError,
    NoDataError, PyPolarsErr, SchemaError, SchemaFieldNotFoundError, StructFieldNotFoundError,
};
use crate::file::{get_either_file, EitherRustPythonFile};
use crate::lazy::dataframe::{PyLazyFrame, PyLazyGroupBy};
use crate::lazy::dsl;
use crate::lazy::dsl::PyExpr;
use crate::prelude::DataType;
use crate::series::PySeries;

#[global_allocator]
#[cfg(all(target_os = "linux", not(use_mimalloc)))]
static ALLOC: Jemalloc = Jemalloc;

#[global_allocator]
#[cfg(any(not(target_os = "linux"), use_mimalloc))]
static ALLOC: MiMalloc = MiMalloc;

#[pyfunction]
fn dtype_str_repr(dtype: Wrap<DataType>) -> PyResult<String> {
    let dtype = dtype.0;
    Ok(dtype.to_string())
}

#[pyfunction]
fn binary_expr(l: dsl::PyExpr, op: u8, r: dsl::PyExpr) -> dsl::PyExpr {
    dsl::binary_expr(l, op, r)
}

const VERSION: &str = env!("CARGO_PKG_VERSION");
#[pyfunction]
fn get_polars_version() -> &'static str {
    VERSION
}

#[pyfunction]
fn enable_string_cache(toggle: bool) {
    polars_rs::enable_string_cache(toggle)
}

#[pyfunction]
fn using_string_cache() -> bool {
    polars_rs::using_string_cache()
}

#[pyfunction]
fn concat_df(dfs: &PyAny, py: Python) -> PyResult<PyDataFrame> {
    use polars_core::error::PolarsResult;
    use polars_core::utils::rayon::prelude::*;

    let mut iter = dfs.iter()?;
    let first = iter.next().unwrap()?;

    let first_rdf = get_df(first)?;
    let identity_df = first_rdf.clear();

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
fn diag_concat_df(dfs: &PyAny) -> PyResult<PyDataFrame> {
    let iter = dfs.iter()?;

    let dfs = iter
        .map(|item| {
            let item = item?;
            get_df(item)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let df = polars_rs::functions::diag_concat_df(&dfs).map_err(PyPolarsErr::from)?;
    Ok(df.into())
}

#[pyfunction]
fn hor_concat_df(dfs: &PyAny) -> PyResult<PyDataFrame> {
    let iter = dfs.iter()?;

    let dfs = iter
        .map(|item| {
            let item = item?;
            get_df(item)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let df = polars_rs::functions::hor_concat_df(&dfs).map_err(PyPolarsErr::from)?;
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
fn get_index_type(py: Python) -> PyObject {
    Wrap(IDX_DTYPE).to_object(py)
}

#[pyfunction]
fn threadpool_size() -> usize {
    POOL.current_num_threads()
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

#[pyfunction]
fn get_float_fmt() -> PyResult<String> {
    use polars_core::fmt::{get_float_fmt, FloatFmt};
    let strfmt = match get_float_fmt() {
        FloatFmt::Full => "full",
        FloatFmt::Mixed => "mixed",
    };
    Ok(strfmt.to_string())
}

#[pymodule]
fn polars(py: Python, m: &PyModule) -> PyResult<()> {
    // Classes
    m.add_class::<PySeries>().unwrap();
    m.add_class::<PyDataFrame>().unwrap();
    m.add_class::<PyLazyFrame>().unwrap();
    m.add_class::<PyLazyGroupBy>().unwrap();
    m.add_class::<dsl::PyExpr>().unwrap();
    #[cfg(feature = "csv")]
    m.add_class::<batched_csv::PyBatchedCsv>().unwrap();
    #[cfg(feature = "sql")]
    m.add_class::<sql::PySQLContext>().unwrap();

    // Functions - eager
    m.add_wrapped(wrap_pyfunction!(concat_df)).unwrap();
    m.add_wrapped(wrap_pyfunction!(concat_series)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::eager::date_range))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(diag_concat_df)).unwrap();
    m.add_wrapped(wrap_pyfunction!(hor_concat_df)).unwrap();

    // Functions - lazy
    m.add_wrapped(wrap_pyfunction!(functions::lazy::arange))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::arg_sort_by))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::arg_where))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::as_struct))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::coalesce))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::col))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::collect_all))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::cols))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::concat_lf))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::concat_list))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::concat_str))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::count))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::cov))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::cumfold))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::cumreduce))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::date_range_lazy))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::datetime))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::diag_concat_lf))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::dtype_cols))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::duration))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::first))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::fold))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::last))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::lit))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::map_mul))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::max_exprs))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::min_exprs))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::pearson_corr))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::reduce))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::repeat))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::spearman_rank_corr))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::sum_exprs))
        .unwrap();

    // Functions - whenthen
    m.add_wrapped(wrap_pyfunction!(functions::whenthen::when))
        .unwrap();

    // Exceptions
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
    m.add("PolarsPanicError", py.get_type::<PanicException>())
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

    // Other
    m.add_wrapped(wrap_pyfunction!(dtype_str_repr)).unwrap();
    m.add_wrapped(wrap_pyfunction!(binary_expr)).unwrap();
    m.add_wrapped(wrap_pyfunction!(get_polars_version)).unwrap();
    m.add_wrapped(wrap_pyfunction!(enable_string_cache))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(using_string_cache)).unwrap();
    #[cfg(feature = "ipc")]
    m.add_wrapped(wrap_pyfunction!(ipc_schema)).unwrap();
    #[cfg(feature = "parquet")]
    m.add_wrapped(wrap_pyfunction!(parquet_schema)).unwrap();
    m.add_wrapped(wrap_pyfunction!(threadpool_size)).unwrap();
    m.add_wrapped(wrap_pyfunction!(get_index_type)).unwrap();
    m.add_wrapped(wrap_pyfunction!(set_float_fmt)).unwrap();
    m.add_wrapped(wrap_pyfunction!(get_float_fmt)).unwrap();
    #[cfg(feature = "object")]
    m.add_wrapped(wrap_pyfunction!(register_object_builder))
        .unwrap();
    #[cfg(feature = "build_info")]
    m.add(
        "_build_info_",
        pyo3_built!(py, build, "build", "time", "deps", "features", "host", "target", "git"),
    )?;

    Ok(())
}
