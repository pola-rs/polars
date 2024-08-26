use polars::lazy::dsl;
use polars::prelude::*;
use polars_plan::prelude::UnionArgs;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyFloat, PyInt, PyString};

use crate::conversion::any_value::py_object_to_any_value;
use crate::conversion::{get_lf, Wrap};
use crate::error::PyPolarsErr;
use crate::expr::ToExprs;
use crate::map::lazy::binary_lambda;
use crate::prelude::{vec_extract_wrapped, ObjectValue};
use crate::{map, PyDataFrame, PyExpr, PyLazyFrame, PySeries};

macro_rules! set_unwrapped_or_0 {
    ($($var:ident),+ $(,)?) => {
        $(let $var = $var.map(|e| e.inner).unwrap_or(dsl::lit(0));)+
    };
}

#[pyfunction]
pub fn rolling_corr(
    x: PyExpr,
    y: PyExpr,
    window_size: IdxSize,
    min_periods: IdxSize,
    ddof: u8,
) -> PyExpr {
    dsl::rolling_corr(
        x.inner,
        y.inner,
        RollingCovOptions {
            min_periods,
            window_size,
            ddof,
        },
    )
    .into()
}

#[pyfunction]
pub fn rolling_cov(
    x: PyExpr,
    y: PyExpr,
    window_size: IdxSize,
    min_periods: IdxSize,
    ddof: u8,
) -> PyExpr {
    dsl::rolling_cov(
        x.inner,
        y.inner,
        RollingCovOptions {
            min_periods,
            window_size,
            ddof,
        },
    )
    .into()
}

#[pyfunction]
pub fn arg_sort_by(
    by: Vec<PyExpr>,
    descending: Vec<bool>,
    nulls_last: Vec<bool>,
    multithreaded: bool,
    maintain_order: bool,
) -> PyExpr {
    let by = by.into_iter().map(|e| e.inner).collect::<Vec<Expr>>();
    dsl::arg_sort_by(
        by,
        SortMultipleOptions {
            descending,
            nulls_last,
            multithreaded,
            maintain_order,
        },
    )
    .into()
}
#[pyfunction]
pub fn arg_where(condition: PyExpr) -> PyExpr {
    dsl::arg_where(condition.inner).into()
}

#[pyfunction]
pub fn as_struct(exprs: Vec<PyExpr>) -> PyResult<PyExpr> {
    let exprs = exprs.to_exprs();
    if exprs.is_empty() {
        return Err(PyValueError::new_err(
            "expected at least 1 expression in 'as_struct'",
        ));
    }
    Ok(dsl::as_struct(exprs).into())
}

#[pyfunction]
pub fn field(names: Vec<String>) -> PyExpr {
    dsl::Expr::Field(
        names
            .into_iter()
            .map(|name| Arc::from(name.as_str()))
            .collect(),
    )
    .into()
}

#[pyfunction]
pub fn coalesce(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    dsl::coalesce(&exprs).into()
}

#[pyfunction]
pub fn col(name: &str) -> PyExpr {
    dsl::col(name).into()
}

#[pyfunction]
pub fn collect_all(lfs: Vec<PyLazyFrame>, py: Python) -> PyResult<Vec<PyDataFrame>> {
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
pub fn collect_all_with_callback(lfs: Vec<PyLazyFrame>, lambda: PyObject) {
    use polars_core::utils::rayon::prelude::*;

    polars_core::POOL.spawn(move || {
        let result = lfs
            .par_iter()
            .map(|lf| {
                let df = lf.ldf.clone().collect()?;
                Ok(PyDataFrame::new(df))
            })
            .collect::<polars_core::error::PolarsResult<Vec<_>>>()
            .map_err(PyPolarsErr::from);

        Python::with_gil(|py| match result {
            Ok(dfs) => {
                lambda.call1(py, (dfs,)).map_err(|err| err.restore(py)).ok();
            },
            Err(err) => {
                lambda
                    .call1(py, (PyErr::from(err).to_object(py),))
                    .map_err(|err| err.restore(py))
                    .ok();
            },
        })
    })
}

#[pyfunction]
pub fn cols(names: Vec<String>) -> PyExpr {
    dsl::cols(names).into()
}

#[pyfunction]
pub fn concat_lf(
    seq: &Bound<'_, PyAny>,
    rechunk: bool,
    parallel: bool,
    to_supertypes: bool,
) -> PyResult<PyLazyFrame> {
    let len = seq.len()?;
    let mut lfs = Vec::with_capacity(len);

    for res in seq.iter()? {
        let item = res?;
        let lf = get_lf(&item)?;
        lfs.push(lf);
    }

    let lf = dsl::concat(
        lfs,
        UnionArgs {
            rechunk,
            parallel,
            to_supertypes,
            ..Default::default()
        },
    )
    .map_err(PyPolarsErr::from)?;
    Ok(lf.into())
}

#[pyfunction]
pub fn concat_list(s: Vec<PyExpr>) -> PyResult<PyExpr> {
    let s = s.into_iter().map(|e| e.inner).collect::<Vec<_>>();
    let expr = dsl::concat_list(s).map_err(PyPolarsErr::from)?;
    Ok(expr.into())
}

#[pyfunction]
pub fn concat_str(s: Vec<PyExpr>, separator: &str, ignore_nulls: bool) -> PyExpr {
    let s = s.into_iter().map(|e| e.inner).collect::<Vec<_>>();
    dsl::concat_str(s, separator, ignore_nulls).into()
}

#[pyfunction]
pub fn len() -> PyExpr {
    dsl::len().into()
}

#[pyfunction]
pub fn cov(a: PyExpr, b: PyExpr, ddof: u8) -> PyExpr {
    dsl::cov(a.inner, b.inner, ddof).into()
}

#[pyfunction]
#[cfg(feature = "trigonometry")]
pub fn arctan2(y: PyExpr, x: PyExpr) -> PyExpr {
    y.inner.arctan2(x.inner).into()
}

#[pyfunction]
pub fn cum_fold(acc: PyExpr, lambda: PyObject, exprs: Vec<PyExpr>, include_init: bool) -> PyExpr {
    let exprs = exprs.to_exprs();

    let func = move |a: Series, b: Series| binary_lambda(&lambda, a, b);
    dsl::cum_fold_exprs(acc.inner, func, exprs, include_init).into()
}

#[pyfunction]
pub fn cum_reduce(lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();

    let func = move |a: Series, b: Series| binary_lambda(&lambda, a, b);
    dsl::cum_reduce_exprs(func, exprs).into()
}

#[pyfunction]
#[pyo3(signature = (year, month, day, hour=None, minute=None, second=None, microsecond=None, time_unit=Wrap(TimeUnit::Microseconds), time_zone=None, ambiguous=None))]
pub fn datetime(
    year: PyExpr,
    month: PyExpr,
    day: PyExpr,
    hour: Option<PyExpr>,
    minute: Option<PyExpr>,
    second: Option<PyExpr>,
    microsecond: Option<PyExpr>,
    time_unit: Wrap<TimeUnit>,
    time_zone: Option<TimeZone>,
    ambiguous: Option<PyExpr>,
) -> PyExpr {
    let year = year.inner;
    let month = month.inner;
    let day = day.inner;
    set_unwrapped_or_0!(hour, minute, second, microsecond);
    let ambiguous = ambiguous
        .map(|e| e.inner)
        .unwrap_or(dsl::lit(String::from("raise")));
    let time_unit = time_unit.0;
    let args = DatetimeArgs {
        year,
        month,
        day,
        hour,
        minute,
        second,
        microsecond,
        time_unit,
        time_zone,
        ambiguous,
    };
    dsl::datetime(args).into()
}

#[pyfunction]
pub fn concat_lf_diagonal(
    lfs: &Bound<'_, PyAny>,
    rechunk: bool,
    parallel: bool,
    to_supertypes: bool,
) -> PyResult<PyLazyFrame> {
    let iter = lfs.iter()?;

    let lfs = iter
        .map(|item| {
            let item = item?;
            get_lf(&item)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let lf = dsl::functions::concat_lf_diagonal(
        lfs,
        UnionArgs {
            rechunk,
            parallel,
            to_supertypes,
            ..Default::default()
        },
    )
    .map_err(PyPolarsErr::from)?;
    Ok(lf.into())
}

#[pyfunction]
pub fn concat_lf_horizontal(lfs: &Bound<'_, PyAny>, parallel: bool) -> PyResult<PyLazyFrame> {
    let iter = lfs.iter()?;

    let lfs = iter
        .map(|item| {
            let item = item?;
            get_lf(&item)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let args = UnionArgs {
        rechunk: false, // No need to rechunk with horizontal concatenation
        parallel,
        to_supertypes: false,
        ..Default::default()
    };
    let lf = dsl::functions::concat_lf_horizontal(lfs, args).map_err(PyPolarsErr::from)?;
    Ok(lf.into())
}

#[pyfunction]
pub fn concat_expr(e: Vec<PyExpr>, rechunk: bool) -> PyResult<PyExpr> {
    let e = e.to_exprs();
    let e = dsl::functions::concat_expr(e, rechunk).map_err(PyPolarsErr::from)?;
    Ok(e.into())
}

#[pyfunction]
pub fn dtype_cols(dtypes: Vec<Wrap<DataType>>) -> PyResult<PyExpr> {
    let dtypes = vec_extract_wrapped(dtypes);
    Ok(dsl::dtype_cols(dtypes).into())
}

#[pyfunction]
pub fn index_cols(indices: Vec<i64>) -> PyExpr {
    if indices.len() == 1 {
        dsl::nth(indices[0])
    } else {
        dsl::index_cols(indices)
    }
    .into()
}

#[pyfunction]
#[pyo3(signature = (weeks, days, hours, minutes, seconds, milliseconds, microseconds, nanoseconds, time_unit))]
pub fn duration(
    weeks: Option<PyExpr>,
    days: Option<PyExpr>,
    hours: Option<PyExpr>,
    minutes: Option<PyExpr>,
    seconds: Option<PyExpr>,
    milliseconds: Option<PyExpr>,
    microseconds: Option<PyExpr>,
    nanoseconds: Option<PyExpr>,
    time_unit: Wrap<TimeUnit>,
) -> PyExpr {
    set_unwrapped_or_0!(
        weeks,
        days,
        hours,
        minutes,
        seconds,
        milliseconds,
        microseconds,
        nanoseconds,
    );
    let args = DurationArgs {
        weeks,
        days,
        hours,
        minutes,
        seconds,
        milliseconds,
        microseconds,
        nanoseconds,
        time_unit: time_unit.0,
    };
    dsl::duration(args).into()
}

#[pyfunction]
pub fn first() -> PyExpr {
    dsl::first().into()
}

#[pyfunction]
pub fn fold(acc: PyExpr, lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();

    let func = move |a: Series, b: Series| binary_lambda(&lambda, a, b);
    dsl::fold_exprs(acc.inner, func, exprs).into()
}

#[pyfunction]
pub fn last() -> PyExpr {
    dsl::last().into()
}

#[pyfunction]
pub fn nth(n: i64) -> PyExpr {
    dsl::nth(n).into()
}

#[pyfunction]
pub fn lit(value: &Bound<'_, PyAny>, allow_object: bool) -> PyResult<PyExpr> {
    if value.is_instance_of::<PyBool>() {
        let val = value.extract::<bool>().unwrap();
        Ok(dsl::lit(val).into())
    } else if let Ok(int) = value.downcast::<PyInt>() {
        let v = int
            .extract::<i128>()
            .map_err(|e| polars_err!(InvalidOperation: "integer too large for Polars: {e}"))
            .map_err(PyPolarsErr::from)?;
        Ok(Expr::Literal(LiteralValue::Int(v)).into())
    } else if let Ok(float) = value.downcast::<PyFloat>() {
        let val = float.extract::<f64>().unwrap();
        Ok(Expr::Literal(LiteralValue::Float(val)).into())
    } else if let Ok(pystr) = value.downcast::<PyString>() {
        Ok(dsl::lit(pystr.to_string()).into())
    } else if let Ok(series) = value.extract::<PySeries>() {
        Ok(dsl::lit(series.series).into())
    } else if value.is_none() {
        Ok(dsl::lit(Null {}).into())
    } else if let Ok(value) = value.downcast::<PyBytes>() {
        Ok(dsl::lit(value.as_bytes()).into())
    } else if matches!(
        value.get_type().qualname().unwrap().as_str(),
        "date" | "datetime" | "time" | "timedelta" | "Decimal"
    ) {
        let av = py_object_to_any_value(value, true)?;
        Ok(Expr::Literal(LiteralValue::try_from(av).unwrap()).into())
    } else if allow_object {
        let s = Python::with_gil(|py| {
            PySeries::new_object(py, "", vec![ObjectValue::from(value.into_py(py))], false).series
        });
        Ok(dsl::lit(s).into())
    } else {
        Err(PyTypeError::new_err(format!(
            "cannot create expression literal for value of type {}: {}\
            \n\nHint: Pass `allow_object=True` to accept any value and create a literal of type Object.",
            value.get_type().qualname()?,
            value.repr()?
        )))
    }
}

#[pyfunction]
#[pyo3(signature = (pyexpr, lambda, output_type, map_groups, returns_scalar))]
pub fn map_mul(
    py: Python,
    pyexpr: Vec<PyExpr>,
    lambda: PyObject,
    output_type: Option<Wrap<DataType>>,
    map_groups: bool,
    returns_scalar: bool,
) -> PyExpr {
    map::lazy::map_mul(&pyexpr, py, lambda, output_type, map_groups, returns_scalar)
}

#[pyfunction]
pub fn pearson_corr(a: PyExpr, b: PyExpr, ddof: u8) -> PyExpr {
    dsl::pearson_corr(a.inner, b.inner, ddof).into()
}

#[pyfunction]
pub fn reduce(lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();

    let func = move |a: Series, b: Series| binary_lambda(&lambda, a, b);
    dsl::reduce_exprs(func, exprs).into()
}

#[pyfunction]
pub fn repeat(value: PyExpr, n: PyExpr, dtype: Option<Wrap<DataType>>) -> PyResult<PyExpr> {
    let mut value = value.inner;
    let n = n.inner;

    if let Some(dtype) = dtype {
        value = value.cast(dtype.0);
    }

    Ok(dsl::repeat(value, n).into())
}

#[pyfunction]
pub fn spearman_rank_corr(a: PyExpr, b: PyExpr, ddof: u8, propagate_nans: bool) -> PyExpr {
    #[cfg(feature = "propagate_nans")]
    {
        dsl::spearman_rank_corr(a.inner, b.inner, ddof, propagate_nans).into()
    }
    #[cfg(not(feature = "propagate_nans"))]
    {
        panic!("activate 'propagate_nans'")
    }
}

#[pyfunction]
#[cfg(feature = "sql")]
pub fn sql_expr(sql: &str) -> PyResult<PyExpr> {
    let expr = polars::sql::sql_expr(sql).map_err(PyPolarsErr::from)?;
    Ok(expr.into())
}
