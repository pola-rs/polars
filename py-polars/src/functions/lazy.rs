use polars::lazy::dsl;
use polars::lazy::dsl::Expr;
use polars::prelude::*;
use pyo3::prelude::*;

use crate::conversion::Wrap;
use crate::lazy::{binary_lambda, ToExprs};
use crate::prelude::{vec_extract_wrapped, DataType, DatetimeArgs, DurationArgs};
use crate::{PyDataFrame, PyExpr, PyLazyFrame, PyPolarsErr};
macro_rules! set_unwrapped_or_0 {
    ($($var:ident),+ $(,)?) => {
        $(let $var = $var.map(|e| e.inner).unwrap_or(dsl::lit(0));)+
    };
}

#[pyfunction]
pub fn arange(start: PyExpr, end: PyExpr, step: i64) -> PyExpr {
    dsl::arange(start.inner, end.inner, step).into()
}

#[pyfunction]
pub fn arg_sort_by(by: Vec<PyExpr>, descending: Vec<bool>) -> PyExpr {
    let by = by.into_iter().map(|e| e.inner).collect::<Vec<Expr>>();
    dsl::arg_sort_by(by, &descending).into()
}

#[pyfunction]
pub fn arg_where(condition: PyExpr) -> PyExpr {
    dsl::arg_where(condition.inner).into()
}

#[pyfunction]
pub fn as_struct(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    dsl::as_struct(&exprs).into()
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
pub fn cols(names: Vec<String>) -> PyExpr {
    dsl::cols(names).into()
}

#[pyfunction]
pub fn concat_list(s: Vec<PyExpr>) -> PyResult<PyExpr> {
    let s = s.into_iter().map(|e| e.inner).collect::<Vec<_>>();
    let expr = dsl::concat_lst(s).map_err(PyPolarsErr::from)?;
    Ok(expr.into())
}

#[pyfunction]
pub fn concat_str(s: Vec<PyExpr>, separator: &str) -> PyExpr {
    let s = s.into_iter().map(|e| e.inner).collect::<Vec<_>>();
    dsl::concat_str(s, separator).into()
}

#[pyfunction]
pub fn count() -> PyExpr {
    dsl::count().into()
}

#[pyfunction]
pub fn cov(a: PyExpr, b: PyExpr) -> PyExpr {
    dsl::cov(a.inner, b.inner).into()
}

#[pyfunction]
pub fn cumfold(acc: PyExpr, lambda: PyObject, exprs: Vec<PyExpr>, include_init: bool) -> PyExpr {
    let exprs = exprs.to_exprs();

    let func = move |a: Series, b: Series| binary_lambda(&lambda, a, b);
    dsl::cumfold_exprs(acc.inner, func, exprs, include_init).into()
}

#[pyfunction]
pub fn cumreduce(lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();

    let func = move |a: Series, b: Series| binary_lambda(&lambda, a, b);
    dsl::cumreduce_exprs(func, exprs).into()
}

#[pyfunction]
pub fn datetime(
    year: PyExpr,
    month: PyExpr,
    day: PyExpr,
    hour: Option<PyExpr>,
    minute: Option<PyExpr>,
    second: Option<PyExpr>,
    microsecond: Option<PyExpr>,
) -> PyExpr {
    let year = year.inner;
    let month = month.inner;
    let day = day.inner;

    set_unwrapped_or_0!(hour, minute, second, microsecond);

    let args = DatetimeArgs {
        year,
        month,
        day,
        hour,
        minute,
        second,
        microsecond,
    };
    dsl::datetime(args).into()
}

#[pyfunction]
pub fn dtype_cols(dtypes: Vec<Wrap<DataType>>) -> PyResult<PyExpr> {
    let dtypes = vec_extract_wrapped(dtypes);
    Ok(dsl::dtype_cols(dtypes).into())
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn duration(
    days: Option<PyExpr>,
    seconds: Option<PyExpr>,
    nanoseconds: Option<PyExpr>,
    microseconds: Option<PyExpr>,
    milliseconds: Option<PyExpr>,
    minutes: Option<PyExpr>,
    hours: Option<PyExpr>,
    weeks: Option<PyExpr>,
) -> PyExpr {
    set_unwrapped_or_0!(
        days,
        seconds,
        nanoseconds,
        microseconds,
        milliseconds,
        minutes,
        hours,
        weeks,
    );
    let args = DurationArgs {
        days,
        seconds,
        nanoseconds,
        microseconds,
        milliseconds,
        minutes,
        hours,
        weeks,
    };
    dsl::duration(args).into()
}

#[pyfunction]
pub fn first() -> PyExpr {
    dsl::first().into()
}

#[pyfunction]
pub fn last() -> PyExpr {
    dsl::last().into()
}
