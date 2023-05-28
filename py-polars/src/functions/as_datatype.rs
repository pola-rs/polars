use polars::lazy::dsl;
use pyo3::prelude::*;

use crate::expr::ToExprs;
use crate::prelude::{DatetimeArgs, DurationArgs};
use crate::{PyExpr, PyPolarsErr};

macro_rules! set_unwrapped_or_0 {
    ($($var:ident),+ $(,)?) => {
        $(let $var = $var.map(|e| e.inner).unwrap_or(dsl::lit(0));)+
    };
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
pub fn concat_list(exprs: Vec<PyExpr>) -> PyResult<PyExpr> {
    let exprs = exprs.to_exprs();
    let expr = dsl::concat_list(exprs).map_err(PyPolarsErr::from)?;
    Ok(expr.into())
}

#[pyfunction]
pub fn as_struct(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    dsl::as_struct(&exprs).into()
}

#[pyfunction]
pub fn concat_str(exprs: Vec<PyExpr>, separator: &str) -> PyExpr {
    let exprs = exprs.to_exprs();
    dsl::concat_str(exprs, separator).into()
}
