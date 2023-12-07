use polars::lazy::dsl;
use pyo3::prelude::*;

use crate::prelude::*;
use crate::PyExpr;

#[pyfunction]
pub fn int_range(start: PyExpr, end: PyExpr, step: i64, dtype: Option<Wrap<DataType>>) -> PyExpr {
    let mut result = dsl::int_range(start.inner, end.inner, step);

    if let Some(dt) = dtype {
        let dt = dt.0;
        result = result.cast(dt);
    }

    result.into()
}

#[pyfunction]
pub fn int_ranges(start: PyExpr, end: PyExpr, step: i64, dtype: Wrap<DataType>) -> PyExpr {
    let dtype = dtype.0;

    let mut result = dsl::int_ranges(start.inner, end.inner, step);

    if dtype != DataType::Int64 {
        result = result.cast(DataType::List(Box::new(dtype)))
    }

    result.into()
}

#[pyfunction]
pub fn date_range(
    start: PyExpr,
    end: PyExpr,
    every: &str,
    closed: Wrap<ClosedWindow>,
    time_unit: Option<Wrap<TimeUnit>>,
    time_zone: Option<TimeZone>,
) -> PyExpr {
    let start = start.inner;
    let end = end.inner;
    let every = Duration::parse(every);
    let closed = closed.0;
    let time_unit = time_unit.map(|x| x.0);
    dsl::date_range(start, end, every, closed, time_unit, time_zone).into()
}

#[pyfunction]
pub fn date_ranges(
    start: PyExpr,
    end: PyExpr,
    every: &str,
    closed: Wrap<ClosedWindow>,
    time_unit: Option<Wrap<TimeUnit>>,
    time_zone: Option<TimeZone>,
) -> PyExpr {
    let start = start.inner;
    let end = end.inner;
    let every = Duration::parse(every);
    let closed = closed.0;
    let time_unit = time_unit.map(|x| x.0);
    dsl::date_ranges(start, end, every, closed, time_unit, time_zone).into()
}

#[pyfunction]
pub fn datetime_range(
    start: PyExpr,
    end: PyExpr,
    every: &str,
    closed: Wrap<ClosedWindow>,
    time_unit: Option<Wrap<TimeUnit>>,
    time_zone: Option<TimeZone>,
) -> PyExpr {
    let start = start.inner;
    let end = end.inner;
    let every = Duration::parse(every);
    let closed = closed.0;
    let time_unit = time_unit.map(|x| x.0);
    dsl::datetime_range(start, end, every, closed, time_unit, time_zone).into()
}

#[pyfunction]
pub fn datetime_ranges(
    start: PyExpr,
    end: PyExpr,
    every: &str,
    closed: Wrap<ClosedWindow>,
    time_unit: Option<Wrap<TimeUnit>>,
    time_zone: Option<TimeZone>,
) -> PyExpr {
    let start = start.inner;
    let end = end.inner;
    let every = Duration::parse(every);
    let closed = closed.0;
    let time_unit = time_unit.map(|x| x.0);
    dsl::datetime_ranges(start, end, every, closed, time_unit, time_zone).into()
}

#[pyfunction]
pub fn time_range(start: PyExpr, end: PyExpr, every: &str, closed: Wrap<ClosedWindow>) -> PyExpr {
    let start = start.inner;
    let end = end.inner;
    let every = Duration::parse(every);
    let closed = closed.0;
    dsl::time_range(start, end, every, closed).into()
}

#[pyfunction]
pub fn time_ranges(start: PyExpr, end: PyExpr, every: &str, closed: Wrap<ClosedWindow>) -> PyExpr {
    let start = start.inner;
    let end = end.inner;
    let every = Duration::parse(every);
    let closed = closed.0;
    dsl::time_ranges(start, end, every, closed).into()
}
