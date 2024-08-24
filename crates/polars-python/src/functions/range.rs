use polars::lazy::dsl;
use polars_core::with_match_physical_integer_polars_type;
use pyo3::prelude::*;

use crate::error::PyPolarsErr;
use crate::prelude::*;
use crate::{PyExpr, PySeries};

#[pyfunction]
pub fn int_range(start: PyExpr, end: PyExpr, step: i64, dtype: Wrap<DataType>) -> PyExpr {
    let start = start.inner;
    let end = end.inner;
    let dtype = dtype.0;
    dsl::int_range(start, end, step, dtype).into()
}

/// Eager version of `int_range` to avoid overhead from the expression engine.
#[pyfunction]
pub fn eager_int_range(
    lower: &Bound<'_, PyAny>,
    upper: &Bound<'_, PyAny>,
    step: &Bound<'_, PyAny>,
    dtype: Wrap<DataType>,
) -> PyResult<PySeries> {
    let dtype = dtype.0;
    if !dtype.is_integer() {
        return Err(PyPolarsErr::from(
            polars_err!(ComputeError: "non-integer `dtype` passed to `int_range`: {:?}", dtype),
        )
        .into());
    }

    let ret = with_match_physical_integer_polars_type!(dtype, |$T| {
        let start_v: <$T as PolarsNumericType>::Native = lower.extract()?;
        let end_v: <$T as PolarsNumericType>::Native = upper.extract()?;
        let step: i64 = step.extract()?;
        new_int_range::<$T>(start_v, end_v, step, "literal")
    });

    let s = ret.map_err(PyPolarsErr::from)?;
    Ok(s.into())
}

#[pyfunction]
pub fn int_ranges(
    start: PyExpr,
    end: PyExpr,
    step: PyExpr,
    dtype: Wrap<DataType>,
) -> PyResult<PyExpr> {
    let dtype = dtype.0;
    if !dtype.is_integer() {
        return Err(PyPolarsErr::from(
            polars_err!(ComputeError: "non-integer `dtype` passed to `int_ranges`: {:?}", dtype),
        )
        .into());
    }

    let mut result = dsl::int_ranges(start.inner, end.inner, step.inner);

    if dtype != DataType::Int64 {
        result = result.cast(DataType::List(Box::new(dtype)))
    }

    Ok(result.into())
}

#[pyfunction]
pub fn date_range(
    start: PyExpr,
    end: PyExpr,
    interval: &str,
    closed: Wrap<ClosedWindow>,
) -> PyExpr {
    let start = start.inner;
    let end = end.inner;
    let interval = Duration::parse(interval);
    let closed = closed.0;
    dsl::date_range(start, end, interval, closed).into()
}

#[pyfunction]
pub fn date_ranges(
    start: PyExpr,
    end: PyExpr,
    interval: &str,
    closed: Wrap<ClosedWindow>,
) -> PyExpr {
    let start = start.inner;
    let end = end.inner;
    let interval = Duration::parse(interval);
    let closed = closed.0;
    dsl::date_ranges(start, end, interval, closed).into()
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
