use polars::lazy::dsl;
use polars_core::with_match_physical_integer_polars_type;
use polars_ops::series::ClosedInterval;
use pyo3::prelude::*;

use crate::error::PyPolarsErr;
use crate::expr::datatype::PyDataTypeExpr;
use crate::prelude::*;
use crate::utils::EnterPolarsExt;
use crate::{PyExpr, PySeries};

#[pyfunction]
pub fn int_range(start: PyExpr, end: PyExpr, step: i64, dtype: PyDataTypeExpr) -> PyExpr {
    let start = start.inner;
    let end = end.inner;
    let dtype = dtype.inner;
    dsl::int_range(start, end, step, dtype).into()
}

/// Eager version of `int_range` to avoid overhead from the expression engine.
#[pyfunction]
pub fn eager_int_range(
    py: Python<'_>,
    lower: &Bound<'_, PyAny>,
    upper: &Bound<'_, PyAny>,
    step: &Bound<'_, PyAny>,
    dtype: PyDataTypeExpr,
) -> PyResult<PySeries> {
    let dtype = dtype.inner;
    let Some(dtype) = dtype.into_literal() else {
        return Err(PyPolarsErr::from(
            polars_err!(ComputeError: "eager `int_range` cannot be given lazy datatype expression"),
        )
        .into());
    };

    if !dtype.is_integer() {
        return Err(PyPolarsErr::from(
            polars_err!(SchemaMismatch: "non-integer `dtype` passed to `int_range`: '{}'", dtype),
        )
        .into());
    }

    with_match_physical_integer_polars_type!(dtype, |$T| {
        let start_v: <$T as PolarsNumericType>::Native = lower.extract()?;
        let end_v: <$T as PolarsNumericType>::Native = upper.extract()?;
        let step: i64 = step.extract()?;
        py.enter_polars_series(|| new_int_range::<$T>(start_v, end_v, step, PlSmallStr::from_static("literal")))
    })
}

#[pyfunction]
pub fn int_ranges(
    start: PyExpr,
    end: PyExpr,
    step: PyExpr,
    dtype: PyDataTypeExpr,
) -> PyResult<PyExpr> {
    let dtype = dtype.inner;
    Ok(dsl::int_ranges(start.inner, end.inner, step.inner, dtype).into())
}

#[pyfunction]
pub fn date_range(
    start: PyExpr,
    end: PyExpr,
    interval: &str,
    closed: Wrap<ClosedWindow>,
) -> PyResult<PyExpr> {
    let start = start.inner;
    let end = end.inner;
    let interval = Duration::try_parse(interval).map_err(PyPolarsErr::from)?;
    let closed = closed.0;
    Ok(dsl::date_range(start, end, interval, closed).into())
}

#[pyfunction]
pub fn date_ranges(
    start: PyExpr,
    end: PyExpr,
    interval: &str,
    closed: Wrap<ClosedWindow>,
) -> PyResult<PyExpr> {
    let start = start.inner;
    let end = end.inner;
    let interval = Duration::try_parse(interval).map_err(PyPolarsErr::from)?;
    let closed = closed.0;
    Ok(dsl::date_ranges(start, end, interval, closed).into())
}

#[pyfunction]
#[pyo3(signature = (start, end, every, closed, time_unit, time_zone))]
pub fn datetime_range(
    start: PyExpr,
    end: PyExpr,
    every: &str,
    closed: Wrap<ClosedWindow>,
    time_unit: Option<Wrap<TimeUnit>>,
    time_zone: Wrap<Option<TimeZone>>,
) -> PyResult<PyExpr> {
    let start = start.inner;
    let end = end.inner;
    let every = Duration::try_parse(every).map_err(PyPolarsErr::from)?;
    let closed = closed.0;
    let time_unit = time_unit.map(|x| x.0);
    let time_zone = time_zone.0;
    Ok(dsl::datetime_range(start, end, every, closed, time_unit, time_zone).into())
}

#[pyfunction]
#[pyo3(signature = (start, end, every, closed, time_unit, time_zone))]
pub fn datetime_ranges(
    start: PyExpr,
    end: PyExpr,
    every: &str,
    closed: Wrap<ClosedWindow>,
    time_unit: Option<Wrap<TimeUnit>>,
    time_zone: Wrap<Option<TimeZone>>,
) -> PyResult<PyExpr> {
    let start = start.inner;
    let end = end.inner;
    let every = Duration::try_parse(every).map_err(PyPolarsErr::from)?;
    let closed = closed.0;
    let time_unit = time_unit.map(|x| x.0);
    let time_zone = time_zone.0;
    Ok(dsl::datetime_ranges(start, end, every, closed, time_unit, time_zone).into())
}

#[pyfunction]
pub fn time_range(
    start: PyExpr,
    end: PyExpr,
    every: &str,
    closed: Wrap<ClosedWindow>,
) -> PyResult<PyExpr> {
    let start = start.inner;
    let end = end.inner;
    let every = Duration::try_parse(every).map_err(PyPolarsErr::from)?;
    let closed = closed.0;
    Ok(dsl::time_range(start, end, every, closed).into())
}

#[pyfunction]
pub fn time_ranges(
    start: PyExpr,
    end: PyExpr,
    every: &str,
    closed: Wrap<ClosedWindow>,
) -> PyResult<PyExpr> {
    let start = start.inner;
    let end = end.inner;
    let every = Duration::try_parse(every).map_err(PyPolarsErr::from)?;
    let closed = closed.0;
    Ok(dsl::time_ranges(start, end, every, closed).into())
}

#[pyfunction]
pub fn linear_space(
    start: PyExpr,
    end: PyExpr,
    num_samples: PyExpr,
    closed: Wrap<ClosedInterval>,
) -> PyResult<PyExpr> {
    let start = start.inner;
    let end = end.inner;
    let num_samples = num_samples.inner;
    let closed = closed.0;
    Ok(dsl::linear_space(start, end, num_samples, closed).into())
}

#[pyfunction]
pub fn linear_spaces(
    start: PyExpr,
    end: PyExpr,
    num_samples: PyExpr,
    closed: Wrap<ClosedInterval>,
    as_array: bool,
) -> PyResult<PyExpr> {
    let start = start.inner;
    let end = end.inner;
    let num_samples = num_samples.inner;
    let closed = closed.0;
    let out =
        dsl::linear_spaces(start, end, num_samples, closed, as_array).map_err(PyPolarsErr::from)?;
    Ok(out.into())
}
