use polars::time;
use polars_core::datatypes::{TimeUnit, TimeZone};
use polars_core::prelude::IntoSeries;
use pyo3::prelude::*;

use crate::conversion::Wrap;
use crate::error::PyPolarsErr;
use crate::prelude::{ClosedWindow, Duration};
use crate::PySeries;

#[pyfunction]
pub fn date_range(
    start: i64,
    stop: i64,
    every: &str,
    closed: Wrap<ClosedWindow>,
    name: &str,
    time_unit: Wrap<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> PyResult<PySeries> {
    let date_range = time::date_range_impl(
        name,
        start,
        stop,
        Duration::parse(every),
        closed.0,
        time_unit.0,
        time_zone.as_ref(),
    )
    .map_err(PyPolarsErr::from)?;
    Ok(date_range.into_series().into())
}
