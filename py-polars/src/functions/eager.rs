use polars::{functions, time};
use polars_core::datatypes::{TimeUnit, TimeZone};
use polars_core::prelude::*;
use pyo3::prelude::*;

use crate::conversion::{get_df, get_series, Wrap};
use crate::error::PyPolarsErr;
use crate::prelude::{ClosedWindow, Duration};
use crate::{PyDataFrame, PySeries};

#[pyfunction]
pub fn concat_df(dfs: &PyAny, py: Python) -> PyResult<PyDataFrame> {
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
pub fn concat_series(series: &PyAny) -> PyResult<PySeries> {
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

#[pyfunction]
pub fn date_range_eager(
    start: i64,
    stop: i64,
    every: &str,
    closed: Wrap<ClosedWindow>,
    time_unit: Wrap<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> PyResult<PySeries> {
    let date_range = time::date_range_impl(
        "date",
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

#[pyfunction]
pub fn diag_concat_df(dfs: &PyAny) -> PyResult<PyDataFrame> {
    let iter = dfs.iter()?;

    let dfs = iter
        .map(|item| {
            let item = item?;
            get_df(item)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let df = functions::diag_concat_df(&dfs).map_err(PyPolarsErr::from)?;
    Ok(df.into())
}

#[pyfunction]
pub fn hor_concat_df(dfs: &PyAny) -> PyResult<PyDataFrame> {
    let iter = dfs.iter()?;

    let dfs = iter
        .map(|item| {
            let item = item?;
            get_df(item)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let df = functions::hor_concat_df(&dfs).map_err(PyPolarsErr::from)?;
    Ok(df.into())
}

#[pyfunction]
pub fn repeat_eager(
    value: Wrap<AnyValue>,
    n: usize,
    dtype: Option<Wrap<DataType>>,
) -> PyResult<PySeries> {
    let value = value.0;
    let dtype = match dtype.map(|wrap| wrap.0) {
        Some(dtype) => dtype,
        None => match value.dtype() {
            // Integer inputs that fit in Int32 are parsed as such
            DataType::Int64 => {
                let int_value: i64 = value.try_extract().unwrap();
                if int_value >= i32::MIN as i64 && int_value <= i32::MAX as i64 {
                    DataType::Int32
                } else {
                    DataType::Int64
                }
            }
            DataType::Unknown => DataType::Null,
            _ => value.dtype(),
        },
    };

    Ok(Series::new("repeat", &[value])
        .cast(&dtype)
        .map_err(PyPolarsErr::from)?
        .new_from_index(0, n)
        .into())
}

#[pyfunction]
pub fn time_range(
    start: i64,
    stop: i64,
    every: &str,
    closed: Wrap<ClosedWindow>,
) -> PyResult<PySeries> {
    let time_range = time::time_range_impl("time", start, stop, Duration::parse(every), closed.0)
        .map_err(PyPolarsErr::from)?;
    Ok(time_range.into_series().into())
}
