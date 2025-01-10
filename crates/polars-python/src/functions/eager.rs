use polars::functions;
use polars_core::utils::accumulate_dataframes_vertical;
use pyo3::prelude::*;

use crate::conversion::{get_df, get_series};
use crate::error::PyPolarsErr;
use crate::{PyDataFrame, PySeries};

#[pyfunction]
pub fn concat_df(dfs: &Bound<'_, PyAny>, _py: Python) -> PyResult<PyDataFrame> {
    let mut iter = dfs.try_iter()?;
    let first = iter.next().unwrap()?;

    let first_rdf = get_df(&first)?;

    let mut rdfs = vec![first_rdf];
    for item in iter {
        rdfs.push(get_df(&item?)?);
    }
    accumulate_dataframes_vertical(rdfs)
        .map(Into::into)
        .map_err(PyPolarsErr::from)
        .map_err(PyErr::from)
}

#[pyfunction]
pub fn concat_series(series: &Bound<'_, PyAny>) -> PyResult<PySeries> {
    let mut iter = series.try_iter()?;
    let first = iter.next().unwrap()?;

    let mut s = get_series(&first)?;

    for res in iter {
        let item = res?;
        let item = get_series(&item)?;
        s.append(&item).map_err(PyPolarsErr::from)?;
    }
    Ok(s.into())
}

#[pyfunction]
pub fn concat_df_diagonal(dfs: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
    let iter = dfs.try_iter()?;

    let dfs = iter
        .map(|item| {
            let item = item?;
            get_df(&item)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let df = functions::concat_df_diagonal(&dfs).map_err(PyPolarsErr::from)?;
    Ok(df.into())
}

#[pyfunction]
pub fn concat_df_horizontal(dfs: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
    let iter = dfs.try_iter()?;

    let dfs = iter
        .map(|item| {
            let item = item?;
            get_df(&item)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let df = functions::concat_df_horizontal(&dfs, true).map_err(PyPolarsErr::from)?;
    Ok(df.into())
}
