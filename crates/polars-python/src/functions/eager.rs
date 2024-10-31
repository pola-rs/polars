use polars::functions;
use polars_core::prelude::*;
use pyo3::prelude::*;

use crate::conversion::{get_df, get_series};
use crate::error::PyPolarsErr;
use crate::{PyDataFrame, PySeries};

#[pyfunction]
pub fn concat_df(dfs: &Bound<'_, PyAny>, py: Python) -> PyResult<PyDataFrame> {
    use polars_core::error::PolarsResult;
    use polars_core::utils::rayon::prelude::*;

    let mut iter = dfs.iter()?;
    let first = iter.next().unwrap()?;

    let first_rdf = get_df(&first)?;
    let identity_df = first_rdf.clear();

    let mut rdfs: Vec<PolarsResult<DataFrame>> = vec![Ok(first_rdf)];

    for item in iter {
        let rdf = get_df(&item?)?;
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
pub fn concat_series(series: &Bound<'_, PyAny>) -> PyResult<PySeries> {
    let mut iter = series.iter()?;
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
    let iter = dfs.iter()?;

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
    let iter = dfs.iter()?;

    let dfs = iter
        .map(|item| {
            let item = item?;
            get_df(&item)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let df = functions::concat_df_horizontal(&dfs, true).map_err(PyPolarsErr::from)?;
    Ok(df.into())
}
