use polars::functions;
use polars_core::prelude::*;
use pyo3::prelude::*;

use crate::conversion::{get_df, get_series};
use crate::utils::EnterPolarsExt;
use crate::{PyDataFrame, PySeries};

#[pyfunction]
pub fn concat_df(dfs: &Bound<'_, PyAny>, py: Python) -> PyResult<PyDataFrame> {
    use polars_core::error::PolarsResult;
    use polars_core::utils::rayon::prelude::*;

    let mut iter = dfs.try_iter()?;
    let first = iter.next().unwrap()?;

    let first_rdf = get_df(&first)?;
    let identity_df = first_rdf.clear();

    let mut rdfs: Vec<PolarsResult<DataFrame>> = vec![Ok(first_rdf)];

    for item in iter {
        let rdf = get_df(&item?)?;
        rdfs.push(Ok(rdf));
    }

    let identity = || Ok(identity_df.clone());

    py.enter_polars_df(|| {
        polars_core::runtime::RAYON.install(|| {
            rdfs.into_par_iter()
                .fold(identity, |acc: PolarsResult<DataFrame>, df| {
                    let mut acc: DataFrame = acc?;
                    acc.vstack_mut_owned(df?)?;
                    Ok(acc)
                })
                .reduce(identity, |acc, df| {
                    let mut acc = acc?;
                    acc.vstack_mut_owned(df?)?;
                    Ok(acc)
                })
        })
    })
}

#[pyfunction]
pub fn concat_series(py: Python<'_>, series: &Bound<'_, PyAny>) -> PyResult<PySeries> {
    let mut iter = series.try_iter()?;
    let first = iter.next().unwrap()?;

    let mut s = get_series(&first)?;
    let rest = iter
        .map(|item| {
            let item = item?;
            get_series(&item)
        })
        .collect::<PyResult<Vec<_>>>()?;

    py.enter_polars_series(move || {
        for item in &rest {
            s.append(item)?;
        }
        Ok(s)
    })
}

#[pyfunction]
pub fn concat_df_diagonal(py: Python<'_>, dfs: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
    let iter = dfs.try_iter()?;

    let dfs = iter
        .map(|item| {
            let item = item?;
            get_df(&item)
        })
        .collect::<PyResult<Vec<_>>>()?;

    py.enter_polars_df(move || functions::concat_df_diagonal(&dfs))
}

#[pyfunction]
pub fn concat_df_horizontal(
    py: Python<'_>,
    dfs: &Bound<'_, PyAny>,
    strict: bool,
) -> PyResult<PyDataFrame> {
    let iter = dfs.try_iter()?;

    let dfs = iter
        .map(|item| {
            let item = item?;
            get_df(&item)
        })
        .collect::<PyResult<Vec<_>>>()?;

    py.enter_polars_df(move || functions::concat_df_horizontal(&dfs, true, strict, false))
}
