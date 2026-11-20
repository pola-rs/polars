use polars::functions;
use polars::series::IsSorted;
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

    // A Dictionary first input appends as columns (without materialising); anything else keeps
    // the Series path (a Scalar column appended to would keep fewer chunks than its Series).
    let get_column = |obj: &Bound<'_, PyAny>| -> PyResult<Column> {
        let s = obj.getattr(pyo3::intern!(obj.py(), "_s"))?;
        Ok(s.extract::<PyRef<PySeries>>()?.column.read().clone())
    };
    if let Some(mut c) = first
        .getattr(pyo3::intern!(py, "_s"))?
        .extract::<PyRef<PySeries>>()?
        .compressed()
        .filter(|c| matches!(c, Column::Dictionary(_)))
    {
        let rest = iter
            .map(|item| get_column(&item?))
            .collect::<PyResult<Vec<_>>>()?;
        // `DictionaryColumn::try_append` drops the sorted flag, which a Series append keeps
        // when both sides are sorted: those take the Series path.
        // (A Series of at most one row always reports sorted.)
        let sorted = |c: &Column| c.len() <= 1 || c.is_sorted_flag() != IsSorted::Not;
        let all_sorted = sorted(&c) && rest.iter().all(sorted);
        if !all_sorted {
            return py.enter_polars_column(move || {
                for item in &rest {
                    c.append(item)?;
                }
                Ok(c)
            });
        }
        let mut s = c.take_materialized_series();
        return py.enter_polars_series(move || {
            for item in &rest {
                s.append(item.as_materialized_series())?;
            }
            Ok(s)
        });
    }

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
