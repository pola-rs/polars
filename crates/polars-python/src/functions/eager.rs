use polars::functions;
use polars::prelude::Expr;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_plan::plans::LiteralValue;
use pyo3::prelude::*;

use crate::conversion::{Wrap, get_df, get_series};
use crate::error::PyPolarsErr;
use crate::utils::EnterPolarsExt;
use crate::{PyDataFrame, PyExpr, PySeries};

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
pub fn concat_df_horizontal(dfs: &Bound<'_, PyAny>, strict: bool) -> PyResult<PyDataFrame> {
    let iter = dfs.try_iter()?;

    let dfs = iter
        .map(|item| {
            let item = item?;
            get_df(&item)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let df =
        functions::concat_df_horizontal(&dfs, true, strict, false).map_err(PyPolarsErr::from)?;
    Ok(df.into())
}

/// Eagerly construct a `Series` of length `n` filled with a single literal value.
///
/// This is a fast option of `pl.repeat(..., eager=True)`. It bypasses the query
/// engine and materializes the Series directly, which reduces the overhead.
///
/// NOTE: `value` must be an `Expr::Literal` holding a *scalar*.
/// list/array/struct literals are scalars, `Series` and `Range` literals aren't.
#[pyfunction]
#[pyo3(signature = (value, n, dtype=None))]
pub fn eager_repeat_fast(
    py: Python<'_>,
    value: PyExpr,
    n: usize,
    dtype: Option<Wrap<DataType>>,
) -> PyResult<PySeries> {
    py.enter_polars(move || {
        // The passed expression should be a LiteralValue which only contains a scalar value
        let Expr::Literal(lv) = &value.inner else {
            polars_bail!(
                ComputeError:
                "eager_repeat_fast expects `value` to be a literal expression, got: {:?}",
                value.inner
            );
        };
        polars_ensure!(
            lv.is_scalar(),
            ShapeMismatch:
            "eager_repeat_fast expects `value` to be a scalar literal, got a {} literal",
            if matches!(lv, LiteralValue::Series(_)) { "Series" } else { "Range" },
        );

        let column = lv.to_column(PlSmallStr::from_static("repeat"))?;

        // Not all dtypes are handled in python (e.g. stripping timezones from datetime)
        // thus cast again, like in `functions::lazy::repeat`
        let column = match dtype {
            Some(dtype) => column.cast_with_options(&dtype.0, CastOptions::NonStrict)?,
            None => column,
        };

        PolarsResult::Ok(column.new_from_index(0, n).take_materialized_series())
    })
    .map(PySeries::from)
}
