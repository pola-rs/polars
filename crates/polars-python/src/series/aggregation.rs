use polars::prelude::*;
use pyo3::prelude::*;

use super::PySeries;
use crate::conversion::Wrap;
use crate::utils::EnterPolarsExt;

fn scalar_to_py(scalar: PyResult<Scalar>, py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    Wrap(scalar?.as_any_value()).into_pyobject(py)
}

#[pymethods]
impl PySeries {
    fn any(&self, py: Python<'_>, ignore_nulls: bool) -> PyResult<Option<bool>> {
        py.enter_polars(|| {
            let s = self.series.read();
            let s = s.bool()?;
            PolarsResult::Ok(if ignore_nulls {
                Some(s.any())
            } else {
                s.any_kleene()
            })
        })
    }

    fn all(&self, py: Python<'_>, ignore_nulls: bool) -> PyResult<Option<bool>> {
        py.enter_polars(|| {
            let s = self.series.read();
            let s = s.bool()?;
            PolarsResult::Ok(if ignore_nulls {
                Some(s.all())
            } else {
                s.all_kleene()
            })
        })
    }

    fn is_empty(&self, py: Python<'_>, ignore_nulls: bool) -> PyResult<bool> {
        py.enter_polars(|| {
            let s = self.series.read();
            PolarsResult::Ok(if ignore_nulls {
                s.is_full_null()
            } else {
                s.is_empty()
            })
        })
    }

    fn arg_max(&self, py: Python) -> PyResult<Option<usize>> {
        py.enter_polars(|| {
            let s = self.series.read();
            polars_ensure!(
                !s.dtype().is_object(),
                InvalidOperation: "`arg_max` operation not supported for dtype `{}`",
                s.dtype()
            );
            PolarsResult::Ok(s.arg_max())
        })
    }

    fn arg_min(&self, py: Python) -> PyResult<Option<usize>> {
        py.enter_polars(|| {
            let s = self.series.read();
            polars_ensure!(
                !s.dtype().is_object(),
                InvalidOperation: "`arg_min` operation not supported for dtype `{}`",
                s.dtype()
            );
            PolarsResult::Ok(s.arg_min())
        })
    }

    fn min<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.read().min_reduce()), py)
    }

    fn max<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.read().max_reduce()), py)
    }

    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.read().mean_reduce()), py)
    }

    fn median<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.read().median_reduce()), py)
    }

    fn product<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.read().product()), py)
    }

    fn quantile<'py>(
        &self,
        py: Python<'py>,
        quantile: PySeries,
        interpolation: Wrap<QuantileMethod>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let fraction = quantile.series.into_inner();
        let out = py.enter_polars(|| {
            quantile_reduce_fraction(&self.series.read(), &fraction, interpolation.0)
        });
        scalar_to_py(out, py)
    }

    fn std<'py>(&self, py: Python<'py>, ddof: u8) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.read().std_reduce(ddof)), py)
    }

    fn var<'py>(&self, py: Python<'py>, ddof: u8) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.read().var_reduce(ddof)), py)
    }

    fn sum<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.read().sum_reduce()), py)
    }

    fn first<'py>(&self, py: Python<'py>, ignore_nulls: bool) -> PyResult<Bound<'py, PyAny>> {
        let result = if ignore_nulls {
            py.enter_polars_ok(|| self.series.read().first_non_null())
        } else {
            py.enter_polars_ok(|| self.series.read().first())
        };
        scalar_to_py(result, py)
    }

    fn last<'py>(&self, py: Python<'py>, ignore_nulls: bool) -> PyResult<Bound<'py, PyAny>> {
        let result = if ignore_nulls {
            py.enter_polars_ok(|| self.series.read().last_non_null())
        } else {
            py.enter_polars_ok(|| self.series.read().last())
        };
        scalar_to_py(result, py)
    }

    #[cfg(feature = "approx_unique")]
    fn approx_n_unique(&self, py: Python) -> PyResult<IdxSize> {
        py.enter_polars(|| self.series.read().approx_n_unique())
    }

    #[cfg(feature = "bitwise")]
    fn bitwise_and<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.read().and_reduce()), py)
    }

    #[cfg(feature = "bitwise")]
    fn bitwise_or<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.read().or_reduce()), py)
    }

    #[cfg(feature = "bitwise")]
    fn bitwise_xor<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.read().xor_reduce()), py)
    }
}
