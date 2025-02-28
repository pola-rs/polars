use polars::prelude::*;
use pyo3::prelude::*;
use DataType::*;

use super::PySeries;
use crate::conversion::Wrap;
use crate::utils::EnterPolarsExt;

fn scalar_to_py(scalar: PyResult<Scalar>, py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    Wrap(scalar?.as_any_value()).into_pyobject(py)
}

#[pymethods]
impl PySeries {
    fn any(&self, py: Python, ignore_nulls: bool) -> PyResult<Option<bool>> {
        py.enter_polars(|| {
            let s = self.series.bool()?;
            PolarsResult::Ok(if ignore_nulls {
                Some(s.any())
            } else {
                s.any_kleene()
            })
        })
    }

    fn all(&self, py: Python, ignore_nulls: bool) -> PyResult<Option<bool>> {
        py.enter_polars(|| {
            let s = self.series.bool()?;
            PolarsResult::Ok(if ignore_nulls {
                Some(s.all())
            } else {
                s.all_kleene()
            })
        })
    }

    fn arg_max(&self, py: Python) -> PyResult<Option<usize>> {
        py.enter_polars_ok(|| self.series.arg_max())
    }

    fn arg_min(&self, py: Python) -> PyResult<Option<usize>> {
        py.enter_polars_ok(|| self.series.arg_min())
    }

    fn min<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.min_reduce()), py)
    }

    fn max<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.max_reduce()), py)
    }

    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.series.dtype() {
            Boolean => scalar_to_py(
                py.enter_polars_ok(|| self.series.cast(&DataType::UInt8).unwrap().mean_reduce()),
                py,
            ),
            // For non-numeric output types we require mean_reduce.
            dt if dt.is_temporal() => {
                scalar_to_py(py.enter_polars_ok(|| self.series.mean_reduce()), py)
            },
            _ => Ok(self.series.mean().into_pyobject(py)?),
        }
    }

    fn median<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.series.dtype() {
            Boolean => scalar_to_py(
                py.enter_polars(|| self.series.cast(&DataType::UInt8).unwrap().median_reduce()),
                py,
            ),
            // For non-numeric output types we require median_reduce.
            dt if dt.is_temporal() => {
                scalar_to_py(py.enter_polars(|| self.series.median_reduce()), py)
            },
            _ => Ok(self.series.median().into_pyobject(py)?),
        }
    }

    fn product<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.product()), py)
    }

    fn quantile<'py>(
        &self,
        py: Python<'py>,
        quantile: f64,
        interpolation: Wrap<QuantileMethod>,
    ) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(
            py.enter_polars(|| self.series.quantile_reduce(quantile, interpolation.0)),
            py,
        )
    }

    fn std<'py>(&self, py: Python<'py>, ddof: u8) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.std_reduce(ddof)), py)
    }

    fn var<'py>(&self, py: Python<'py>, ddof: u8) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.var_reduce(ddof)), py)
    }

    fn sum<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.sum_reduce()), py)
    }

    fn first<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars_ok(|| self.series.first()), py)
    }

    fn last<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars_ok(|| self.series.last()), py)
    }

    #[cfg(feature = "approx_unique")]
    fn approx_n_unique(&self, py: Python) -> PyResult<IdxSize> {
        py.enter_polars(|| self.series.approx_n_unique())
    }

    #[cfg(feature = "bitwise")]
    fn bitwise_and<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.and_reduce()), py)
    }

    #[cfg(feature = "bitwise")]
    fn bitwise_or<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.or_reduce()), py)
    }

    #[cfg(feature = "bitwise")]
    fn bitwise_xor<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series.xor_reduce()), py)
    }
}
