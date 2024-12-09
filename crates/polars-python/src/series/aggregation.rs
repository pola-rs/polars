use polars::prelude::*;
use pyo3::prelude::*;
use DataType::*;

use super::PySeries;
use crate::conversion::Wrap;
use crate::error::PyPolarsErr;

#[pymethods]
impl PySeries {
    fn any(&self, py: Python, ignore_nulls: bool) -> PyResult<Option<bool>> {
        py.allow_threads(|| {
            let s = self.series.bool().map_err(PyPolarsErr::from)?;
            Ok(if ignore_nulls {
                Some(s.any())
            } else {
                s.any_kleene()
            })
        })
    }

    fn all(&self, py: Python, ignore_nulls: bool) -> PyResult<Option<bool>> {
        py.allow_threads(|| {
            let s = self.series.bool().map_err(PyPolarsErr::from)?;
            Ok(if ignore_nulls {
                Some(s.all())
            } else {
                s.all_kleene()
            })
        })
    }

    fn arg_max(&self, py: Python) -> Option<usize> {
        py.allow_threads(|| self.series.arg_max())
    }

    fn arg_min(&self, py: Python) -> Option<usize> {
        py.allow_threads(|| self.series.arg_min())
    }

    fn max<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Wrap(
            py.allow_threads(|| self.series.max_reduce().map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_pyobject(py)
    }

    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.series.dtype() {
            Boolean => Wrap(
                py.allow_threads(|| self.series.cast(&DataType::UInt8).unwrap().mean_reduce())
                    .as_any_value(),
            )
            .into_pyobject(py),
            // For non-numeric output types we require mean_reduce.
            dt if dt.is_temporal() => Wrap(
                py.allow_threads(|| self.series.mean_reduce())
                    .as_any_value(),
            )
            .into_pyobject(py),
            _ => py
                .allow_threads(|| self.series.mean())
                .into_pyobject(py)
                .map_err(Into::into),
        }
    }

    fn median<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.series.dtype() {
            Boolean => Wrap(
                py.allow_threads(|| self.series.cast(&DataType::UInt8).unwrap().median_reduce())
                    .map_err(PyPolarsErr::from)?
                    .as_any_value(),
            )
            .into_pyobject(py),
            // For non-numeric output types we require median_reduce.
            dt if dt.is_temporal() => Wrap(
                py.allow_threads(|| self.series.median_reduce())
                    .map_err(PyPolarsErr::from)?
                    .as_any_value(),
            )
            .into_pyobject(py),
            _ => py
                .allow_threads(|| self.series.median())
                .into_pyobject(py)
                .map_err(Into::into),
        }
    }

    fn min<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Wrap(
            py.allow_threads(|| self.series.min_reduce().map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_pyobject(py)
    }

    fn product<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Wrap(
            py.allow_threads(|| self.series.product().map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_pyobject(py)
    }

    fn quantile<'py>(
        &self,
        py: Python<'py>,
        quantile: f64,
        interpolation: Wrap<QuantileMethod>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let bind = py.allow_threads(|| self.series.quantile_reduce(quantile, interpolation.0));
        let sc = bind.map_err(PyPolarsErr::from)?;

        Wrap(sc.as_any_value()).into_pyobject(py)
    }

    fn std<'py>(&self, py: Python<'py>, ddof: u8) -> PyResult<Bound<'py, PyAny>> {
        Wrap(
            py.allow_threads(|| self.series.std_reduce(ddof).map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_pyobject(py)
    }

    fn var<'py>(&self, py: Python<'py>, ddof: u8) -> PyResult<Bound<'py, PyAny>> {
        Wrap(
            py.allow_threads(|| self.series.var_reduce(ddof).map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_pyobject(py)
    }

    fn sum<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Wrap(
            py.allow_threads(|| self.series.sum_reduce().map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_pyobject(py)
    }

    fn first<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Wrap(py.allow_threads(|| self.series.first()).as_any_value()).into_pyobject(py)
    }

    fn last<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Wrap(py.allow_threads(|| self.series.last()).as_any_value()).into_pyobject(py)
    }

    #[cfg(feature = "approx_unique")]
    fn approx_n_unique(&self, py: Python) -> Result<IdxSize, PyPolarsErr> {
        py.allow_threads(|| self.series.approx_n_unique().map_err(PyPolarsErr::from))
    }

    #[cfg(feature = "bitwise")]
    fn bitwise_and<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Wrap(
            py.allow_threads(|| self.series.and_reduce().map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_pyobject(py)
    }

    #[cfg(feature = "bitwise")]
    fn bitwise_or<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Wrap(
            py.allow_threads(|| self.series.or_reduce().map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_pyobject(py)
    }

    #[cfg(feature = "bitwise")]
    fn bitwise_xor<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Wrap(
            py.allow_threads(|| self.series.xor_reduce().map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_pyobject(py)
    }
}
