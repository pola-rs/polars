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

    fn max(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            py.allow_threads(|| self.series.max_reduce().map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_py(py))
    }

    fn mean(&self, py: Python) -> PyResult<PyObject> {
        match self.series.dtype() {
            Boolean => Ok(Wrap(
                py.allow_threads(|| self.series.cast(&DataType::UInt8).unwrap().mean_reduce())
                    .as_any_value(),
            )
            .into_py(py)),
            // For non-numeric output types we require mean_reduce.
            dt if dt.is_temporal() => Ok(Wrap(
                py.allow_threads(|| self.series.mean_reduce())
                    .as_any_value(),
            )
            .into_py(py)),
            _ => Ok(py.allow_threads(|| self.series.mean()).into_py(py)),
        }
    }

    fn median(&self, py: Python) -> PyResult<PyObject> {
        match self.series.dtype() {
            Boolean => Ok(Wrap(
                py.allow_threads(|| self.series.cast(&DataType::UInt8).unwrap().median_reduce())
                    .map_err(PyPolarsErr::from)?
                    .as_any_value(),
            )
            .into_py(py)),
            // For non-numeric output types we require median_reduce.
            dt if dt.is_temporal() => Ok(Wrap(
                py.allow_threads(|| self.series.median_reduce())
                    .map_err(PyPolarsErr::from)?
                    .as_any_value(),
            )
            .into_py(py)),
            _ => Ok(py.allow_threads(|| self.series.median()).into_py(py)),
        }
    }

    fn min(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            py.allow_threads(|| self.series.min_reduce().map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_py(py))
    }

    fn product(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            py.allow_threads(|| self.series.product().map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_py(py))
    }

    fn quantile(
        &self,
        py: Python,
        quantile: f64,
        interpolation: Wrap<QuantileMethod>,
    ) -> PyResult<PyObject> {
        let bind = py.allow_threads(|| self.series.quantile_reduce(quantile, interpolation.0));
        let sc = bind.map_err(PyPolarsErr::from)?;

        Ok(Wrap(sc.as_any_value()).into_py(py))
    }

    fn std(&self, py: Python, ddof: u8) -> PyResult<PyObject> {
        Ok(Wrap(
            py.allow_threads(|| self.series.std_reduce(ddof).map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_py(py))
    }

    fn var(&self, py: Python, ddof: u8) -> PyResult<PyObject> {
        Ok(Wrap(
            py.allow_threads(|| self.series.var_reduce(ddof).map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_py(py))
    }

    fn sum(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            py.allow_threads(|| self.series.sum_reduce().map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_py(py))
    }

    fn first(&self, py: Python) -> PyObject {
        Wrap(py.allow_threads(|| self.series.first()).as_any_value()).into_py(py)
    }

    fn last(&self, py: Python) -> PyObject {
        Wrap(py.allow_threads(|| self.series.last()).as_any_value()).into_py(py)
    }

    #[cfg(feature = "approx_unique")]
    fn approx_n_unique(&self, py: Python) -> PyResult<PyObject> {
        Ok(py
            .allow_threads(|| self.series.approx_n_unique().map_err(PyPolarsErr::from))?
            .into_py(py))
    }

    #[cfg(feature = "bitwise")]
    fn bitwise_and(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            py.allow_threads(|| self.series.and_reduce().map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_py(py))
    }

    #[cfg(feature = "bitwise")]
    fn bitwise_or(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            py.allow_threads(|| self.series.or_reduce().map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_py(py))
    }

    #[cfg(feature = "bitwise")]
    fn bitwise_xor(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            py.allow_threads(|| self.series.xor_reduce().map_err(PyPolarsErr::from))?
                .as_any_value(),
        )
        .into_py(py))
    }
}
