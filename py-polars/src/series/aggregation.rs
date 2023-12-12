use polars_ops::chunked_array::nan_propagating_aggregate::*;
use pyo3::prelude::*;

use crate::error::PyPolarsErr;
use crate::prelude::*;
use crate::PySeries;

#[pymethods]
impl PySeries {
    fn any(&self) -> Option<bool> {
        match self.series.dtype() {
            DataType::Boolean => Some(self.series.bool().unwrap().any()),
            _ => None,
        }
    }

    fn all(&self) -> Option<bool> {
        match self.series.dtype() {
            DataType::Boolean => Some(self.series.bool().unwrap().all()),
            _ => None,
        }
    }

    fn arg_max(&self) -> Option<usize> {
        self.series.arg_max()
    }

    fn arg_min(&self) -> Option<usize> {
        self.series.arg_min()
    }

    fn max(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .max_as_series()
                .get(0)
                .map_err(PyPolarsErr::from)?,
        )
        .into_py(py))
    }

    fn mean(&self) -> Option<f64> {
        match self.series.dtype() {
            DataType::Boolean => {
                let s = self.series.cast(&DataType::UInt8).unwrap();
                s.mean()
            },
            _ => self.series.mean(),
        }
    }

    fn median(&self) -> Option<f64> {
        match self.series.dtype() {
            DataType::Boolean => {
                let s = self.series.cast(&DataType::UInt8).unwrap();
                s.median()
            },
            _ => self.series.median(),
        }
    }

    fn min(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .min_as_series()
                .get(0)
                .map_err(PyPolarsErr::from)?,
        )
        .into_py(py))
    }

    fn nan_min(&self, py: Python) -> PyResult<PyObject> {
        match self.series.dtype() {
            DataType::Float32 | DataType::Float64 => Ok(Wrap(
                nan_min_s(&self.series, self.name())
                    .get(0)
                    .map_err(PyPolarsErr::from)?,
            )
            .into_py(py)),
            _ => self.min(py),
        }
    }

    fn nan_max(&self, py: Python) -> PyResult<PyObject> {
        match self.series.dtype() {
            DataType::Float32 | DataType::Float64 => Ok(Wrap(
                nan_max_s(&self.series, self.name())
                    .get(0)
                    .map_err(PyPolarsErr::from)?,
            )
            .into_py(py)),
            _ => self.max(py),
        }
    }

    fn product(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(self.series.product().get(0).map_err(PyPolarsErr::from)?).into_py(py))
    }

    fn quantile(&self, quantile: f64, interpolation: Wrap<QuantileInterpolOptions>) -> PyObject {
        Python::with_gil(|py| {
            Wrap(
                self.series
                    .quantile_as_series(quantile, interpolation.0)
                    .expect("invalid quantile")
                    .get(0)
                    .unwrap_or(AnyValue::Null),
            )
            .into_py(py)
        })
    }

    fn std(&self, py: Python, ddof: u8) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .std_as_series(ddof)
                .get(0)
                .map_err(PyPolarsErr::from)?,
        )
        .into_py(py))
    }

    fn var(&self, py: Python, ddof: u8) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .var_as_series(ddof)
                .get(0)
                .map_err(PyPolarsErr::from)?,
        )
        .into_py(py))
    }

    fn sum(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .sum_as_series()
                .get(0)
                .map_err(PyPolarsErr::from)?,
        )
        .into_py(py))
    }

    fn entropy(&self, base: f64, normalize: bool) -> Option<f64> {
        self.series.entropy(base, normalize)
    }
}
