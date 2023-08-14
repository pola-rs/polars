use pyo3::prelude::*;

use crate::error::PyPolarsErr;
use crate::prelude::*;
use crate::PySeries;

#[pymethods]
impl PySeries {
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

    fn sum(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .sum_as_series()
                .get(0)
                .map_err(PyPolarsErr::from)?,
        )
        .into_py(py))
    }
}
