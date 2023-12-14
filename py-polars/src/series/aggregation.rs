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
                .map_err(PyPolarsErr::from)?
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
                .map_err(PyPolarsErr::from)?
                .get(0)
                .map_err(PyPolarsErr::from)?,
        )
        .into_py(py))
    }

    fn quantile(
        &self,
        quantile: f64,
        interpolation: Wrap<QuantileInterpolOptions>,
    ) -> PyResult<PyObject> {
        let tmp = self
            .series
            .quantile_as_series(quantile, interpolation.0)
            .map_err(PyPolarsErr::from)?;
        let out = tmp.get(0).unwrap_or(AnyValue::Null);

        Ok(Python::with_gil(|py| Wrap(out).into_py(py)))
    }

    fn sum(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .sum_as_series()
                .map_err(PyPolarsErr::from)?
                .get(0)
                .map_err(PyPolarsErr::from)?,
        )
        .into_py(py))
    }
}
