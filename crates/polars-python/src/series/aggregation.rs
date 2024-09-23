use polars::prelude::*;
use pyo3::prelude::*;
use DataType::*;

use super::PySeries;
use crate::conversion::Wrap;
use crate::error::PyPolarsErr;

#[pymethods]
impl PySeries {
    fn any(&self, ignore_nulls: bool) -> PyResult<Option<bool>> {
        let s = self.series.bool().map_err(PyPolarsErr::from)?;
        Ok(if ignore_nulls {
            Some(s.any())
        } else {
            s.any_kleene()
        })
    }

    fn all(&self, ignore_nulls: bool) -> PyResult<Option<bool>> {
        let s = self.series.bool().map_err(PyPolarsErr::from)?;
        Ok(if ignore_nulls {
            Some(s.all())
        } else {
            s.all_kleene()
        })
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
                .max_reduce()
                .map_err(PyPolarsErr::from)?
                .as_any_value(),
        )
        .into_py(py))
    }

    fn mean(&self, py: Python) -> PyResult<PyObject> {
        match self.series.dtype() {
            Boolean => Ok(Wrap(
                self.series
                    .cast(&DataType::UInt8)
                    .unwrap()
                    .mean_reduce()
                    .as_any_value(),
            )
            .into_py(py)),
            // For non-numeric output types we require mean_reduce.
            dt if dt.is_temporal() => {
                Ok(Wrap(self.series.mean_reduce().as_any_value()).into_py(py))
            },
            _ => Ok(self.series.mean().into_py(py)),
        }
    }

    fn median(&self, py: Python) -> PyResult<PyObject> {
        match self.series.dtype() {
            Boolean => Ok(Wrap(
                self.series
                    .cast(&DataType::UInt8)
                    .unwrap()
                    .median_reduce()
                    .map_err(PyPolarsErr::from)?
                    .as_any_value(),
            )
            .into_py(py)),
            // For non-numeric output types we require median_reduce.
            dt if dt.is_temporal() => Ok(Wrap(
                self.series
                    .median_reduce()
                    .map_err(PyPolarsErr::from)?
                    .as_any_value(),
            )
            .into_py(py)),
            _ => Ok(self.series.median().into_py(py)),
        }
    }

    fn min(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .min_reduce()
                .map_err(PyPolarsErr::from)?
                .as_any_value(),
        )
        .into_py(py))
    }

    fn product(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .product()
                .map_err(PyPolarsErr::from)?
                .as_any_value(),
        )
        .into_py(py))
    }

    fn quantile(
        &self,
        quantile: f64,
        interpolation: Wrap<QuantileInterpolOptions>,
    ) -> PyResult<PyObject> {
        let bind = self.series.quantile_reduce(quantile, interpolation.0);
        let sc = bind.map_err(PyPolarsErr::from)?;

        Ok(Python::with_gil(|py| Wrap(sc.as_any_value()).into_py(py)))
    }

    fn std(&self, py: Python, ddof: u8) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .std_reduce(ddof)
                .map_err(PyPolarsErr::from)?
                .as_any_value(),
        )
        .into_py(py))
    }

    fn var(&self, py: Python, ddof: u8) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .var_reduce(ddof)
                .map_err(PyPolarsErr::from)?
                .as_any_value(),
        )
        .into_py(py))
    }

    fn sum(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .sum_reduce()
                .map_err(PyPolarsErr::from)?
                .as_any_value(),
        )
        .into_py(py))
    }
}
