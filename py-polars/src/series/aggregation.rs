use pyo3::prelude::*;

use crate::error::PyPolarsErr;
use crate::prelude::*;
use crate::PySeries;

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
                .max_as_series()
                .map_err(PyPolarsErr::from)?
                .get(0)
                .map_err(PyPolarsErr::from)?,
        )
        .into_py(py))
    }

    fn mean(&self, py: Python) -> PyResult<PyObject> {
        match self.series.dtype() {
            DataType::Boolean => Ok(Wrap(
                self.series
                    .cast(&DataType::UInt8)
                    .unwrap()
                    .mean_as_series()
                    .get(0)
                    .map_err(PyPolarsErr::from)?,
            )
            .into_py(py)),
            DataType::Date | DataType::Datetime(_, _) | DataType::Duration(_) | DataType::Time => {
                Ok(Wrap(
                    self.series
                        .mean_as_series()
                        .get(0)
                        .map_err(PyPolarsErr::from)?,
                )
                .into_py(py))
            },
            _ => Ok(self.series.mean().into_py(py)),
        }
    }

    fn median(&self, py: Python) -> PyResult<PyObject> {
        match self.series.dtype() {
            DataType::Boolean => Ok(Wrap(
                self.series
                    .cast(&DataType::UInt8)
                    .map_err(PyPolarsErr::from)?
                    .median_as_series()
                    .map_err(PyPolarsErr::from)?
                    .get(0)
                    .map_err(PyPolarsErr::from)?,
            )
            .into_py(py)),
            DataType::Date | DataType::Datetime(_, _) | DataType::Duration(_) | DataType::Time => {
                Ok(Wrap(
                    self.series
                        .median_as_series()
                        .map_err(PyPolarsErr::from)?
                        .get(0)
                        .map_err(PyPolarsErr::from)?,
                )
                .into_py(py))
            },
            _ => Ok(self.series.median().into_py(py)),
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

    fn product(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(self.series.product().get(0).map_err(PyPolarsErr::from)?).into_py(py))
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

    fn std(&self, py: Python, ddof: u8) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .std_as_series(ddof)
                .map_err(PyPolarsErr::from)?
                .get(0)
                .map_err(PyPolarsErr::from)?,
        )
        .into_py(py))
    }

    fn var(&self, py: Python, ddof: u8) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .var_as_series(ddof)
                .map_err(PyPolarsErr::from)?
                .get(0)
                .map_err(PyPolarsErr::from)?,
        )
        .into_py(py))
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
