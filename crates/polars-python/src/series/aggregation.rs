use DataType::*;
use polars::prelude::*;
use pyo3::prelude::*;

use super::PySeries;
use crate::conversion::Wrap;
use crate::utils::EnterPolarsExt;

fn scalar_to_py(scalar: PyResult<Scalar>, py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    Wrap(scalar?.as_any_value()).into_pyobject(py)
}

impl PySeries {
    /// `arg_min`/`arg_max` of a scalar column from its first rows: the same index for every
    /// length (the first row, or no index), or the last row for every length (the last
    /// of equal values wins, for `arg_max` of most dtypes). `None` where the first three lengths
    /// show neither (and for any other column): ask the materialised Series.
    fn scalar_arg_extreme(&self, f: fn(&Series) -> Option<usize>) -> Option<Option<usize>> {
        let n = match &*self.column.read() {
            Column::Scalar(sc) => sc.len(),
            _ => return None,
        };
        if matches!(self.column.read().dtype(), DataType::Object(_)) {
            return None;
        }
        let at = |k: usize| f(&self.repeat_invariant_series(|n| n.min(k)));
        let (r1, r2, r3) = (at(1), at(2), at(3));
        if n <= 3 {
            return Some(at(n));
        }
        if r1 == r2 && r2 == r3 {
            Some(r1)
        } else if (r1, r2, r3) == (Some(0), Some(1), Some(2)) {
            Some(Some(n - 1))
        } else {
            None
        }
    }

    /// `min`/`max` of the column: of one row of a scalar column, of the values a dictionary
    /// column's rows use.
    fn min_max_reduce(&self, max: bool) -> PolarsResult<Scalar> {
        let compressed = self.compressed();
        match &compressed {
            Some(Column::Dictionary(_)) if max => compressed.unwrap().max_reduce(),
            Some(Column::Dictionary(_)) => compressed.unwrap().min_reduce(),
            _ => {
                let s = self.repeat_invariant_series(|n| n.min(1));
                if max { s.max_reduce() } else { s.min_reduce() }
            },
        }
    }
}

#[pymethods]
impl PySeries {
    fn any(&self, py: Python<'_>, ignore_nulls: bool) -> PyResult<Option<bool>> {
        py.enter_polars(|| {
            let s = self.repeat_invariant_series(|n| n.min(1));
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
            let s = self.repeat_invariant_series(|n| n.min(1));
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
            let s = self.repeat_invariant_series(|n| n.min(1));
            PolarsResult::Ok(if ignore_nulls {
                s.is_full_null()
            } else {
                s.is_empty()
            })
        })
    }

    fn arg_max(&self, py: Python) -> PyResult<Option<usize>> {
        py.enter_polars(|| {
            let s = match self.scalar_arg_extreme(Series::arg_max) {
                Some(answer) => return PolarsResult::Ok(answer),
                None => self.series(),
            };
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
            let s = match self.scalar_arg_extreme(Series::arg_min) {
                Some(answer) => return PolarsResult::Ok(answer),
                None => self.series(),
            };
            polars_ensure!(
                !s.dtype().is_object(),
                InvalidOperation: "`arg_min` operation not supported for dtype `{}`",
                s.dtype()
            );
            PolarsResult::Ok(s.arg_min())
        })
    }

    fn min<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.min_max_reduce(false)), py)
    }

    fn max<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.min_max_reduce(true)), py)
    }

    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let s = self.series();
        match s.dtype() {
            Boolean => scalar_to_py(
                py.enter_polars(|| s.cast(&DataType::UInt8).unwrap().mean_reduce()),
                py,
            ),
            // For non-numeric output types we require mean_reduce.
            dt if dt.is_temporal() => scalar_to_py(py.enter_polars(|| s.mean_reduce()), py),
            _ => Ok(s.mean().into_pyobject(py)?),
        }
    }

    fn median<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // The middle rows of a scalar column are the same two (or one) values whatever its
        // length: as many rows as it has, up to the parity.
        let s = self.repeat_invariant_series(|n| if n % 2 == 0 { n.min(2) } else { n.min(3) });
        match s.dtype() {
            Boolean => scalar_to_py(
                py.enter_polars(|| s.cast(&DataType::UInt8).unwrap().median_reduce()),
                py,
            ),
            // For non-numeric output types we require median_reduce.
            dt if dt.is_temporal() => scalar_to_py(py.enter_polars(|| s.median_reduce()), py),
            _ => Ok(s.median().into_pyobject(py)?),
        }
    }

    fn product<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series().product()), py)
    }

    fn quantile<'py>(
        &self,
        py: Python<'py>,
        quantile: Bound<'py, PyAny>,
        interpolation: Wrap<QuantileMethod>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Accept either a single float or a list of floats
        if let Ok(q_float) = quantile.extract::<f64>() {
            // Single quantile: use quantile_reduce
            scalar_to_py(
                py.enter_polars(|| self.series().quantile_reduce(q_float, interpolation.0)),
                py,
            )
        } else if let Ok(q_list) = quantile.extract::<Vec<f64>>() {
            // Multiple quantiles: use quantiles_reduce
            scalar_to_py(
                py.enter_polars(|| self.series().quantiles_reduce(&q_list, interpolation.0)),
                py,
            )
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "quantile must be a float or a list of floats",
            ))
        }
    }

    fn std<'py>(&self, py: Python<'py>, ddof: u8) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series().std_reduce(ddof)), py)
    }

    fn var<'py>(&self, py: Python<'py>, ddof: u8) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series().var_reduce(ddof)), py)
    }

    fn sum<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series().sum_reduce()), py)
    }

    fn first<'py>(&self, py: Python<'py>, ignore_nulls: bool) -> PyResult<Bound<'py, PyAny>> {
        let s = self.repeat_invariant_series(|n| n.min(1));
        let result = if ignore_nulls {
            py.enter_polars_ok(|| s.first_non_null())
        } else {
            py.enter_polars_ok(|| s.first())
        };
        scalar_to_py(result, py)
    }

    fn last<'py>(&self, py: Python<'py>, ignore_nulls: bool) -> PyResult<Bound<'py, PyAny>> {
        let s = self.repeat_invariant_series(|n| n.min(1));
        let result = if ignore_nulls {
            py.enter_polars_ok(|| s.last_non_null())
        } else {
            py.enter_polars_ok(|| s.last())
        };
        scalar_to_py(result, py)
    }

    #[cfg(feature = "approx_unique")]
    fn approx_n_unique(&self, py: Python) -> PyResult<IdxSize> {
        py.enter_polars(|| self.series().approx_n_unique())
    }

    #[cfg(feature = "bitwise")]
    fn bitwise_and<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series().and_reduce()), py)
    }

    #[cfg(feature = "bitwise")]
    fn bitwise_or<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series().or_reduce()), py)
    }

    #[cfg(feature = "bitwise")]
    fn bitwise_xor<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py.enter_polars(|| self.series().xor_reduce()), py)
    }
}
