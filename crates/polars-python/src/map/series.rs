use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyNone, PyTuple};

use super::*;
use crate::error::PyPolarsErr;
use crate::prelude::ObjectValue;
use crate::{PySeries, Wrap};

pub trait ApplyLambdaGeneric<'py> {
    fn apply_generic(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        skip_nulls: bool,
    ) -> PyResult<Series>;

    fn apply_generic_with_dtype(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        datatype: &DataType,
        skip_nulls: bool,
    ) -> PyResult<Series>;
}

fn call_and_collect_anyvalues<'py, T, I>(
    py: Python<'py>,
    lambda: &Bound<'py, PyAny>,
    len: usize,
    iter: I,
    skip_nulls: bool,
) -> PyResult<Vec<AnyValue<'static>>>
where
    T: IntoPyObject<'py>,
    I: Iterator<Item = Option<T>>,
{
    let mut avs = Vec::with_capacity(len);
    for opt_val in iter {
        let arg = match opt_val {
            None if skip_nulls => {
                avs.push(AnyValue::Null);
                continue;
            },
            None => PyTuple::new(py, [PyNone::get(py)])?,
            Some(val) => PyTuple::new(py, [val])?,
        };
        let out = lambda.call1(arg)?;
        let av: Option<Wrap<AnyValue>> = if out.is_none() {
            Ok(None)
        } else {
            out.extract().map(Some)
        }?;
        avs.push(av.map(|w| w.0).unwrap_or(AnyValue::Null));
    }
    Ok(avs)
}

impl<'py> ApplyLambdaGeneric<'py> for BooleanChunked {
    fn apply_generic(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), self.into_iter(), skip_nulls)?;
        Ok(Series::from_any_values(self.name().clone(), &avs, true).map_err(PyPolarsErr::from)?)
    }

    fn apply_generic_with_dtype(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        datatype: &DataType,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), self.into_iter(), skip_nulls)?;
        Ok(
            Series::from_any_values_and_dtype(self.name().clone(), &avs, datatype, false)
                .map_err(PyPolarsErr::from)?,
        )
    }
}

impl<'py, T> ApplyLambdaGeneric<'py> for ChunkedArray<T>
where
    T: PyPolarsNumericType,
    T::Native: IntoPyObject<'py> + for<'a> FromPyObject<'a, 'py>,
{
    fn apply_generic(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), self.into_iter(), skip_nulls)?;
        Ok(Series::from_any_values(self.name().clone(), &avs, true).map_err(PyPolarsErr::from)?)
    }

    fn apply_generic_with_dtype(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        datatype: &DataType,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), self.into_iter(), skip_nulls)?;
        Ok(
            Series::from_any_values_and_dtype(self.name().clone(), &avs, datatype, false)
                .map_err(PyPolarsErr::from)?,
        )
    }
}

impl<'py> ApplyLambdaGeneric<'py> for StringChunked {
    fn apply_generic(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), self.into_iter(), skip_nulls)?;
        Ok(Series::from_any_values(self.name().clone(), &avs, true).map_err(PyPolarsErr::from)?)
    }

    fn apply_generic_with_dtype(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        datatype: &DataType,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), self.into_iter(), skip_nulls)?;
        Ok(
            Series::from_any_values_and_dtype(self.name().clone(), &avs, datatype, false)
                .map_err(PyPolarsErr::from)?,
        )
    }
}

impl<'py> ApplyLambdaGeneric<'py> for ListChunked {
    fn apply_generic(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let it = self.into_iter().map(|opt_s| opt_s.map(Wrap));
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), it, skip_nulls)?;
        Ok(Series::from_any_values(self.name().clone(), &avs, true).map_err(PyPolarsErr::from)?)
    }

    fn apply_generic_with_dtype(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        datatype: &DataType,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let it = self.into_iter().map(|opt_s| opt_s.map(Wrap));
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), it, skip_nulls)?;
        Ok(
            Series::from_any_values_and_dtype(self.name().clone(), &avs, datatype, false)
                .map_err(PyPolarsErr::from)?,
        )
    }
}

#[cfg(feature = "dtype-array")]
impl<'py> ApplyLambdaGeneric<'py> for ArrayChunked {
    fn apply_generic(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let it = self.into_iter().map(|opt_s| Some(PySeries::new(opt_s?)));
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), it, skip_nulls)?;
        Ok(Series::from_any_values(self.name().clone(), &avs, true).map_err(PyPolarsErr::from)?)
    }

    fn apply_generic_with_dtype(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        datatype: &DataType,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let it = self.into_iter().map(|opt_s| Some(PySeries::new(opt_s?)));
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), it, skip_nulls)?;
        Ok(
            Series::from_any_values_and_dtype(self.name().clone(), &avs, datatype, false)
                .map_err(PyPolarsErr::from)?,
        )
    }
}

#[cfg(feature = "object")]
impl<'py> ApplyLambdaGeneric<'py> for ObjectChunked<ObjectValue> {
    fn apply_generic(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), self.into_iter(), skip_nulls)?;
        Ok(Series::from_any_values(self.name().clone(), &avs, true).map_err(PyPolarsErr::from)?)
    }

    fn apply_generic_with_dtype(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        datatype: &DataType,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), self.into_iter(), skip_nulls)?;
        Ok(
            Series::from_any_values_and_dtype(self.name().clone(), &avs, datatype, false)
                .map_err(PyPolarsErr::from)?,
        )
    }
}

impl<'py> ApplyLambdaGeneric<'py> for StructChunked {
    fn apply_generic(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let it = (0..self.len())
            .map(|i| unsafe { self.get_any_value_unchecked(i).null_to_none().map(Wrap) });
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), it, skip_nulls)?;
        Ok(Series::from_any_values(self.name().clone(), &avs, true).map_err(PyPolarsErr::from)?)
    }

    fn apply_generic_with_dtype(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        datatype: &DataType,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let it = (0..self.len())
            .map(|i| unsafe { self.get_any_value_unchecked(i).null_to_none().map(Wrap) });
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), it, skip_nulls)?;
        Ok(
            Series::from_any_values_and_dtype(self.name().clone(), &avs, datatype, false)
                .map_err(PyPolarsErr::from)?,
        )
    }
}

impl<'py> ApplyLambdaGeneric<'py> for BinaryChunked {
    fn apply_generic(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), self.into_iter(), skip_nulls)?;
        Ok(Series::from_any_values(self.name().clone(), &avs, true).map_err(PyPolarsErr::from)?)
    }

    fn apply_generic_with_dtype(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        datatype: &DataType,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), self.into_iter(), skip_nulls)?;
        Ok(
            Series::from_any_values_and_dtype(self.name().clone(), &avs, datatype, false)
                .map_err(PyPolarsErr::from)?,
        )
    }
}

impl<'py, L, P> ApplyLambdaGeneric<'py> for Logical<L, P>
where
    L: PolarsDataType,
    P: PolarsDataType,
    Logical<L, P>: LogicalType,
{
    fn apply_generic(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let it = (0..self.len())
            .map(|i| unsafe { self.get_any_value_unchecked(i).null_to_none().map(Wrap) });
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), it, skip_nulls)?;
        Ok(Series::from_any_values(self.name().clone(), &avs, true).map_err(PyPolarsErr::from)?)
    }

    fn apply_generic_with_dtype(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        datatype: &DataType,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let it = (0..self.len())
            .map(|i| unsafe { self.get_any_value_unchecked(i).null_to_none().map(Wrap) });
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), it, skip_nulls)?;
        Ok(
            Series::from_any_values_and_dtype(self.name().clone(), &avs, datatype, false)
                .map_err(PyPolarsErr::from)?,
        )
    }
}

impl<'py> ApplyLambdaGeneric<'py> for NullChunked {
    fn apply_generic(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let it = (0..self.len()).map(|_| None::<Wrap<AnyValue<'static>>>);
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), it, skip_nulls)?;
        Ok(Series::from_any_values(self.name().clone(), &avs, true).map_err(PyPolarsErr::from)?)
    }

    fn apply_generic_with_dtype(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        datatype: &DataType,
        skip_nulls: bool,
    ) -> PyResult<Series> {
        let it = (0..self.len()).map(|_| None::<Wrap<AnyValue<'static>>>);
        let avs = call_and_collect_anyvalues(py, lambda, self.len(), it, skip_nulls)?;
        Ok(
            Series::from_any_values_and_dtype(self.name().clone(), &avs, datatype, false)
                .map_err(PyPolarsErr::from)?,
        )
    }
}

impl<'py> ApplyLambdaGeneric<'py> for ExtensionChunked {
    fn apply_generic(
        &self,
        _py: Python<'py>,
        _lambda: &Bound<'py, PyAny>,
        _skip_nulls: bool,
    ) -> PyResult<Series> {
        unreachable!()
    }

    fn apply_generic_with_dtype(
        &self,
        _py: Python<'py>,
        _lambda: &Bound<'py, PyAny>,
        _datatype: &DataType,
        _skip_nulls: bool,
    ) -> PyResult<Series> {
        unreachable!()
    }
}
