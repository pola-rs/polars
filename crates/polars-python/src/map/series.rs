use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyBool, PyCFunction, PyFloat, PyList, PyString, PyTuple};

use super::*;
use crate::py_modules::{pl_series, polars};

/// Find the output type and dispatch to that implementation.
fn infer_and_finish<'a, A: ApplyLambda<'a>>(
    applyer: &'a A,
    py: Python<'a>,
    lambda: &'a Bound<'a, PyAny>,
    out: &Bound<'a, PyAny>,
    null_count: usize,
) -> PyResult<PySeries> {
    if out.is_instance_of::<PyBool>() {
        let first_value = out.extract::<bool>().unwrap();
        applyer
            .apply_lambda_with_bool_out_type(py, lambda, null_count, Some(first_value))
            .map(|ca| ca.into_series().into())
    } else if out.is_instance_of::<PyFloat>() {
        let first_value = out.extract::<f64>().unwrap();
        applyer
            .apply_lambda_with_primitive_out_type::<Float64Type>(
                py,
                lambda,
                null_count,
                Some(first_value),
            )
            .map(|ca| ca.into_series().into())
    } else if out.is_instance_of::<PyString>() {
        let first_value = out.extract::<PyBackedStr>().unwrap();
        applyer
            .apply_lambda_with_string_out_type(py, lambda, null_count, Some(first_value))
            .map(|ca| ca.into_series().into())
    } else if out.hasattr("_s")? {
        let py_pyseries = out.getattr("_s").unwrap();
        let series = py_pyseries.extract::<PySeries>().unwrap().series;
        let dt = series.dtype();
        applyer
            .apply_lambda_with_list_out_type(
                py,
                lambda.to_owned().unbind(),
                null_count,
                Some(&series),
                dt,
            )
            .map(|ca| ca.into_series().into())
    } else if out.is_instance_of::<PyList>() || out.is_instance_of::<PyTuple>() {
        let series = pl_series(py).call1(py, (out,))?;
        let py_pyseries = series.getattr(py, "_s").unwrap();
        let series = py_pyseries.extract::<PySeries>(py).unwrap().series;

        let dt = series.dtype();

        // Null dtype may be incorrect, fall back to AnyValues logic.
        if dt.is_nested_null() {
            let av = out.extract::<Wrap<AnyValue>>()?;
            return applyer
                .apply_extract_any_values(py, lambda, null_count, av.0)
                .map(|s| s.into());
        }

        // make a new python function that is:
        // def new_lambda(lambda: Callable):
        //     pl.Series(lambda(value))
        let lambda_owned = lambda.to_owned().unbind();
        let new_lambda = PyCFunction::new_closure(py, None, None, move |args, _kwargs| {
            Python::with_gil(|py| {
                let out = lambda_owned.call1(py, args)?;
                // check if Series, if not, call series constructor on it
                pl_series(py).call1(py, (out,))
            })
        })?
        .into_any()
        .unbind();

        let result = applyer
            .apply_lambda_with_list_out_type(py, new_lambda, null_count, Some(&series), dt)
            .map(|ca| ca.into_series().into());
        match result {
            Ok(out) => Ok(out),
            // Try AnyValue
            Err(_) => {
                let av = out.extract::<Wrap<AnyValue>>()?;
                applyer
                    .apply_extract_any_values(py, lambda, null_count, av.0)
                    .map(|s| s.into())
            },
        }
    } else if out.is_instance_of::<PyDict>() {
        let first = out.extract::<Wrap<AnyValue<'_>>>()?;
        applyer.apply_into_struct(py, lambda, null_count, first.0)
    }
    // this succeeds for numpy ints as well, where checking if it is pyint fails
    // we do this later in the chain so that we don't extract integers from string chars.
    else if out.extract::<i64>().is_ok() {
        let first_value = out.extract::<i64>().unwrap();
        applyer
            .apply_lambda_with_primitive_out_type::<Int64Type>(
                py,
                lambda,
                null_count,
                Some(first_value),
            )
            .map(|ca| ca.into_series().into())
    } else if let Ok(av) = out.extract::<Wrap<AnyValue>>() {
        applyer
            .apply_extract_any_values(py, lambda, null_count, av.0)
            .map(|s| s.into())
    } else {
        #[cfg(feature = "object")]
        {
            applyer
                .apply_lambda_with_object_out_type(
                    py,
                    lambda,
                    null_count,
                    Some(out.to_owned().unbind().into()),
                )
                .map(|ca| ca.into_series().into())
        }
        #[cfg(not(feature = "object"))]
        {
            todo!()
        }
    }
}

pub trait ApplyLambda<'a> {
    fn apply_lambda_unknown(
        &'a self,
        _py: Python<'a>,
        _lambda: &'a Bound<'a, PyAny>,
    ) -> PyResult<PySeries>;

    // Used to store a struct type
    fn apply_into_struct(
        &'a self,
        py: Python<'a>,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<PySeries>;

    /// Apply a lambda with a primitive output type
    fn apply_lambda_with_primitive_out_type<D>(
        &'a self,
        py: Python<'a>,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyPolarsNumericType,
        D::Native: IntoPyObject<'a> + FromPyObject<'a>;

    /// Apply a lambda with a boolean output type
    fn apply_lambda_with_bool_out_type(
        &'a self,
        py: Python<'a>,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<bool>,
    ) -> PyResult<ChunkedArray<BooleanType>>;

    /// Apply a lambda with string output type
    fn apply_lambda_with_string_out_type(
        &'a self,
        py: Python<'a>,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<PyBackedStr>,
    ) -> PyResult<StringChunked>;

    /// Apply a lambda with list output type
    fn apply_lambda_with_list_out_type(
        &'a self,
        py: Python<'a>,
        lambda: PyObject,
        init_null_count: usize,
        first_value: Option<&Series>,
        dt: &DataType,
    ) -> PyResult<ListChunked>;

    fn apply_extract_any_values(
        &'a self,
        py: Python<'a>,
        lambda: &'a Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<Series>;

    /// Apply a lambda with list output type
    #[cfg(feature = "object")]
    fn apply_lambda_with_object_out_type(
        &'a self,
        py: Python<'a>,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<ObjectValue>,
    ) -> PyResult<ObjectChunked<ObjectValue>>;
}

pub fn call_lambda<'a, T>(
    py: Python<'a>,
    lambda: &Bound<'a, PyAny>,
    in_val: T,
) -> PyResult<Bound<'a, PyAny>>
where
    T: IntoPyObject<'a>,
{
    let arg = PyTuple::new(py, [in_val])?;
    lambda.call1(arg)
}

pub(crate) fn call_lambda_and_extract<'py, T, S>(
    py: Python<'py>,
    lambda: &Bound<'py, PyAny>,
    in_val: T,
) -> PyResult<Option<S>>
where
    T: IntoPyObject<'py>,
    S: FromPyObject<'py>,
{
    let out = call_lambda(py, lambda, in_val)?;
    if out.is_none() {
        Ok(None)
    } else {
        out.extract::<S>().map(Some)
    }
}

fn call_lambda_series_out<'py, T>(
    py: Python<'py>,
    lambda: &Bound<PyAny>,
    in_val: T,
) -> PyResult<Series>
where
    T: IntoPyObject<'py>,
{
    let arg = PyTuple::new(py, [in_val])?;
    let out = lambda.call1(arg)?;
    let py_series = out.getattr("_s")?;
    py_series.extract::<PySeries>().map(|s| s.series)
}

fn extract_anyvalues<'a, T, I>(
    py: Python<'a>,
    lambda: &'a Bound<PyAny>,
    len: usize,
    init_null_count: usize,
    iter: I,
    first_value: AnyValue<'a>,
) -> PyResult<Vec<AnyValue<'a>>>
where
    T: IntoPyObject<'a>,
    I: Iterator<Item = Option<T>> + 'a,
{
    let mut avs = Vec::with_capacity(len);
    avs.extend(std::iter::repeat(AnyValue::Null).take(init_null_count));
    avs.push(first_value);

    for opt_val in iter {
        let av = match opt_val {
            None => AnyValue::Null,
            Some(val) => {
                let val: Option<Wrap<AnyValue>> = call_lambda_and_extract(py, lambda, val)?;
                match val {
                    None => AnyValue::Null,
                    Some(av) => av.0,
                }
            },
        };
        avs.push(av)
    }
    Ok(avs)
}

impl<'a> ApplyLambda<'a> for BooleanChunked {
    fn apply_lambda_unknown(&'a self, py: Python, lambda: &Bound<'a, PyAny>) -> PyResult<PySeries> {
        let mut null_count = 0;
        for opt_v in self.into_iter() {
            if let Some(v) = opt_v {
                let arg = PyTuple::new(py, [v])?;
                let out = lambda.call1(arg)?;
                if out.is_none() {
                    null_count += 1;
                    continue;
                }
                return infer_and_finish(self, py, lambda, &out, null_count);
            } else {
                null_count += 1
            }
        }
        Ok(Self::full_null(self.name().clone(), self.len())
            .into_series()
            .into())
    }

    fn apply_into_struct(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<PySeries> {
        let skip = 1;
        let it = self
            .into_iter()
            .skip(init_null_count + skip)
            .map(|opt_val| opt_val.map(|val| call_lambda(py, lambda, val)).transpose());
        iterator_to_struct(
            py,
            it,
            init_null_count,
            first_value,
            self.name().clone(),
            self.len(),
        )
    }

    fn apply_lambda_with_primitive_out_type<D>(
        &'a self,
        py: Python<'a>,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyPolarsNumericType,
        D::Native: IntoPyObject<'a> + FromPyObject<'a>,
    {
        let skip = usize::from(first_value.is_some());
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| call_lambda_and_extract(py, lambda, val).transpose())
                        .transpose()
                });
            iterator_to_primitive(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_lambda_with_bool_out_type(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<bool>,
    ) -> PyResult<BooleanChunked> {
        let skip = usize::from(first_value.is_some());
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| call_lambda_and_extract(py, lambda, val).transpose())
                        .transpose()
                });
            iterator_to_bool(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_lambda_with_string_out_type(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<PyBackedStr>,
    ) -> PyResult<StringChunked> {
        let skip = usize::from(first_value.is_some());
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| call_lambda_and_extract(py, lambda, val).transpose())
                        .transpose()
                });
            iterator_to_string(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_lambda_with_list_out_type(
        &'a self,
        py: Python,
        lambda: PyObject,
        init_null_count: usize,
        first_value: Option<&Series>,
        dt: &DataType,
    ) -> PyResult<ListChunked> {
        let skip = usize::from(first_value.is_some());
        let lambda = lambda.bind(py);
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .map(|val| call_lambda_series_out(py, lambda, val))
                        .transpose()
                });
            iterator_to_list(
                dt,
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_extract_any_values(
        &'a self,
        py: Python,
        lambda: &'a Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<Series> {
        let iter = self.into_iter().skip(init_null_count + 1);
        let avs = extract_anyvalues(py, lambda, self.len(), init_null_count, iter, first_value)?;

        Ok(Series::new(self.name().clone(), &avs))
    }

    #[cfg(feature = "object")]
    fn apply_lambda_with_object_out_type(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<ObjectValue>,
    ) -> PyResult<ObjectChunked<ObjectValue>> {
        let skip = usize::from(first_value.is_some());
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| call_lambda_and_extract(py, lambda, val).transpose())
                        .transpose()
                });
            iterator_to_object(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }
}

impl<'a, T> ApplyLambda<'a> for ChunkedArray<T>
where
    T: PyPolarsNumericType,
    T::Native: IntoPyObject<'a> + FromPyObject<'a>,
    ChunkedArray<T>: IntoSeries,
{
    fn apply_lambda_unknown(
        &'a self,
        py: Python<'a>,
        lambda: &'a Bound<'a, PyAny>,
    ) -> PyResult<PySeries> {
        let mut null_count = 0;
        for opt_v in self.into_iter() {
            if let Some(v) = opt_v {
                let arg = PyTuple::new(py, [v])?;
                let out = lambda.call1(arg)?;
                if out.is_none() {
                    null_count += 1;
                    continue;
                }
                return infer_and_finish(self, py, lambda, &out, null_count);
            } else {
                null_count += 1
            }
        }
        Ok(Self::full_null(self.name().clone(), self.len())
            .into_series()
            .into())
    }

    fn apply_into_struct(
        &'a self,
        py: Python<'a>,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<PySeries> {
        let skip = 1;

        let it = self
            .into_iter()
            .skip(init_null_count + skip)
            .map(|opt_val| opt_val.map(|val| call_lambda(py, lambda, val)).transpose());
        iterator_to_struct(
            py,
            it,
            init_null_count,
            first_value,
            self.name().clone(),
            self.len(),
        )
    }

    fn apply_lambda_with_primitive_out_type<D>(
        &'a self,
        py: Python<'a>,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyPolarsNumericType,
        D::Native: IntoPyObject<'a> + FromPyObject<'a>,
    {
        let skip = usize::from(first_value.is_some());
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| call_lambda_and_extract(py, lambda, val).transpose())
                        .transpose()
                });
            iterator_to_primitive(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_lambda_with_bool_out_type(
        &'a self,
        py: Python<'a>,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<bool>,
    ) -> PyResult<BooleanChunked> {
        let skip = usize::from(first_value.is_some());
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| call_lambda_and_extract(py, lambda, val).transpose())
                        .transpose()
                });
            iterator_to_bool(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_lambda_with_string_out_type(
        &'a self,
        py: Python<'a>,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<PyBackedStr>,
    ) -> PyResult<StringChunked> {
        let skip = usize::from(first_value.is_some());
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| call_lambda_and_extract(py, lambda, val).transpose())
                        .transpose()
                });
            iterator_to_string(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_lambda_with_list_out_type(
        &'a self,
        py: Python<'a>,
        lambda: PyObject,
        init_null_count: usize,
        first_value: Option<&Series>,
        dt: &DataType,
    ) -> PyResult<ListChunked> {
        let skip = usize::from(first_value.is_some());
        let lambda = lambda.bind(py);
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .map(|val| call_lambda_series_out(py, lambda, val))
                        .transpose()
                });
            iterator_to_list(
                dt,
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_extract_any_values(
        &'a self,
        py: Python<'a>,
        lambda: &'a Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<Series> {
        let iter = self.into_iter().skip(init_null_count + 1);
        let avs = extract_anyvalues(py, lambda, self.len(), init_null_count, iter, first_value)?;

        Ok(Series::new(self.name().clone(), &avs))
    }

    #[cfg(feature = "object")]
    fn apply_lambda_with_object_out_type(
        &'a self,
        py: Python<'a>,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<ObjectValue>,
    ) -> PyResult<ObjectChunked<ObjectValue>> {
        let skip = usize::from(first_value.is_some());
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| call_lambda_and_extract(py, lambda, val).transpose())
                        .transpose()
                });
            iterator_to_object(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }
}

impl<'a> ApplyLambda<'a> for StringChunked {
    fn apply_lambda_unknown(
        &'a self,
        py: Python,
        lambda: &'a Bound<'a, PyAny>,
    ) -> PyResult<PySeries> {
        let mut null_count = 0;
        for opt_v in self.into_iter() {
            if let Some(v) = opt_v {
                let arg = PyTuple::new(py, [v])?;
                let out = lambda.call1(arg)?;
                if out.is_none() {
                    null_count += 1;
                    continue;
                }
                return infer_and_finish(self, py, lambda, &out, null_count);
            } else {
                null_count += 1
            }
        }
        Ok(Self::full_null(self.name().clone(), self.len())
            .into_series()
            .into())
    }

    fn apply_into_struct(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<PySeries> {
        let skip = 1;

        let it = self
            .into_iter()
            .skip(init_null_count + skip)
            .map(|opt_val| opt_val.map(|val| call_lambda(py, lambda, val)).transpose());
        iterator_to_struct(
            py,
            it,
            init_null_count,
            first_value,
            self.name().clone(),
            self.len(),
        )
    }

    fn apply_lambda_with_primitive_out_type<D>(
        &'a self,
        py: Python<'a>,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyPolarsNumericType,
        D::Native: IntoPyObject<'a> + FromPyObject<'a>,
    {
        let skip = usize::from(first_value.is_some());
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| call_lambda_and_extract(py, lambda, val).transpose())
                        .transpose()
                });
            iterator_to_primitive(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_lambda_with_bool_out_type(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<bool>,
    ) -> PyResult<BooleanChunked> {
        let skip = usize::from(first_value.is_some());
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| call_lambda_and_extract(py, lambda, val).transpose())
                        .transpose()
                });
            iterator_to_bool(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_lambda_with_string_out_type(
        &self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<PyBackedStr>,
    ) -> PyResult<StringChunked> {
        let skip = usize::from(first_value.is_some());
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| call_lambda_and_extract(py, lambda, val).transpose())
                        .transpose()
                });
            iterator_to_string(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }
    fn apply_lambda_with_list_out_type(
        &'a self,
        py: Python,
        lambda: PyObject,
        init_null_count: usize,
        first_value: Option<&Series>,
        dt: &DataType,
    ) -> PyResult<ListChunked> {
        let skip = usize::from(first_value.is_some());
        let lambda = lambda.bind(py);
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .map(|val| call_lambda_series_out(py, lambda, val))
                        .transpose()
                });
            iterator_to_list(
                dt,
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_extract_any_values(
        &'a self,
        py: Python<'a>,
        lambda: &'a Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<Series> {
        let iter = self.into_iter().skip(init_null_count + 1);
        let avs = extract_anyvalues(py, lambda, self.len(), init_null_count, iter, first_value)?;
        Ok(Series::new(self.name().clone(), &avs))
    }

    #[cfg(feature = "object")]
    fn apply_lambda_with_object_out_type(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<ObjectValue>,
    ) -> PyResult<ObjectChunked<ObjectValue>> {
        let skip = usize::from(first_value.is_some());
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| call_lambda_and_extract(py, lambda, val).transpose())
                        .transpose()
                });
            iterator_to_object(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }
}

fn call_series_lambda(
    pypolars: &Bound<PyModule>,
    lambda: &Bound<PyAny>,
    series: Series,
) -> PyResult<Option<Series>> {
    // create a PySeries struct/object for Python
    let pyseries = PySeries::new(series);
    // Wrap this PySeries object in the python side Series wrapper
    let python_series_wrapper = pypolars
        .getattr("wrap_s")
        .unwrap()
        .call1((pyseries,))
        .unwrap();

    // call the lambda en get a python side Series wrapper
    let out = lambda.call1((python_series_wrapper,))?;
    // unpack the wrapper in a PySeries
    let py_pyseries = out
        .getattr("_s")
        .expect("could not get Series attribute '_s'");
    Ok(py_pyseries.extract::<PySeries>().ok().map(|s| s.series))
}

impl<'a> ApplyLambda<'a> for ListChunked {
    fn apply_lambda_unknown(
        &'a self,
        py: Python,
        lambda: &'a Bound<'a, PyAny>,
    ) -> PyResult<PySeries> {
        let pypolars = polars(py).bind(py);
        let mut null_count = 0;
        for opt_v in self.into_iter() {
            if let Some(v) = opt_v {
                // create a PySeries struct/object for Python
                let pyseries = PySeries::new(v);
                // Wrap this PySeries object in the python side Series wrapper
                let python_series_wrapper = pypolars
                    .getattr("wrap_s")
                    .unwrap()
                    .call1((pyseries,))
                    .unwrap();

                let out = lambda.call1((python_series_wrapper,))?;
                if out.is_none() {
                    null_count += 1;
                    continue;
                }
                return infer_and_finish(self, py, lambda, &out, null_count);
            } else {
                null_count += 1
            }
        }
        Ok(Self::full_null(self.name().clone(), self.len())
            .into_series()
            .into())
    }

    fn apply_into_struct(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<PySeries> {
        let skip = 1;
        // get the pypolars module
        let pypolars = polars(py).bind(py);

        let it = self
            .into_iter()
            .skip(init_null_count + skip)
            .map(|opt_val| {
                opt_val
                    .map(|val| {
                        // create a PySeries struct/object for Python
                        let pyseries = PySeries::new(val);
                        // Wrap this PySeries object in the python side Series wrapper
                        let python_series_wrapper = pypolars
                            .getattr("wrap_s")
                            .unwrap()
                            .call1((pyseries,))
                            .unwrap();
                        call_lambda(py, lambda, python_series_wrapper)
                    })
                    .transpose()
            });
        iterator_to_struct(
            py,
            it,
            init_null_count,
            first_value,
            self.name().clone(),
            self.len(),
        )
    }

    fn apply_lambda_with_primitive_out_type<D>(
        &'a self,
        py: Python<'a>,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyPolarsNumericType,
        D::Native: IntoPyObject<'a> + FromPyObject<'a>,
    {
        let skip = usize::from(first_value.is_some());
        let pypolars = polars(py).bind(py);
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| {
                            // create a PySeries struct/object for Python
                            let pyseries = PySeries::new(val);
                            // Wrap this PySeries object in the python side Series wrapper
                            let python_series_wrapper = pypolars
                                .getattr("wrap_s")
                                .unwrap()
                                .call1((pyseries,))
                                .unwrap();
                            call_lambda_and_extract(py, lambda, python_series_wrapper).transpose()
                        })
                        .transpose()
                });
            iterator_to_primitive(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_lambda_with_bool_out_type(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<bool>,
    ) -> PyResult<BooleanChunked> {
        let skip = usize::from(first_value.is_some());
        let pypolars = polars(py).bind(py);
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| {
                            // create a PySeries struct/object for Python
                            let pyseries = PySeries::new(val);
                            // Wrap this PySeries object in the python side Series wrapper
                            let python_series_wrapper = pypolars
                                .getattr("wrap_s")
                                .unwrap()
                                .call1((pyseries,))
                                .unwrap();
                            call_lambda_and_extract(py, lambda, python_series_wrapper).transpose()
                        })
                        .transpose()
                });
            iterator_to_bool(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_lambda_with_string_out_type(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<PyBackedStr>,
    ) -> PyResult<StringChunked> {
        let skip = usize::from(first_value.is_some());
        // get the pypolars module
        let pypolars = polars(py).bind(py);

        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| {
                            // create a PySeries struct/object for Python
                            let pyseries = PySeries::new(val);
                            // Wrap this PySeries object in the python side Series wrapper
                            let python_series_wrapper = pypolars
                                .getattr("wrap_s")
                                .unwrap()
                                .call1((pyseries,))
                                .unwrap();
                            call_lambda_and_extract(py, lambda, python_series_wrapper).transpose()
                        })
                        .transpose()
                });
            iterator_to_string(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }
    fn apply_lambda_with_list_out_type(
        &'a self,
        py: Python,
        lambda: PyObject,
        init_null_count: usize,
        first_value: Option<&Series>,
        dt: &DataType,
    ) -> PyResult<ListChunked> {
        let skip = usize::from(first_value.is_some());
        let pypolars = polars(py).bind(py);
        let lambda = lambda.bind(py);
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .map(|val| call_series_lambda(pypolars, lambda, val))
                        .transpose()
                        .map(|v| v.flatten())
                });
            iterator_to_list(
                dt,
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_extract_any_values(
        &'a self,
        py: Python<'a>,
        lambda: &'a Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<Series> {
        let pypolars = polars(py).bind(py);
        let mut avs = Vec::with_capacity(self.len());
        avs.extend(std::iter::repeat(AnyValue::Null).take(init_null_count));
        avs.push(first_value);

        let call_with_value = |val: Series| {
            // create a PySeries struct/object for Python
            let pyseries = PySeries::new(val);
            // Wrap this PySeries object in the python side Series wrapper
            let python_series_wrapper = pypolars
                .getattr("wrap_s")
                .unwrap()
                .call1((pyseries,))
                .unwrap();
            call_lambda_and_extract::<_, Wrap<AnyValue>>(py, lambda, python_series_wrapper).map(
                |opt_wrap| match opt_wrap {
                    None => AnyValue::Null,
                    Some(w) => w.0,
                },
            )
        };

        for opt_val in self.into_iter().skip(init_null_count + 1) {
            if let Some(s) = opt_val {
                let av = call_with_value(s)?;
                avs.push(av);
            } else {
                avs.push(AnyValue::Null);
            }
        }
        Ok(Series::new(self.name().clone(), &avs))
    }

    #[cfg(feature = "object")]
    fn apply_lambda_with_object_out_type(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<ObjectValue>,
    ) -> PyResult<ObjectChunked<ObjectValue>> {
        let skip = usize::from(first_value.is_some());
        let pypolars = polars(py).bind(py);
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| {
                            // create a PySeries struct/object for Python
                            let pyseries = PySeries::new(val);
                            // Wrap this PySeries object in the python side Series wrapper
                            let python_series_wrapper = pypolars
                                .getattr("wrap_s")
                                .unwrap()
                                .call1((pyseries,))
                                .unwrap();
                            call_lambda_and_extract(py, lambda, python_series_wrapper).transpose()
                        })
                        .transpose()
                });
            iterator_to_object(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }
}

#[cfg(feature = "dtype-array")]
impl<'a> ApplyLambda<'a> for ArrayChunked {
    fn apply_lambda_unknown(
        &'a self,
        py: Python,
        lambda: &'a Bound<'a, PyAny>,
    ) -> PyResult<PySeries> {
        let pypolars = polars(py).bind(py);
        let mut null_count = 0;
        for opt_v in self.into_iter() {
            if let Some(v) = opt_v {
                // create a PySeries struct/object for Python
                let pyseries = PySeries::new(v);
                // Wrap this PySeries object in the python side Series wrapper
                let python_series_wrapper = pypolars
                    .getattr("wrap_s")
                    .unwrap()
                    .call1((pyseries,))
                    .unwrap();

                let out = lambda.call1((python_series_wrapper,))?;
                if out.is_none() {
                    null_count += 1;
                    continue;
                }
                return infer_and_finish(self, py, lambda, &out, null_count);
            } else {
                null_count += 1
            }
        }
        Ok(Self::full_null(self.name().clone(), self.len())
            .into_series()
            .into())
    }

    fn apply_into_struct(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<PySeries> {
        let skip = 1;
        // get the pypolars module
        let pypolars = polars(py).bind(py);

        let it = self
            .into_iter()
            .skip(init_null_count + skip)
            .map(|opt_val| {
                opt_val
                    .map(|val| {
                        // create a PySeries struct/object for Python
                        let pyseries = PySeries::new(val);
                        // Wrap this PySeries object in the python side Series wrapper
                        let python_series_wrapper = pypolars
                            .getattr("wrap_s")
                            .unwrap()
                            .call1((pyseries,))
                            .unwrap();
                        call_lambda(py, lambda, python_series_wrapper)
                    })
                    .transpose()
            });
        iterator_to_struct(
            py,
            it,
            init_null_count,
            first_value,
            self.name().clone(),
            self.len(),
        )
    }

    fn apply_lambda_with_primitive_out_type<D>(
        &'a self,
        py: Python<'a>,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyPolarsNumericType,
        D::Native: IntoPyObject<'a> + FromPyObject<'a>,
    {
        let skip = usize::from(first_value.is_some());
        let pypolars = polars(py).bind(py);
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| {
                            // create a PySeries struct/object for Python
                            let pyseries = PySeries::new(val);
                            // Wrap this PySeries object in the python side Series wrapper
                            let python_series_wrapper = pypolars
                                .getattr("wrap_s")
                                .unwrap()
                                .call1((pyseries,))
                                .unwrap();
                            call_lambda_and_extract(py, lambda, python_series_wrapper).transpose()
                        })
                        .transpose()
                });
            iterator_to_primitive(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_lambda_with_bool_out_type(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<bool>,
    ) -> PyResult<BooleanChunked> {
        let skip = usize::from(first_value.is_some());
        let pypolars = polars(py).bind(py);
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| {
                            // create a PySeries struct/object for Python
                            let pyseries = PySeries::new(val);
                            // Wrap this PySeries object in the python side Series wrapper
                            let python_series_wrapper = pypolars
                                .getattr("wrap_s")
                                .unwrap()
                                .call1((pyseries,))
                                .unwrap();
                            call_lambda_and_extract(py, lambda, python_series_wrapper).transpose()
                        })
                        .transpose()
                });
            iterator_to_bool(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_lambda_with_string_out_type(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<PyBackedStr>,
    ) -> PyResult<StringChunked> {
        let skip = usize::from(first_value.is_some());
        // get the pypolars module
        let pypolars = polars(py).bind(py);

        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| {
                            // create a PySeries struct/object for Python
                            let pyseries = PySeries::new(val);
                            // Wrap this PySeries object in the python side Series wrapper
                            let python_series_wrapper = pypolars
                                .getattr("wrap_s")
                                .unwrap()
                                .call1((pyseries,))
                                .unwrap();
                            call_lambda_and_extract(py, lambda, python_series_wrapper).transpose()
                        })
                        .transpose()
                });
            iterator_to_string(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }
    fn apply_lambda_with_list_out_type(
        &'a self,
        py: Python,
        lambda: PyObject,
        init_null_count: usize,
        first_value: Option<&Series>,
        dt: &DataType,
    ) -> PyResult<ListChunked> {
        let skip = usize::from(first_value.is_some());
        let pypolars = polars(py).bind(py);
        let lambda = lambda.bind(py);
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .map(|val| call_series_lambda(pypolars, lambda, val))
                        .transpose()
                        .map(|v| v.flatten())
                });
            iterator_to_list(
                dt,
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_extract_any_values(
        &'a self,
        py: Python<'a>,
        lambda: &'a Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<Series> {
        let pypolars = polars(py).bind(py);
        let mut avs = Vec::with_capacity(self.len());
        avs.extend(std::iter::repeat(AnyValue::Null).take(init_null_count));
        avs.push(first_value);

        let call_with_value = |val: Series| {
            // create a PySeries struct/object for Python
            let pyseries = PySeries::new(val);
            // Wrap this PySeries object in the python side Series wrapper
            let python_series_wrapper = pypolars
                .getattr("wrap_s")
                .unwrap()
                .call1((pyseries,))
                .unwrap();
            call_lambda_and_extract::<_, Wrap<AnyValue>>(py, lambda, python_series_wrapper).map(
                |opt_wrap| match opt_wrap {
                    None => AnyValue::Null,
                    Some(w) => w.0,
                },
            )
        };

        for opt_val in self.into_iter().skip(init_null_count + 1) {
            if let Some(s) = opt_val {
                let av = call_with_value(s)?;
                avs.push(av);
            } else {
                avs.push(AnyValue::Null);
            }
        }

        Ok(Series::new(self.name().clone(), &avs))
    }

    #[cfg(feature = "object")]
    fn apply_lambda_with_object_out_type(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<ObjectValue>,
    ) -> PyResult<ObjectChunked<ObjectValue>> {
        let skip = usize::from(first_value.is_some());
        let pypolars = polars(py).bind(py);
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| {
                            // create a PySeries struct/object for Python
                            let pyseries = PySeries::new(val);
                            // Wrap this PySeries object in the python side Series wrapper
                            let python_series_wrapper = pypolars
                                .getattr("wrap_s")
                                .unwrap()
                                .call1((pyseries,))
                                .unwrap();
                            call_lambda_and_extract(py, lambda, python_series_wrapper).transpose()
                        })
                        .transpose()
                });
            iterator_to_object(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }
}

#[cfg(feature = "object")]
impl<'a> ApplyLambda<'a> for ObjectChunked<ObjectValue> {
    fn apply_lambda_unknown(
        &'a self,
        py: Python,
        lambda: &'a Bound<'a, PyAny>,
    ) -> PyResult<PySeries> {
        let mut null_count = 0;
        for opt_v in self.into_iter() {
            if let Some(v) = opt_v {
                let arg = PyTuple::new(py, [v])?;
                let out = lambda.call1(arg)?;
                if out.is_none() {
                    null_count += 1;
                    continue;
                }
                return infer_and_finish(self, py, lambda, &out, null_count);
            } else {
                null_count += 1
            }
        }
        Ok(Self::full_null(self.name().clone(), self.len())
            .into_series()
            .into())
    }

    fn apply_into_struct(
        &'a self,
        py: Python<'a>,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<PySeries> {
        let skip = 1;
        let it = self
            .into_iter()
            .skip(init_null_count + skip)
            .map(|object_value| lambda.call1((object_value.map(|v| &v.inner),)).map(Some));
        iterator_to_struct(
            py,
            it,
            init_null_count,
            first_value,
            self.name().clone(),
            self.len(),
        )
    }

    fn apply_lambda_with_primitive_out_type<D>(
        &'a self,
        py: Python<'a>,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyPolarsNumericType,
        D::Native: IntoPyObject<'a> + FromPyObject<'a>,
    {
        let skip = usize::from(first_value.is_some());
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| call_lambda_and_extract(py, lambda, val).transpose())
                        .transpose()
                });
            iterator_to_primitive(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_lambda_with_bool_out_type(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<bool>,
    ) -> PyResult<BooleanChunked> {
        let skip = usize::from(first_value.is_some());
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| call_lambda_and_extract(py, lambda, val).transpose())
                        .transpose()
                });
            iterator_to_bool(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_lambda_with_string_out_type(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<PyBackedStr>,
    ) -> PyResult<StringChunked> {
        let skip = usize::from(first_value.is_some());
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| call_lambda_and_extract(py, lambda, val).transpose())
                        .transpose()
                });
            iterator_to_string(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_lambda_with_list_out_type(
        &'a self,
        py: Python,
        lambda: PyObject,
        init_null_count: usize,
        first_value: Option<&Series>,
        dt: &DataType,
    ) -> PyResult<ListChunked> {
        let skip = usize::from(first_value.is_some());
        let lambda = lambda.bind(py);
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .map(|val| call_lambda_series_out(py, lambda, val))
                        .transpose()
                });
            iterator_to_list(
                dt,
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }

    fn apply_extract_any_values(
        &'a self,
        py: Python<'a>,
        lambda: &'a Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<Series> {
        let iter = self.into_iter().skip(init_null_count + 1);
        let avs = extract_anyvalues(py, lambda, self.len(), init_null_count, iter, first_value)?;
        Ok(Series::new(self.name().clone(), &avs))
    }

    #[cfg(feature = "object")]
    fn apply_lambda_with_object_out_type(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<ObjectValue>,
    ) -> PyResult<ObjectChunked<ObjectValue>> {
        let skip = usize::from(first_value.is_some());
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name().clone(), self.len()))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val
                        .and_then(|val| call_lambda_and_extract(py, lambda, val).transpose())
                        .transpose()
                });
            iterator_to_object(
                it,
                init_null_count,
                first_value,
                self.name().clone(),
                self.len(),
            )
        }
    }
}

fn iter_struct(ca: &StructChunked) -> impl Iterator<Item = AnyValue> {
    (0..ca.len()).map(|i| unsafe { ca.get_any_value_unchecked(i) })
}

impl<'a> ApplyLambda<'a> for StructChunked {
    fn apply_lambda_unknown(
        &'a self,
        py: Python,
        lambda: &'a Bound<'a, PyAny>,
    ) -> PyResult<PySeries> {
        let mut null_count = 0;

        for val in iter_struct(self) {
            match val {
                AnyValue::Null => null_count += 1,
                _ => {
                    let out = lambda.call1((Wrap(val),))?;
                    if out.is_none() {
                        null_count += 1;
                        continue;
                    }
                    return infer_and_finish(self, py, lambda, &out, null_count);
                },
            }
        }

        Ok(Self::full_null(self.name().clone(), self.len())
            .into_series()
            .into())
    }

    fn apply_into_struct(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<PySeries> {
        let skip = 1;
        let it = iter_struct(self)
            .skip(init_null_count + skip)
            .map(|val| match val {
                AnyValue::Null => Ok(None),
                _ => lambda.call1((Wrap(val),)).map(Some),
            });
        iterator_to_struct(
            py,
            it,
            init_null_count,
            first_value,
            self.name().clone(),
            self.len(),
        )
    }

    fn apply_lambda_with_primitive_out_type<D>(
        &'a self,
        py: Python<'a>,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyPolarsNumericType,
        D::Native: IntoPyObject<'a> + FromPyObject<'a>,
    {
        let skip = usize::from(first_value.is_some());
        let it = iter_struct(self)
            .skip(init_null_count + skip)
            .map(|val| match val {
                AnyValue::Null => Ok(None),
                _ => call_lambda_and_extract(py, lambda, Wrap(val)),
            });

        iterator_to_primitive(
            it,
            init_null_count,
            first_value,
            self.name().clone(),
            self.len(),
        )
    }

    fn apply_lambda_with_bool_out_type(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<bool>,
    ) -> PyResult<BooleanChunked> {
        let skip = usize::from(first_value.is_some());
        let it = iter_struct(self)
            .skip(init_null_count + skip)
            .map(|val| match val {
                AnyValue::Null => Ok(None),
                _ => call_lambda_and_extract(py, lambda, Wrap(val)),
            });

        iterator_to_bool(
            it,
            init_null_count,
            first_value,
            self.name().clone(),
            self.len(),
        )
    }

    fn apply_lambda_with_string_out_type(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<PyBackedStr>,
    ) -> PyResult<StringChunked> {
        let skip = usize::from(first_value.is_some());
        let it = iter_struct(self)
            .skip(init_null_count + skip)
            .map(|val| match val {
                AnyValue::Null => Ok(None),
                _ => call_lambda_and_extract(py, lambda, Wrap(val)),
            });

        iterator_to_string(
            it,
            init_null_count,
            first_value,
            self.name().clone(),
            self.len(),
        )
    }
    fn apply_lambda_with_list_out_type(
        &'a self,
        py: Python,
        lambda: PyObject,
        init_null_count: usize,
        first_value: Option<&Series>,
        dt: &DataType,
    ) -> PyResult<ListChunked> {
        let skip = usize::from(first_value.is_some());
        let lambda = lambda.bind(py);
        let it = iter_struct(self)
            .skip(init_null_count + skip)
            .map(|val| match val {
                AnyValue::Null => Ok(None),
                _ => call_lambda_series_out(py, lambda, Wrap(val)).map(Some),
            });
        iterator_to_list(
            dt,
            it,
            init_null_count,
            first_value,
            self.name().clone(),
            self.len(),
        )
    }

    fn apply_extract_any_values(
        &'a self,
        py: Python<'a>,
        lambda: &'a Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<Series> {
        let mut avs = Vec::with_capacity(self.len());
        avs.extend(std::iter::repeat(AnyValue::Null).take(init_null_count));
        avs.push(first_value);

        for val in iter_struct(self).skip(init_null_count + 1) {
            let av: Option<Wrap<AnyValue>> = call_lambda_and_extract(py, lambda, Wrap(val))?;
            let out = match av {
                None => AnyValue::Null,
                Some(av) => av.0,
            };
            avs.push(out)
        }

        Ok(Series::new(self.name().clone(), &avs))
    }

    #[cfg(feature = "object")]
    fn apply_lambda_with_object_out_type(
        &'a self,
        py: Python,
        lambda: &Bound<'a, PyAny>,
        init_null_count: usize,
        first_value: Option<ObjectValue>,
    ) -> PyResult<ObjectChunked<ObjectValue>> {
        let skip = usize::from(first_value.is_some());
        let it = iter_struct(self)
            .skip(init_null_count + skip)
            .map(|val| match val {
                AnyValue::Null => Ok(None),
                _ => call_lambda_and_extract(py, lambda, Wrap(val)),
            });

        iterator_to_object(
            it,
            init_null_count,
            first_value,
            self.name().clone(),
            self.len(),
        )
    }
}
