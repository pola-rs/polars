use super::*;
use crate::series::PySeries;
use crate::Wrap;
use polars::chunked_array::builder::get_list_builder;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyCFunction, PyFloat, PyInt, PyList, PyString, PyTuple};

/// Find the output type and dispatch to that implementation.
fn infer_and_finish<'a, A: ApplyLambda<'a>>(
    applyer: &'a A,
    py: Python,
    lambda: &'a PyAny,
    out: &'a PyAny,
    null_count: usize,
) -> PyResult<PySeries> {
    if out.is_instance::<PyInt>().unwrap() {
        let first_value = out.extract::<i64>().unwrap();
        applyer
            .apply_lambda_with_primitive_out_type::<Int64Type>(
                py,
                lambda,
                null_count,
                Some(first_value),
            )
            .map(|ca| ca.into_series().into())
    } else if out.is_instance::<PyFloat>().unwrap() {
        let first_value = out.extract::<f64>().unwrap();
        applyer
            .apply_lambda_with_primitive_out_type::<Float64Type>(
                py,
                lambda,
                null_count,
                Some(first_value),
            )
            .map(|ca| ca.into_series().into())
    } else if out.is_instance::<PyBool>().unwrap() {
        let first_value = out.extract::<bool>().unwrap();
        applyer
            .apply_lambda_with_bool_out_type(py, lambda, null_count, Some(first_value))
            .map(|ca| ca.into_series().into())
    } else if out.is_instance::<PyString>().unwrap() {
        let first_value = out.extract::<&str>().unwrap();
        applyer
            .apply_lambda_with_utf8_out_type(py, lambda, null_count, Some(first_value))
            .map(|ca| ca.into_series().into())
    } else if out.hasattr("_s")? {
        let py_pyseries = out.getattr("_s").unwrap();
        let series = py_pyseries.extract::<PySeries>().unwrap().series;
        let dt = series.dtype();
        applyer
            .apply_lambda_with_list_out_type(py, lambda.to_object(py), null_count, &series, dt)
            .map(|ca| ca.into_series().into())
    } else if out.is_instance::<PyList>().unwrap() {
        let pypolars = PyModule::import(py, "polars").unwrap().to_object(py);
        let series = pypolars.getattr(py, "Series").unwrap().call1(py, (out,))?;
        let py_pyseries = series.getattr(py, "_s").unwrap();
        let series = py_pyseries.extract::<PySeries>(py).unwrap().series;
        let dt = series.dtype();

        // make a new python function that is:
        // def new_lambda(lambda: Callable):
        //     pl.Series(lambda(value))
        let lambda_owned = lambda.to_object(py);
        let new_lambda = PyCFunction::new_closure(
            move |args, _kwargs| {
                Python::with_gil(|py| {
                    let out = lambda_owned.call1(py, args)?;
                    pypolars.getattr(py, "Series").unwrap().call1(py, (out,))
                })
            },
            py,
        )?
        .to_object(py);

        applyer
            .apply_lambda_with_list_out_type(py, new_lambda, null_count, &series, dt)
            .map(|ca| ca.into_series().into())
    } else if out.is_instance::<PyTuple>().unwrap() {
        let first = out.extract::<Wrap<AnyValue<'_>>>()?;
        applyer.apply_to_struct(py, lambda, null_count, first.0)
    } else {
        applyer
            .apply_lambda_with_object_out_type(
                py,
                lambda,
                null_count,
                Some(out.to_object(py).into()),
            )
            .map(|ca| ca.into_series().into())
    }
}

pub trait ApplyLambda<'a> {
    fn apply_lambda_unknown(&'a self, _py: Python, _lambda: &'a PyAny) -> PyResult<PySeries>;

    /// Apply a lambda that doesn't change output types
    fn apply_lambda(&'a self, _py: Python, _lambda: &'a PyAny) -> PyResult<PySeries>;

    // Used to store a struct type
    fn apply_to_struct(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<PySeries>;

    /// Apply a lambda with a primitive output type
    fn apply_lambda_with_primitive_out_type<D>(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyArrowPrimitiveType,
        D::Native: ToPyObject + FromPyObject<'a>;

    /// Apply a lambda with a boolean output type
    fn apply_lambda_with_bool_out_type(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<bool>,
    ) -> PyResult<ChunkedArray<BooleanType>>;

    /// Apply a lambda with utf8 output type
    fn apply_lambda_with_utf8_out_type(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<&str>,
    ) -> PyResult<Utf8Chunked>;

    /// Apply a lambda with list output type
    fn apply_lambda_with_list_out_type(
        &'a self,
        py: Python,
        lambda: PyObject,
        init_null_count: usize,
        first_value: &Series,
        dt: &DataType,
    ) -> PyResult<ListChunked>;

    /// Apply a lambda with list output type
    fn apply_lambda_with_object_out_type(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<ObjectValue>,
    ) -> PyResult<ObjectChunked<ObjectValue>>;
}

fn call_lambda<'a, T>(py: Python, lambda: &'a PyAny, in_val: T) -> PyResult<&'a PyAny>
where
    T: ToPyObject,
{
    let arg = PyTuple::new(py, &[in_val]);
    lambda.call1(arg)
}

fn call_lambda_and_extract<'a, T, S>(py: Python, lambda: &'a PyAny, in_val: T) -> PyResult<S>
where
    T: ToPyObject,
    S: FromPyObject<'a>,
{
    match call_lambda(py, lambda, in_val) {
        Ok(out) => out.extract::<S>(),
        Err(e) => panic!("python function failed {}", e),
    }
}

fn call_lambda_series_out<T>(py: Python, lambda: &PyAny, in_val: T) -> PyResult<Series>
where
    T: ToPyObject,
{
    let arg = PyTuple::new(py, &[in_val]);
    let out = lambda.call1(arg)?;
    let py_series = out.getattr("_s")?;
    Ok(py_series.extract::<PySeries>().unwrap().series)
}

impl<'a> ApplyLambda<'a> for BooleanChunked {
    fn apply_lambda_unknown(&'a self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        let mut null_count = 0;
        for opt_v in self.into_iter() {
            if let Some(v) = opt_v {
                let arg = PyTuple::new(py, &[v]);
                let out = lambda.call1(arg)?;
                if out.is_none() {
                    null_count += 1;
                    continue;
                }
                return infer_and_finish(self, py, lambda, out, null_count);
            } else {
                null_count += 1
            }
        }
        Ok(Self::full_null(self.name(), self.len())
            .into_series()
            .into())
    }

    fn apply_lambda(&'a self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        self.apply_lambda_with_bool_out_type(py, lambda, 0, None)
            .map(|ca| PySeries::new(ca.into_series()))
    }

    fn apply_to_struct(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<PySeries> {
        let skip = 1;
        if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda(py, lambda, val).ok());
            iterator_to_struct(it, init_null_count, first_value, self.name(), self.len())
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| opt_val.and_then(|val| call_lambda(py, lambda, val).ok()));
            iterator_to_struct(it, init_null_count, first_value, self.name(), self.len())
        }
    }

    fn apply_lambda_with_primitive_out_type<D>(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyArrowPrimitiveType,
        D::Native: ToPyObject + FromPyObject<'a>,
    {
        let skip = if first_value.is_some() { 1 } else { 0 };
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_and_extract(py, lambda, val).ok());
            Ok(iterator_to_primitive(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_and_extract(py, lambda, val).ok())
                });
            Ok(iterator_to_primitive(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }

    fn apply_lambda_with_bool_out_type(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<bool>,
    ) -> PyResult<BooleanChunked> {
        let skip = if first_value.is_some() { 1 } else { 0 };
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_and_extract(py, lambda, val).ok());
            Ok(iterator_to_bool(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_and_extract(py, lambda, val).ok())
                });
            Ok(iterator_to_bool(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }

    fn apply_lambda_with_utf8_out_type(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<&str>,
    ) -> PyResult<Utf8Chunked> {
        let skip = if first_value.is_some() { 1 } else { 0 };
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_and_extract(py, lambda, val).ok());

            Ok(iterator_to_utf8(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_and_extract(py, lambda, val).ok())
                });
            Ok(iterator_to_utf8(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }

    fn apply_lambda_with_list_out_type(
        &'a self,
        py: Python,
        lambda: PyObject,
        init_null_count: usize,
        first_value: &Series,
        dt: &DataType,
    ) -> PyResult<ListChunked> {
        let skip = 1;
        let lambda = lambda.as_ref(py);
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_series_out(py, lambda, val).ok());

            Ok(iterator_to_list(
                dt,
                it,
                init_null_count,
                Some(first_value),
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_series_out(py, lambda, val).ok())
                });
            Ok(iterator_to_list(
                dt,
                it,
                init_null_count,
                Some(first_value),
                self.name(),
                self.len(),
            ))
        }
    }

    fn apply_lambda_with_object_out_type(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<ObjectValue>,
    ) -> PyResult<ObjectChunked<ObjectValue>> {
        let skip = if first_value.is_some() { 1 } else { 0 };
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_and_extract(py, lambda, val).ok());

            Ok(iterator_to_object(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_and_extract(py, lambda, val).ok())
                });
            Ok(iterator_to_object(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }
}

impl<'a, T> ApplyLambda<'a> for ChunkedArray<T>
where
    T: PyArrowPrimitiveType + PolarsNumericType,
    T::Native: ToPyObject + FromPyObject<'a>,
    ChunkedArray<T>: IntoSeries,
{
    fn apply_lambda_unknown(&'a self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        let mut null_count = 0;
        for opt_v in self.into_iter() {
            if let Some(v) = opt_v {
                let arg = PyTuple::new(py, &[v]);
                let out = lambda.call1(arg)?;
                if out.is_none() {
                    null_count += 1;
                    continue;
                }
                return infer_and_finish(self, py, lambda, out, null_count);
            } else {
                null_count += 1
            }
        }
        Ok(Self::full_null(self.name(), self.len())
            .into_series()
            .into())
    }

    fn apply_lambda(&'a self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        self.apply_lambda_with_primitive_out_type::<T>(py, lambda, 0, None)
            .map(|ca| PySeries::new(ca.into_series()))
    }

    fn apply_to_struct(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<PySeries> {
        let skip = 1;
        if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda(py, lambda, val).ok());
            iterator_to_struct(it, init_null_count, first_value, self.name(), self.len())
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| opt_val.and_then(|val| call_lambda(py, lambda, val).ok()));
            iterator_to_struct(it, init_null_count, first_value, self.name(), self.len())
        }
    }

    fn apply_lambda_with_primitive_out_type<D>(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyArrowPrimitiveType,
        D::Native: ToPyObject + FromPyObject<'a>,
    {
        let skip = if first_value.is_some() { 1 } else { 0 };
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_and_extract(py, lambda, val).ok());
            Ok(iterator_to_primitive(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_and_extract(py, lambda, val).ok())
                });
            Ok(iterator_to_primitive(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }

    fn apply_lambda_with_bool_out_type(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<bool>,
    ) -> PyResult<BooleanChunked> {
        let skip = if first_value.is_some() { 1 } else { 0 };
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_and_extract(py, lambda, val).ok());
            Ok(iterator_to_bool(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_and_extract(py, lambda, val).ok())
                });
            Ok(iterator_to_bool(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }

    fn apply_lambda_with_utf8_out_type(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<&str>,
    ) -> PyResult<Utf8Chunked> {
        let skip = if first_value.is_some() { 1 } else { 0 };
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_and_extract(py, lambda, val).ok());

            Ok(iterator_to_utf8(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_and_extract(py, lambda, val).ok())
                });
            Ok(iterator_to_utf8(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }

    fn apply_lambda_with_list_out_type(
        &'a self,
        py: Python,
        lambda: PyObject,
        init_null_count: usize,
        first_value: &Series,
        dt: &DataType,
    ) -> PyResult<ListChunked> {
        let skip = 1;
        let lambda = lambda.as_ref(py);
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_series_out(py, lambda, val).ok());

            Ok(iterator_to_list(
                dt,
                it,
                init_null_count,
                Some(first_value),
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_series_out(py, lambda, val).ok())
                });
            Ok(iterator_to_list(
                dt,
                it,
                init_null_count,
                Some(first_value),
                self.name(),
                self.len(),
            ))
        }
    }

    fn apply_lambda_with_object_out_type(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<ObjectValue>,
    ) -> PyResult<ObjectChunked<ObjectValue>> {
        let skip = if first_value.is_some() { 1 } else { 0 };
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_and_extract(py, lambda, val).ok());

            Ok(iterator_to_object(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_and_extract(py, lambda, val).ok())
                });
            Ok(iterator_to_object(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }
}

impl<'a> ApplyLambda<'a> for Utf8Chunked {
    fn apply_lambda_unknown(&'a self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        let mut null_count = 0;
        for opt_v in self.into_iter() {
            if let Some(v) = opt_v {
                let arg = PyTuple::new(py, &[v]);
                let out = lambda.call1(arg)?;
                if out.is_none() {
                    null_count += 1;
                    continue;
                }
                return infer_and_finish(self, py, lambda, out, null_count);
            } else {
                null_count += 1
            }
        }
        Ok(Self::full_null(self.name(), self.len())
            .into_series()
            .into())
    }

    fn apply_lambda(&'a self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        let ca = self.apply_lambda_with_utf8_out_type(py, lambda, 0, None)?;
        Ok(ca.into_series().into())
    }

    fn apply_to_struct(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<PySeries> {
        let skip = 1;
        if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda(py, lambda, val).ok());
            iterator_to_struct(it, init_null_count, first_value, self.name(), self.len())
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| opt_val.and_then(|val| call_lambda(py, lambda, val).ok()));
            iterator_to_struct(it, init_null_count, first_value, self.name(), self.len())
        }
    }

    fn apply_lambda_with_primitive_out_type<D>(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyArrowPrimitiveType,
        D::Native: ToPyObject + FromPyObject<'a>,
    {
        let skip = if first_value.is_some() { 1 } else { 0 };
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_and_extract(py, lambda, val).ok());
            Ok(iterator_to_primitive(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_and_extract(py, lambda, val).ok())
                });
            Ok(iterator_to_primitive(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }

    fn apply_lambda_with_bool_out_type(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<bool>,
    ) -> PyResult<BooleanChunked> {
        let skip = if first_value.is_some() { 1 } else { 0 };
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_and_extract(py, lambda, val).ok());
            Ok(iterator_to_bool(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_and_extract(py, lambda, val).ok())
                });
            Ok(iterator_to_bool(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }

    fn apply_lambda_with_utf8_out_type(
        &self,
        py: Python,
        lambda: &PyAny,
        init_null_count: usize,
        first_value: Option<&str>,
    ) -> PyResult<Utf8Chunked> {
        let skip = if first_value.is_some() { 1 } else { 0 };
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_and_extract(py, lambda, val).ok());

            Ok(iterator_to_utf8(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_and_extract(py, lambda, val).ok())
                });
            Ok(iterator_to_utf8(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }
    fn apply_lambda_with_list_out_type(
        &'a self,
        py: Python,
        lambda: PyObject,
        init_null_count: usize,
        first_value: &Series,
        dt: &DataType,
    ) -> PyResult<ListChunked> {
        let skip = 1;
        let lambda = lambda.as_ref(py);
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_series_out(py, lambda, val).ok());

            Ok(iterator_to_list(
                dt,
                it,
                init_null_count,
                Some(first_value),
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_series_out(py, lambda, val).ok())
                });
            Ok(iterator_to_list(
                dt,
                it,
                init_null_count,
                Some(first_value),
                self.name(),
                self.len(),
            ))
        }
    }

    fn apply_lambda_with_object_out_type(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<ObjectValue>,
    ) -> PyResult<ObjectChunked<ObjectValue>> {
        let skip = if first_value.is_some() { 1 } else { 0 };
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_and_extract(py, lambda, val).ok());

            Ok(iterator_to_object(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_and_extract(py, lambda, val).ok())
                });
            Ok(iterator_to_object(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }
}

fn append_series(
    pypolars: &PyModule,
    builder: &mut (impl ListBuilderTrait + ?Sized),
    lambda: &PyAny,
    series: Series,
) -> PyResult<()> {
    // create a PySeries struct/object for Python
    let pyseries = PySeries::new(series);
    // Wrap this PySeries object in the python side Series wrapper
    let python_series_wrapper = pypolars
        .getattr("wrap_s")
        .unwrap()
        .call1((pyseries,))
        .unwrap();
    // call the lambda en get a python side Series wrapper
    let out = lambda.call1((python_series_wrapper,));
    match out {
        Ok(out) => {
            // unpack the wrapper in a PySeries
            let py_pyseries = out
                .getattr("_s")
                .expect("could net get series attribute '_s'");
            let pyseries = py_pyseries.extract::<PySeries>()?;
            builder.append_series(&pyseries.series);
        }
        Err(_) => {
            builder.append_opt_series(None);
        }
    };
    Ok(())
}

fn call_series_lambda(pypolars: &PyModule, lambda: &PyAny, series: Series) -> Option<Series> {
    // create a PySeries struct/object for Python
    let pyseries = PySeries::new(series);
    // Wrap this PySeries object in the python side Series wrapper
    let python_series_wrapper = pypolars
        .getattr("wrap_s")
        .unwrap()
        .call1((pyseries,))
        .unwrap();

    // call the lambda en get a python side Series wrapper
    let out = lambda.call1((python_series_wrapper,));
    match out {
        Ok(out) => {
            // unpack the wrapper in a PySeries
            let py_pyseries = out
                .getattr("_s")
                .expect("could net get series attribute '_s'");
            let pyseries = py_pyseries.extract::<PySeries>().unwrap();
            Some(pyseries.series)
        }
        Err(_) => None,
    }
}

impl<'a> ApplyLambda<'a> for ListChunked {
    fn apply_lambda_unknown(&'a self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        let pypolars = PyModule::import(py, "polars")?;
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
                return infer_and_finish(self, py, lambda, out, null_count);
            } else {
                null_count += 1
            }
        }
        Ok(Self::full_null(self.name(), self.len())
            .into_series()
            .into())
    }

    fn apply_lambda(&'a self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        // get the pypolars module
        let pypolars = PyModule::import(py, "polars")?;

        match self.dtype() {
            DataType::List(dt) => {
                let mut builder = get_list_builder(dt, self.len() * 5, self.len(), self.name());
                if !self.has_validity() {
                    let mut it = self.into_no_null_iter();
                    // use first value to get dtype and replace default builder
                    if let Some(series) = it.next() {
                        let out_series = call_series_lambda(pypolars, lambda, series)
                            .expect("Cannot determine dtype because lambda failed; Make sure that your udf returns a Series");
                        let dt = out_series.dtype();
                        builder = get_list_builder(dt, self.len() * 5, self.len(), self.name());
                        builder.append_opt_series(Some(&out_series));
                    } else {
                        let mut builder = get_list_builder(dt, 0, 1, self.name());
                        let ca = builder.finish();
                        return Ok(PySeries::new(ca.into_series()));
                    }
                    for series in it {
                        append_series(pypolars, &mut *builder, lambda, series)?;
                    }
                } else {
                    let mut it = self.into_iter();
                    let mut nulls = 0;

                    // use first values to get dtype and replace default builders
                    // continue until no null is found
                    for opt_series in &mut it {
                        if let Some(series) = opt_series {
                            let out_series = call_series_lambda(pypolars, lambda, series)
                                .expect("Cannot determine dtype because lambda failed; Make sure that your udf returns a Series");
                            let dt = out_series.dtype();
                            builder = get_list_builder(dt, self.len() * 5, self.len(), self.name());
                            builder.append_opt_series(Some(&out_series));
                            break;
                        } else {
                            nulls += 1;
                        }
                    }
                    for _ in 0..nulls {
                        builder.append_opt_series(None);
                    }
                    for opt_series in it {
                        if let Some(series) = opt_series {
                            append_series(pypolars, &mut *builder, lambda, series)?;
                        } else {
                            builder.append_opt_series(None)
                        }
                    }
                };
                let ca = builder.finish();
                Ok(PySeries::new(ca.into_series()))
            }
            _ => unimplemented!(),
        }
    }

    fn apply_to_struct(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: AnyValue<'a>,
    ) -> PyResult<PySeries> {
        let skip = 1;
        // get the pypolars module
        let pypolars = PyModule::import(py, "polars")?;
        if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| {
                    // create a PySeries struct/object for Python
                    let pyseries = PySeries::new(val);
                    // Wrap this PySeries object in the python side Series wrapper
                    let python_series_wrapper = pypolars
                        .getattr("wrap_s")
                        .unwrap()
                        .call1((pyseries,))
                        .unwrap();
                    call_lambda(py, lambda, python_series_wrapper).ok()
                });
            iterator_to_struct(it, init_null_count, first_value, self.name(), self.len())
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| {
                        // create a PySeries struct/object for Python
                        let pyseries = PySeries::new(val);
                        // Wrap this PySeries object in the python side Series wrapper
                        let python_series_wrapper = pypolars
                            .getattr("wrap_s")
                            .unwrap()
                            .call1((pyseries,))
                            .unwrap();
                        call_lambda(py, lambda, python_series_wrapper).ok()
                    })
                });
            iterator_to_struct(it, init_null_count, first_value, self.name(), self.len())
        }
    }

    fn apply_lambda_with_primitive_out_type<D>(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyArrowPrimitiveType,
        D::Native: ToPyObject + FromPyObject<'a>,
    {
        let skip = if first_value.is_some() { 1 } else { 0 };
        let pypolars = PyModule::import(py, "polars")?;
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| {
                    // create a PySeries struct/object for Python
                    let pyseries = PySeries::new(val);
                    // Wrap this PySeries object in the python side Series wrapper
                    let python_series_wrapper = pypolars
                        .getattr("wrap_s")
                        .unwrap()
                        .call1((pyseries,))
                        .unwrap();
                    call_lambda_and_extract(py, lambda, python_series_wrapper).ok()
                });
            Ok(iterator_to_primitive(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| {
                        // create a PySeries struct/object for Python
                        let pyseries = PySeries::new(val);
                        // Wrap this PySeries object in the python side Series wrapper
                        let python_series_wrapper = pypolars
                            .getattr("wrap_s")
                            .unwrap()
                            .call1((pyseries,))
                            .unwrap();
                        call_lambda_and_extract(py, lambda, python_series_wrapper).ok()
                    })
                });
            Ok(iterator_to_primitive(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }

    fn apply_lambda_with_bool_out_type(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<bool>,
    ) -> PyResult<BooleanChunked> {
        let skip = if first_value.is_some() { 1 } else { 0 };
        let pypolars = PyModule::import(py, "polars")?;
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| {
                    // create a PySeries struct/object for Python
                    let pyseries = PySeries::new(val);
                    // Wrap this PySeries object in the python side Series wrapper
                    let python_series_wrapper = pypolars
                        .getattr("wrap_s")
                        .unwrap()
                        .call1((pyseries,))
                        .unwrap();
                    call_lambda_and_extract(py, lambda, python_series_wrapper).ok()
                });
            Ok(iterator_to_bool(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| {
                        // create a PySeries struct/object for Python
                        let pyseries = PySeries::new(val);
                        // Wrap this PySeries object in the python side Series wrapper
                        let python_series_wrapper = pypolars
                            .getattr("wrap_s")
                            .unwrap()
                            .call1((pyseries,))
                            .unwrap();
                        call_lambda_and_extract(py, lambda, python_series_wrapper).ok()
                    })
                });
            Ok(iterator_to_bool(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }

    fn apply_lambda_with_utf8_out_type(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<&str>,
    ) -> PyResult<Utf8Chunked> {
        let skip = if first_value.is_some() { 1 } else { 0 };
        // get the pypolars module
        let pypolars = PyModule::import(py, "polars")?;

        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| {
                    // create a PySeries struct/object for Python
                    let pyseries = PySeries::new(val);
                    // Wrap this PySeries object in the python side Series wrapper
                    let python_series_wrapper = pypolars
                        .getattr("wrap_s")
                        .unwrap()
                        .call1((pyseries,))
                        .unwrap();
                    call_lambda_and_extract(py, lambda, python_series_wrapper).ok()
                });

            Ok(iterator_to_utf8(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| {
                        // create a PySeries struct/object for Python
                        let pyseries = PySeries::new(val);
                        // Wrap this PySeries object in the python side Series wrapper
                        let python_series_wrapper = pypolars
                            .getattr("wrap_s")
                            .unwrap()
                            .call1((pyseries,))
                            .unwrap();
                        call_lambda_and_extract(py, lambda, python_series_wrapper).ok()
                    })
                });
            Ok(iterator_to_utf8(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }
    fn apply_lambda_with_list_out_type(
        &'a self,
        py: Python,
        lambda: PyObject,
        init_null_count: usize,
        first_value: &Series,
        dt: &DataType,
    ) -> PyResult<ListChunked> {
        let skip = 1;
        let pypolars = PyModule::import(py, "polars")?;
        let lambda = lambda.as_ref(py);
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_series_lambda(pypolars, lambda, val));

            Ok(iterator_to_list(
                dt,
                it,
                init_null_count,
                Some(first_value),
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| opt_val.and_then(|val| call_series_lambda(pypolars, lambda, val)));
            Ok(iterator_to_list(
                dt,
                it,
                init_null_count,
                Some(first_value),
                self.name(),
                self.len(),
            ))
        }
    }
    fn apply_lambda_with_object_out_type(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<ObjectValue>,
    ) -> PyResult<ObjectChunked<ObjectValue>> {
        let skip = if first_value.is_some() { 1 } else { 0 };
        let pypolars = PyModule::import(py, "polars")?;
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| {
                    // create a PySeries struct/object for Python
                    let pyseries = PySeries::new(val);
                    // Wrap this PySeries object in the python side Series wrapper
                    let python_series_wrapper = pypolars
                        .getattr("wrap_s")
                        .unwrap()
                        .call1((pyseries,))
                        .unwrap();
                    call_lambda_and_extract(py, lambda, python_series_wrapper).ok()
                });

            Ok(iterator_to_object(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| {
                        // create a PySeries struct/object for Python
                        let pyseries = PySeries::new(val);
                        // Wrap this PySeries object in the python side Series wrapper
                        let python_series_wrapper = pypolars
                            .getattr("wrap_s")
                            .unwrap()
                            .call1((pyseries,))
                            .unwrap();
                        call_lambda_and_extract(py, lambda, python_series_wrapper).ok()
                    })
                });
            Ok(iterator_to_object(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }
}

impl<'a> ApplyLambda<'a> for ObjectChunked<ObjectValue> {
    fn apply_lambda_unknown(&'a self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        let mut null_count = 0;
        for opt_v in self.into_iter() {
            if let Some(v) = opt_v {
                let arg = PyTuple::new(py, &[v]);
                let out = lambda.call1(arg)?;
                if out.is_none() {
                    null_count += 1;
                    continue;
                }
                return infer_and_finish(self, py, lambda, out, null_count);
            } else {
                null_count += 1
            }
        }
        Ok(Self::full_null(self.name(), self.len())
            .into_series()
            .into())
    }

    fn apply_lambda(&'a self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        self.apply_lambda_with_object_out_type(py, lambda, 0, None)
            .map(|ca| PySeries::new(ca.into_series()))
    }

    fn apply_to_struct(
        &'a self,
        _py: Python,
        _lambda: &'a PyAny,
        _init_null_count: usize,
        _first_value: AnyValue<'a>,
    ) -> PyResult<PySeries> {
        todo!()
    }

    fn apply_lambda_with_primitive_out_type<D>(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyArrowPrimitiveType,
        D::Native: ToPyObject + FromPyObject<'a>,
    {
        let skip = if first_value.is_some() { 1 } else { 0 };
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_and_extract(py, lambda, val).ok());
            Ok(iterator_to_primitive(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_and_extract(py, lambda, val).ok())
                });
            Ok(iterator_to_primitive(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }

    fn apply_lambda_with_bool_out_type(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<bool>,
    ) -> PyResult<BooleanChunked> {
        let skip = if first_value.is_some() { 1 } else { 0 };
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_and_extract(py, lambda, val).ok());
            Ok(iterator_to_bool(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_and_extract(py, lambda, val).ok())
                });
            Ok(iterator_to_bool(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }

    fn apply_lambda_with_utf8_out_type(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<&str>,
    ) -> PyResult<Utf8Chunked> {
        let skip = if first_value.is_some() { 1 } else { 0 };
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_and_extract(py, lambda, val).ok());

            Ok(iterator_to_utf8(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_and_extract(py, lambda, val).ok())
                });
            Ok(iterator_to_utf8(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }

    fn apply_lambda_with_list_out_type(
        &'a self,
        py: Python,
        lambda: PyObject,
        init_null_count: usize,
        first_value: &Series,
        dt: &DataType,
    ) -> PyResult<ListChunked> {
        let skip = 1;
        let lambda = lambda.as_ref(py);
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_series_out(py, lambda, val).ok());

            Ok(iterator_to_list(
                dt,
                it,
                init_null_count,
                Some(first_value),
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_series_out(py, lambda, val).ok())
                });
            Ok(iterator_to_list(
                dt,
                it,
                init_null_count,
                Some(first_value),
                self.name(),
                self.len(),
            ))
        }
    }

    fn apply_lambda_with_object_out_type(
        &'a self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<ObjectValue>,
    ) -> PyResult<ObjectChunked<ObjectValue>> {
        let skip = if first_value.is_some() { 1 } else { 0 };
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if !self.has_validity() {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda_and_extract(py, lambda, val).ok());

            Ok(iterator_to_object(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self
                .into_iter()
                .skip(init_null_count + skip)
                .map(|opt_val| {
                    opt_val.and_then(|val| call_lambda_and_extract(py, lambda, val).ok())
                });
            Ok(iterator_to_object(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }
}
