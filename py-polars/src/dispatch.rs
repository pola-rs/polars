use crate::error::PyPolarsEr;
use crate::series::PySeries;
use polars::chunked_array::builder::get_list_builder;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyTuple};
use pyo3::PyClass;

pub trait PyArrowPrimitiveType: PolarsPrimitiveType {}

impl PyArrowPrimitiveType for UInt8Type {}
impl PyArrowPrimitiveType for UInt16Type {}
impl PyArrowPrimitiveType for UInt32Type {}
impl PyArrowPrimitiveType for UInt64Type {}
impl PyArrowPrimitiveType for Int8Type {}
impl PyArrowPrimitiveType for Int16Type {}
impl PyArrowPrimitiveType for Int32Type {}
impl PyArrowPrimitiveType for Int64Type {}
impl PyArrowPrimitiveType for Float32Type {}
impl PyArrowPrimitiveType for Float64Type {}
impl PyArrowPrimitiveType for Date32Type {}
impl PyArrowPrimitiveType for Date64Type {}
impl PyArrowPrimitiveType for Time64NanosecondType {}
impl PyArrowPrimitiveType for DurationNanosecondType {}
impl PyArrowPrimitiveType for DurationMillisecondType {}
impl PyArrowPrimitiveType for BooleanType {}

pub trait ApplyLambda<'a, 'b> {
    fn apply_lambda_unknown(&'b self, _py: Python, _lambda: &'a PyAny) -> PyResult<PySeries> {
        unimplemented!()
    }

    /// Apply a lambda that doesn't change output types
    fn apply_lambda(&'b self, _py: Python, _lambda: &'a PyAny) -> PyResult<PySeries> {
        unimplemented!()
    }

    /// Apply a lambda with a primitive output type
    fn apply_lambda_with_primitive_out_type<D>(
        &'b self,
        _py: Python,
        _lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyArrowPrimitiveType,
        D::Native: ToPyObject + FromPyObject<'a>,
    {
        unimplemented!()
    }

    /// Apply a lambda with utf8 output type
    fn apply_lambda_with_utf8_out_type(
        &'b self,
        _py: Python,
        _lambda: &'a PyAny,
    ) -> PyResult<Utf8Chunked> {
        unimplemented!()
    }
}

macro_rules! impl_lambda_with_primitive_type {
    ($self:ident, $dtype:ty, $py:ident, $lambda:ident) => {{
        let ca = if $self.null_count() == 0 {
            let mut it = $self.into_no_null_iter();
            let mut builder = PrimitiveChunkedBuilder::<$dtype>::new($self.name(), $self.len());

            while let Some(v) = it.next() {
                let arg = PyTuple::new($py, &[v]);
                let out = $lambda.call1(arg)?;

                match out.extract() {
                    Ok(v) => builder.append_value(v),
                    Err(_) => builder.append_null(),
                }
            }
            builder.finish()
        } else {
            let mut it = $self.into_iter();
            let mut builder = PrimitiveChunkedBuilder::<$dtype>::new($self.name(), $self.len());
            while let Some(opt_v) = it.next() {
                if let Some(v) = opt_v {
                    let arg = PyTuple::new($py, &[v]);
                    let out = $lambda.call1(arg)?;

                    match out.extract() {
                        Ok(v) => builder.append_value(v),
                        Err(_) => builder.append_null(),
                    }
                } else {
                    builder.append_null()
                }
            }
            builder.finish()
        };
        Ok(ca)
    }};
}

macro_rules! impl_lambda_with_utf8_out_type {
    ($self:ident, $py:ident, $lambda:ident) => {{
        let ca = if $self.null_count() == 0 {
            let mut it = $self.into_no_null_iter();
            let mut builder = Utf8ChunkedBuilder::new($self.name(), $self.len());

            while let Some(v) = it.next() {
                let arg = PyTuple::new($py, &[v]);
                let out = $lambda.call1(arg)?;

                match out.extract::<&str>() {
                    Ok(s) => builder.append_value(s),
                    Err(_) => builder.append_null(),
                }
            }
            builder.finish()
        } else {
            let mut it = $self.into_iter();
            let mut builder = Utf8ChunkedBuilder::new($self.name(), $self.len());
            while let Some(opt_v) = it.next() {
                if let Some(v) = opt_v {
                    let arg = PyTuple::new($py, &[v]);
                    let out = $lambda.call1(arg)?;

                    match out.extract::<&str>() {
                        Ok(s) => builder.append_value(s),
                        Err(_) => builder.append_null(),
                    }
                } else {
                    builder.append_null()
                }
            }
            builder.finish()
        };
        Ok(ca)
    }};
}

fn iterator_to_primitive<T>(
    mut it: impl Iterator<Item = Option<T::Native>>,
    init_null_count: usize,
    first_value: Option<T::Native>,
    name: &str,
    capacity: usize,
) -> ChunkedArray<T>
where
    T: PyArrowPrimitiveType,
{
    let mut builder = PrimitiveChunkedBuilder::<T>::new(name, capacity);
    for _ in 0..init_null_count {
        builder.append_null();
    }
    if let Some(val) = first_value {
        builder.append_value(val)
    }
    for opt_val in it {
        match opt_val {
            Some(val) => builder.append_value(val),
            None => builder.append_null(),
        }
    }
    builder.finish()
}

fn call_lambda_with_primitive<'a, T, S>(py: Python, lambda: &'a PyAny, in_val: T) -> PyResult<S>
where
    T: ToPyObject,
    S: FromPyObject<'a>,
{
    let arg = PyTuple::new(py, &[in_val]);
    let out = lambda.call1(arg).expect("lambda failed");
    out.extract::<S>()
}

impl<'a, 'b, T> ApplyLambda<'a, 'b> for ChunkedArray<T>
where
    T: PyArrowPrimitiveType,
    T::Native: ToPyObject + FromPyObject<'a>,
    ChunkedArray<T>: IntoSeries,
    &'b ChunkedArray<T>:
        IntoIterator<Item = Option<T::Native>> + IntoNoNullIterator<Item = T::Native>,
{
    fn apply_lambda_unknown(&'b self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        let mut null_count = 0;
        let mut it = self.into_iter();
        while let Some(opt_v) = it.next() {
            if let Some(v) = opt_v {
                let arg = PyTuple::new(py, &[v]);
                let out = lambda.call1(arg)?;
                if out.is_none() {
                    null_count += 1;
                    continue;
                } else if out.is_instance::<PyInt>().unwrap() {
                    let first_value = out.extract::<i64>().unwrap();
                    return self
                        .apply_lambda_with_primitive_out_type::<Int64Type>(
                            py,
                            lambda,
                            null_count,
                            Some(first_value),
                        )
                        .map(|ca| ca.into_series().into());
                } else if out.is_instance::<PyFloat>().unwrap() {
                    let first_value = out.extract::<f64>().unwrap();
                    return self
                        .apply_lambda_with_primitive_out_type::<Float64Type>(
                            py,
                            lambda,
                            null_count,
                            Some(first_value),
                        )
                        .map(|ca| ca.into_series().into());
                } else if out.is_instance::<PyBool>().unwrap() {
                    let first_value = out.extract::<bool>().unwrap();
                    return self
                        .apply_lambda_with_primitive_out_type::<BooleanType>(
                            py,
                            lambda,
                            null_count,
                            Some(first_value),
                        )
                        .map(|ca| ca.into_series().into());
                } else {
                    return Err(PyPolarsEr::Other("Could not determine output type".into()).into());
                }
            } else {
                null_count += 1
            }
        }
        Ok(Self::full_null(self.name(), self.len())
            .into_series()
            .into())
    }

    fn apply_lambda(&'b self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        self.apply_lambda_with_primitive_out_type::<T>(py, lambda, 0, None)
            .map(|ca| PySeries::new(ca.into_series()))
    }

    fn apply_lambda_with_primitive_out_type<D>(
        &'b self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyArrowPrimitiveType,
        D::Native: ToPyObject + FromPyObject<'a>,
    {
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if self.null_count() == 0 {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count)
                .map(|val| call_lambda_with_primitive(py, lambda, val).ok());
            Ok(iterator_to_primitive(
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        } else {
            let it = self.into_iter().skip(init_null_count).map(|opt_val| {
                opt_val.and_then(|val| call_lambda_with_primitive(py, lambda, val).ok())
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

    fn apply_lambda_with_utf8_out_type(
        &'b self,
        py: Python,
        lambda: &'a PyAny,
    ) -> PyResult<Utf8Chunked> {
        impl_lambda_with_utf8_out_type!(self, py, lambda)
    }
}

impl<'a, 'b> ApplyLambda<'a, 'b> for Utf8Chunked {
    fn apply_lambda(&'b self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        let ca = self.apply_lambda_with_utf8_out_type(py, lambda)?;
        Ok(ca.into_series().into())
    }

    fn apply_lambda_with_primitive_out_type<D>(
        &'b self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyArrowPrimitiveType,
        D::Native: ToPyObject + FromPyObject<'a>,
    {
        impl_lambda_with_primitive_type!(self, D, py, lambda)
    }

    fn apply_lambda_with_utf8_out_type(&self, py: Python, lambda: &PyAny) -> PyResult<Utf8Chunked> {
        impl_lambda_with_utf8_out_type!(self, py, lambda)
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
    let python_series_wrapper = pypolars.call1("wrap_s", (pyseries,))?;
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
    let python_series_wrapper = pypolars.call1("wrap_s", (pyseries,)).unwrap();
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

fn list_lambda_append_primitive<'a, D>(
    pypolars: &PyModule,
    builder: &mut PrimitiveChunkedBuilder<D>,
    lambda: &'a PyAny,
    series: Series,
) -> PyResult<()>
where
    D: PyArrowPrimitiveType,
    D::Native: ToPyObject + FromPyObject<'a>,
{
    // create a PySeries struct/object for Python
    let pyseries = PySeries::new(series);
    // Wrap this PySeries object in the python side Series wrapper
    let python_series_wrapper = pypolars.call1("wrap_s", (pyseries,))?;
    // call the lambda en get a python side Series wrapper
    let out = lambda.call1((python_series_wrapper,))?;
    builder.append_value(out.extract()?);
    Ok(())
}

fn list_lambda_append_utf8<'a>(
    pypolars: &PyModule,
    builder: &mut Utf8ChunkedBuilder,
    lambda: &'a PyAny,
    series: Series,
) -> PyResult<()> {
    // create a PySeries struct/object for Python
    let pyseries = PySeries::new(series);
    // Wrap this PySeries object in the python side Series wrapper
    let python_series_wrapper = pypolars.call1("wrap_s", (pyseries,))?;
    // call the lambda en get a python side Series wrapper
    let out = lambda.call1((python_series_wrapper,))?;
    let s: &str = out.extract()?;
    builder.append_value(s);
    Ok(())
}

impl<'a, 'b> ApplyLambda<'a, 'b> for ListChunked {
    fn apply_lambda(&'b self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        // get the pypolars module
        let pypolars = PyModule::import(py, "pypolars")?;

        match self.dtype() {
            ArrowDataType::List(dt) => {
                let mut builder = get_list_builder(dt, self.len(), self.name());

                let ca = if self.null_count() == 0 {
                    let mut it = self.into_no_null_iter();

                    // use first value to get dtype and replace default builder
                    if let Some(series) = it.next() {
                        let out_series = call_series_lambda(pypolars, lambda, series)
                            .expect("Cannot determine dtype because lambda failed; Make sure that your udf returns a Series");
                        let dt = out_series.dtype();
                        builder = get_list_builder(dt, self.len(), self.name());
                        builder.append_opt_series(Some(&out_series));
                    } else {
                        let mut builder = get_list_builder(dt, 1, self.name());
                        let ca = builder.finish();
                        return Ok(PySeries::new(ca.into_series()));
                    }

                    while let Some(series) = it.next() {
                        append_series(pypolars, &mut *builder, lambda, series)?;
                    }
                    builder.finish()
                } else {
                    let mut it = self.into_iter();
                    let mut nulls = 0;

                    // use first values to get dtype and replace default builders
                    // continue untill no null is found
                    while let Some(opt_series) = it.next() {
                        if let Some(series) = opt_series {
                            let out_series = call_series_lambda(pypolars, lambda, series)
                                .expect("Cannot determine dtype because lambda failed; Make sure that your udf returns a Series");
                            let dt = out_series.dtype();
                            builder = get_list_builder(dt, self.len(), self.name());
                            builder.append_opt_series(Some(&out_series));
                            break;
                        } else {
                            nulls += 1;
                        }
                    }
                    for _ in 0..nulls {
                        builder.append_opt_series(None);
                    }

                    while let Some(opt_series) = it.next() {
                        if let Some(series) = opt_series {
                            append_series(pypolars, &mut *builder, lambda, series)?;
                        } else {
                            builder.append_opt_series(None)
                        }
                    }
                    builder.finish()
                };
                Ok(PySeries::new(ca.into_series()))
            }
            _ => unimplemented!(),
        }
    }

    fn apply_lambda_with_primitive_out_type<D>(
        &'b self,
        py: Python,
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyArrowPrimitiveType,
        D::Native: ToPyObject + FromPyObject<'a>,
    {
        // get the pypolars module
        let pypolars = PyModule::import(py, "pypolars")?;
        let mut builder = PrimitiveChunkedBuilder::<D>::new(self.name(), self.len());

        if self.null_count() == 0 {
            let mut it = self.into_no_null_iter();

            while let Some(series) = it.next() {
                list_lambda_append_primitive(pypolars, &mut builder, lambda, series)?;
            }
        } else {
            let mut it = self.into_iter();
            while let Some(opt_series) = it.next() {
                if let Some(series) = opt_series {
                    list_lambda_append_primitive(pypolars, &mut builder, lambda, series)?;
                } else {
                    builder.append_null()
                }
            }
        };
        let ca = builder.finish();
        Ok(ca)
    }

    fn apply_lambda_with_utf8_out_type(
        &'b self,
        py: Python,
        lambda: &'a PyAny,
    ) -> PyResult<Utf8Chunked> {
        // get the pypolars module
        let pypolars = PyModule::import(py, "pypolars")?;
        let mut builder = Utf8ChunkedBuilder::new(self.name(), self.len());

        if self.null_count() == 0 {
            let mut it = self.into_no_null_iter();

            while let Some(series) = it.next() {
                list_lambda_append_utf8(pypolars, &mut builder, lambda, series)?;
            }
        } else {
            let mut it = self.into_iter();
            while let Some(opt_series) = it.next() {
                if let Some(series) = opt_series {
                    list_lambda_append_utf8(pypolars, &mut builder, lambda, series)?;
                } else {
                    builder.append_null()
                }
            }
        };
        let ca = builder.finish();
        Ok(ca)
    }
}
