use crate::error::PyPolarsEr;
use crate::series::PySeries;
use polars::chunked_array::builder::get_list_builder;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString, PyTuple};

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

pub trait ApplyLambda<'a> {
    fn apply_lambda_unknown(&'a self, _py: Python, _lambda: &'a PyAny) -> PyResult<PySeries> {
        unimplemented!()
    }

    /// Apply a lambda that doesn't change output types
    fn apply_lambda(&'a self, _py: Python, _lambda: &'a PyAny) -> PyResult<PySeries> {
        unimplemented!()
    }

    /// Apply a lambda with a primitive output type
    fn apply_lambda_with_primitive_out_type<D>(
        &'a self,
        _py: Python,
        _lambda: &'a PyAny,
        _init_null_count: usize,
        _first_value: Option<D::Native>,
    ) -> PyResult<ChunkedArray<D>>
    where
        D: PyArrowPrimitiveType,
        D::Native: ToPyObject + FromPyObject<'a>,
    {
        unimplemented!()
    }

    /// Apply a lambda with a boolean output type
    fn apply_lambda_with_bool_out_type(
        &'a self,
        _py: Python,
        _lambda: &'a PyAny,
        _init_null_count: usize,
        _first_value: Option<bool>,
    ) -> PyResult<ChunkedArray<BooleanType>> {
        unimplemented!()
    }

    /// Apply a lambda with utf8 output type
    fn apply_lambda_with_utf8_out_type(
        &'a self,
        _py: Python,
        _lambda: &'a PyAny,
        _init_null_count: usize,
        _first_value: Option<&str>,
    ) -> PyResult<Utf8Chunked> {
        unimplemented!()
    }

    /// Apply a lambda with list output type
    fn apply_lambda_with_list_out_type(
        &'a self,
        _py: Python,
        _lambda: &'a PyAny,
        _init_null_count: usize,
        _first_value: &Series,
        _dt: &DataType,
    ) -> PyResult<ListChunked> {
        unimplemented!()
    }
}

fn iterator_to_primitive<T>(
    it: impl Iterator<Item = Option<T::Native>>,
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

fn iterator_to_bool(
    it: impl Iterator<Item = Option<bool>>,
    init_null_count: usize,
    first_value: Option<bool>,
    name: &str,
    capacity: usize,
) -> ChunkedArray<BooleanType> {
    let mut builder = BooleanChunkedBuilder::new(name, capacity);
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

fn iterator_to_utf8<'a>(
    it: impl Iterator<Item = Option<&'a str>>,
    init_null_count: usize,
    first_value: Option<&str>,
    name: &str,
    capacity: usize,
) -> Utf8Chunked {
    let mut builder = Utf8ChunkedBuilder::new(name, capacity, capacity * 25);
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

fn iterator_to_list(
    dt: &DataType,
    it: impl Iterator<Item = Option<Series>>,
    init_null_count: usize,
    first_value: &Series,
    name: &str,
    capacity: usize,
) -> ListChunked {
    let mut builder = get_list_builder(dt, capacity * 5, capacity, name);
    for _ in 0..init_null_count {
        builder.append_opt_series(None);
    }
    builder.append_series(first_value);
    for opt_val in it {
        builder.append_opt_series(opt_val.as_ref())
    }
    builder.finish()
}

fn call_lambda<'a, T, S>(py: Python, lambda: &'a PyAny, in_val: T) -> PyResult<S>
where
    T: ToPyObject,
    S: FromPyObject<'a>,
{
    let arg = PyTuple::new(py, &[in_val]);
    let out = lambda.call1(arg).expect("lambda failed");
    out.extract::<S>()
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
                        .apply_lambda_with_bool_out_type(py, lambda, null_count, Some(first_value))
                        .map(|ca| ca.into_series().into());
                } else if out.is_instance::<PyString>().unwrap() {
                    let first_value = out.extract::<&str>().unwrap();
                    return self
                        .apply_lambda_with_utf8_out_type(py, lambda, null_count, Some(first_value))
                        .map(|ca| ca.into_series().into());
                } else if out.hasattr("_s")? {
                    let py_pyseries = out.getattr("_s").unwrap();
                    let series = py_pyseries.extract::<PySeries>().unwrap().series;
                    let dt = series.dtype();
                    return self
                        .apply_lambda_with_list_out_type(py, lambda, null_count, &series, dt)
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

    fn apply_lambda(&'a self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        self.apply_lambda_with_bool_out_type(py, lambda, 0, None)
            .map(|ca| PySeries::new(ca.into_series()))
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
        } else if self.null_count() == 0 {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda(py, lambda, val).ok());
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
                .map(|opt_val| opt_val.and_then(|val| call_lambda(py, lambda, val).ok()));
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
        } else if self.null_count() == 0 {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda(py, lambda, val).ok());
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
                .map(|opt_val| opt_val.and_then(|val| call_lambda(py, lambda, val).ok()));
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
        } else if self.null_count() == 0 {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda(py, lambda, val).ok());

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
                .map(|opt_val| opt_val.and_then(|val| call_lambda(py, lambda, val).ok()));
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
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: &Series,
        dt: &DataType,
    ) -> PyResult<ListChunked> {
        let skip = 1;
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if self.null_count() == 0 {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| {
                    let pyseries: Option<PySeries> = call_lambda(py, lambda, val).ok();
                    pyseries.map(|ps| ps.series)
                });

            Ok(iterator_to_list(
                dt,
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
                        let pyseries: Option<PySeries> = call_lambda(py, lambda, val).ok();
                        pyseries.map(|ps| ps.series)
                    })
                });
            Ok(iterator_to_list(
                dt,
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
    T: PyArrowPrimitiveType,
    T::Native: ToPyObject + FromPyObject<'a>,
    ChunkedArray<T>: IntoSeries,
    &'a ChunkedArray<T>:
        IntoIterator<Item = Option<T::Native>> + IntoNoNullIterator<Item = T::Native>,
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
                        .apply_lambda_with_bool_out_type(py, lambda, null_count, Some(first_value))
                        .map(|ca| ca.into_series().into());
                } else if out.is_instance::<PyString>().unwrap() {
                    let first_value = out.extract::<&str>().unwrap();
                    return self
                        .apply_lambda_with_utf8_out_type(py, lambda, null_count, Some(first_value))
                        .map(|ca| ca.into_series().into());
                } else if out.hasattr("_s")? {
                    let py_pyseries = out.getattr("_s").unwrap();
                    let series = py_pyseries.extract::<PySeries>().unwrap().series;
                    let dt = series.dtype();
                    return self
                        .apply_lambda_with_list_out_type(py, lambda, null_count, &series, dt)
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

    fn apply_lambda(&'a self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        self.apply_lambda_with_primitive_out_type::<T>(py, lambda, 0, None)
            .map(|ca| PySeries::new(ca.into_series()))
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
        } else if self.null_count() == 0 {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda(py, lambda, val).ok());
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
                .map(|opt_val| opt_val.and_then(|val| call_lambda(py, lambda, val).ok()));
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
        } else if self.null_count() == 0 {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda(py, lambda, val).ok());
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
                .map(|opt_val| opt_val.and_then(|val| call_lambda(py, lambda, val).ok()));
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
        } else if self.null_count() == 0 {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda(py, lambda, val).ok());

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
                .map(|opt_val| opt_val.and_then(|val| call_lambda(py, lambda, val).ok()));
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
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: &Series,
        dt: &DataType,
    ) -> PyResult<ListChunked> {
        let skip = 1;
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if self.null_count() == 0 {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| {
                    let pyseries: Option<PySeries> = call_lambda(py, lambda, val).ok();
                    pyseries.map(|ps| ps.series)
                });

            Ok(iterator_to_list(
                dt,
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
                        let pyseries: Option<PySeries> = call_lambda(py, lambda, val).ok();
                        pyseries.map(|ps| ps.series)
                    })
                });
            Ok(iterator_to_list(
                dt,
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
                        .apply_lambda_with_bool_out_type(py, lambda, null_count, Some(first_value))
                        .map(|ca| ca.into_series().into());
                } else if out.is_instance::<PyString>().unwrap() {
                    let first_value = out.extract::<&str>().unwrap();
                    return self
                        .apply_lambda_with_utf8_out_type(py, lambda, null_count, Some(first_value))
                        .map(|ca| ca.into_series().into());
                } else if out.hasattr("_s")? {
                    let py_pyseries = out.getattr("_s").unwrap();
                    let series = py_pyseries.extract::<PySeries>().unwrap().series;
                    let dt = series.dtype();
                    return self
                        .apply_lambda_with_list_out_type(py, lambda, null_count, &series, dt)
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

    fn apply_lambda(&'a self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        let ca = self.apply_lambda_with_utf8_out_type(py, lambda, 0, None)?;
        Ok(ca.into_series().into())
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
        } else if self.null_count() == 0 {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda(py, lambda, val).ok());
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
                .map(|opt_val| opt_val.and_then(|val| call_lambda(py, lambda, val).ok()));
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
        } else if self.null_count() == 0 {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda(py, lambda, val).ok());
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
                .map(|opt_val| opt_val.and_then(|val| call_lambda(py, lambda, val).ok()));
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
        } else if self.null_count() == 0 {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_lambda(py, lambda, val).ok());

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
                .map(|opt_val| opt_val.and_then(|val| call_lambda(py, lambda, val).ok()));
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
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: &Series,
        dt: &DataType,
    ) -> PyResult<ListChunked> {
        let skip = 1;
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if self.null_count() == 0 {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| {
                    let pyseries: Option<PySeries> = call_lambda(py, lambda, val).ok();
                    pyseries.map(|ps| ps.series)
                });

            Ok(iterator_to_list(
                dt,
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
                        let pyseries: Option<PySeries> = call_lambda(py, lambda, val).ok();
                        pyseries.map(|ps| ps.series)
                    })
                });
            Ok(iterator_to_list(
                dt,
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

impl<'a> ApplyLambda<'a> for ListChunked {
    fn apply_lambda_unknown(&'a self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        let pypolars = PyModule::import(py, "polars")?;
        let mut null_count = 0;
        for opt_v in self.into_iter() {
            if let Some(v) = opt_v {
                // create a PySeries struct/object for Python
                let pyseries = PySeries::new(v);
                // Wrap this PySeries object in the python side Series wrapper
                let python_series_wrapper = pypolars.call1("wrap_s", (pyseries,)).unwrap();

                let out = lambda.call1((python_series_wrapper,))?;
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
                        .apply_lambda_with_bool_out_type(py, lambda, null_count, Some(first_value))
                        .map(|ca| ca.into_series().into());
                } else if out.is_instance::<PyString>().unwrap() {
                    let first_value = out.extract::<&str>().unwrap();
                    return self
                        .apply_lambda_with_utf8_out_type(py, lambda, null_count, Some(first_value))
                        .map(|ca| ca.into_series().into());
                } else if out.hasattr("_s")? {
                    let py_pyseries = out.getattr("_s").unwrap();
                    let series = py_pyseries.extract::<PySeries>().unwrap().series;
                    let dt = series.dtype();
                    return self
                        .apply_lambda_with_list_out_type(py, lambda, null_count, &series, dt)
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

    fn apply_lambda(&'a self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        // get the pypolars module
        let pypolars = PyModule::import(py, "polars")?;

        match self.dtype() {
            DataType::List(dt) => {
                let mut builder =
                    get_list_builder(&dt.into(), self.len() * 5, self.len(), self.name());

                let ca = if self.null_count() == 0 {
                    let mut it = self.into_no_null_iter();

                    // use first value to get dtype and replace default builder
                    if let Some(series) = it.next() {
                        let out_series = call_series_lambda(pypolars, lambda, series)
                            .expect("Cannot determine dtype because lambda failed; Make sure that your udf returns a Series");
                        let dt = out_series.dtype();
                        builder = get_list_builder(dt, self.len() * 5, self.len(), self.name());
                        builder.append_opt_series(Some(&out_series));
                    } else {
                        let mut builder = get_list_builder(&dt.into(), 0, 1, self.name());
                        let ca = builder.finish();
                        return Ok(PySeries::new(ca.into_series()));
                    }

                    for series in it {
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
                    builder.finish()
                };
                Ok(PySeries::new(ca.into_series()))
            }
            _ => unimplemented!(),
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
        } else if self.null_count() == 0 {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| {
                    // create a PySeries struct/object for Python
                    let pyseries = PySeries::new(val);
                    // Wrap this PySeries object in the python side Series wrapper
                    let python_series_wrapper = pypolars.call1("wrap_s", (pyseries,)).unwrap();
                    call_lambda(py, lambda, python_series_wrapper).ok()
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
                        let python_series_wrapper = pypolars.call1("wrap_s", (pyseries,)).unwrap();
                        call_lambda(py, lambda, python_series_wrapper).ok()
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
        } else if self.null_count() == 0 {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| {
                    // create a PySeries struct/object for Python
                    let pyseries = PySeries::new(val);
                    // Wrap this PySeries object in the python side Series wrapper
                    let python_series_wrapper = pypolars.call1("wrap_s", (pyseries,)).unwrap();
                    call_lambda(py, lambda, python_series_wrapper).ok()
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
                        let python_series_wrapper = pypolars.call1("wrap_s", (pyseries,)).unwrap();
                        call_lambda(py, lambda, python_series_wrapper).ok()
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
        } else if self.null_count() == 0 {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| {
                    // create a PySeries struct/object for Python
                    let pyseries = PySeries::new(val);
                    // Wrap this PySeries object in the python side Series wrapper
                    let python_series_wrapper = pypolars.call1("wrap_s", (pyseries,)).unwrap();
                    call_lambda(py, lambda, python_series_wrapper).ok()
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
                        let python_series_wrapper = pypolars.call1("wrap_s", (pyseries,)).unwrap();
                        call_lambda(py, lambda, python_series_wrapper).ok()
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
        lambda: &'a PyAny,
        init_null_count: usize,
        first_value: &Series,
        dt: &DataType,
    ) -> PyResult<ListChunked> {
        let skip = 1;
        let pypolars = PyModule::import(py, "polars")?;
        if init_null_count == self.len() {
            Ok(ChunkedArray::full_null(self.name(), self.len()))
        } else if self.null_count() == 0 {
            let it = self
                .into_no_null_iter()
                .skip(init_null_count + skip)
                .map(|val| call_series_lambda(pypolars, lambda, val));

            Ok(iterator_to_list(
                dt,
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
                .map(|opt_val| opt_val.and_then(|val| call_series_lambda(pypolars, lambda, val)));
            Ok(iterator_to_list(
                dt,
                it,
                init_null_count,
                first_value,
                self.name(),
                self.len(),
            ))
        }
    }
}
