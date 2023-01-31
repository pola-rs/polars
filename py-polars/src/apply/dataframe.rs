use polars::prelude::*;
use polars_core::frame::row::{rows_to_schema_first_non_null, Row};
use polars_core::series::SeriesIter;
use pyo3::conversion::{FromPyObject, IntoPy};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PyString, PyTuple};

use super::*;
use crate::conversion::Wrap;
use crate::error::PyPolarsErr;
use crate::series::PySeries;
use crate::PyDataFrame;

fn get_iters(df: &DataFrame) -> Vec<SeriesIter> {
    df.get_columns().iter().map(|s| s.iter()).collect()
}

fn get_iters_skip(df: &DataFrame, skip: usize) -> Vec<std::iter::Skip<SeriesIter>> {
    df.get_columns()
        .iter()
        .map(|s| s.iter().skip(skip))
        .collect()
}

// the return type is Union[PySeries, PyDataFrame] and a boolean indicating if it is a dataframe or not
pub fn apply_lambda_unknown<'a>(
    df: &'a DataFrame,
    py: Python,
    lambda: &'a PyAny,
    inference_size: usize,
) -> PyResult<(PyObject, bool)> {
    let mut null_count = 0;
    let mut iters = get_iters(df);

    for _ in 0..df.height() {
        let iter = iters.iter_mut().map(|it| Wrap(it.next().unwrap()));
        let arg = (PyTuple::new(py, iter),);
        let out = lambda.call1(arg)?;

        if out.is_none() {
            null_count += 1;
            continue;
        } else if out.is_instance_of::<PyBool>().unwrap() {
            let first_value = out.extract::<bool>().ok();
            return Ok((
                PySeries::new(
                    apply_lambda_with_bool_out_type(df, py, lambda, null_count, first_value)
                        .into_series(),
                )
                .into_py(py),
                false,
            ));
        } else if out.is_instance_of::<PyFloat>().unwrap() {
            let first_value = out.extract::<f64>().ok();

            return Ok((
                PySeries::new(
                    apply_lambda_with_primitive_out_type::<Float64Type>(
                        df,
                        py,
                        lambda,
                        null_count,
                        first_value,
                    )
                    .into_series(),
                )
                .into_py(py),
                false,
            ));
        } else if out.is_instance_of::<PyInt>().unwrap() {
            let first_value = out.extract::<i64>().ok();
            return Ok((
                PySeries::new(
                    apply_lambda_with_primitive_out_type::<Int64Type>(
                        df,
                        py,
                        lambda,
                        null_count,
                        first_value,
                    )
                    .into_series(),
                )
                .into_py(py),
                false,
            ));
        } else if out.is_instance_of::<PyString>().unwrap() {
            let first_value = out.extract::<&str>().ok();
            return Ok((
                PySeries::new(
                    apply_lambda_with_utf8_out_type(df, py, lambda, null_count, first_value)
                        .into_series(),
                )
                .into_py(py),
                false,
            ));
        } else if out.hasattr("_s")? {
            let py_pyseries = out.getattr("_s").unwrap();
            let series = py_pyseries.extract::<PySeries>().unwrap().series;
            let dt = series.dtype();
            return Ok((
                PySeries::new(
                    apply_lambda_with_list_out_type(df, py, lambda, null_count, Some(&series), dt)?
                        .into_series(),
                )
                .into_py(py),
                false,
            ));
        } else if out.extract::<Wrap<Row<'a>>>().is_ok() {
            let first_value = out.extract::<Wrap<Row<'a>>>().unwrap().0;
            return Ok((
                PyDataFrame::from(
                    apply_lambda_with_rows_output(
                        df,
                        py,
                        lambda,
                        null_count,
                        first_value,
                        inference_size,
                    )
                    .map_err(PyPolarsErr::from)?,
                )
                .into_py(py),
                true,
            ));
        } else if out.is_instance_of::<PyList>().unwrap() {
            return Err(PyPolarsErr::Other(
                "A list output type is invalid. Do you mean to create polars List Series?\
Then return a Series object."
                    .into(),
            )
            .into());
        } else {
            return Err(PyPolarsErr::Other("Could not determine output type".into()).into());
        }
    }
    Err(PyPolarsErr::Other("Could not determine output type".into()).into())
}

fn apply_iter<'a, T>(
    df: &'a DataFrame,
    py: Python<'a>,
    lambda: &'a PyAny,
    init_null_count: usize,
    skip: usize,
) -> impl Iterator<Item = Option<T>> + 'a
where
    T: FromPyObject<'a>,
{
    let mut iters = get_iters_skip(df, init_null_count + skip);
    ((init_null_count + skip)..df.height()).map(move |_| {
        let iter = iters.iter_mut().map(|it| Wrap(it.next().unwrap()));
        let tpl = (PyTuple::new(py, iter),);
        match lambda.call1(tpl) {
            Ok(val) => val.extract::<T>().ok(),
            Err(e) => panic!("python function failed {e}"),
        }
    })
}

/// Apply a lambda with a primitive output type
pub fn apply_lambda_with_primitive_out_type<'a, D>(
    df: &'a DataFrame,
    py: Python<'a>,
    lambda: &'a PyAny,
    init_null_count: usize,
    first_value: Option<D::Native>,
) -> ChunkedArray<D>
where
    D: PyArrowPrimitiveType,
    D::Native: ToPyObject + FromPyObject<'a>,
{
    let skip = usize::from(first_value.is_some());
    if init_null_count == df.height() {
        ChunkedArray::full_null("apply", df.height())
    } else {
        let iter = apply_iter(df, py, lambda, init_null_count, skip);
        iterator_to_primitive(iter, init_null_count, first_value, "apply", df.height())
    }
}

/// Apply a lambda with a boolean output type
pub fn apply_lambda_with_bool_out_type<'a>(
    df: &'a DataFrame,
    py: Python,
    lambda: &'a PyAny,
    init_null_count: usize,
    first_value: Option<bool>,
) -> ChunkedArray<BooleanType> {
    let skip = usize::from(first_value.is_some());
    if init_null_count == df.height() {
        ChunkedArray::full_null("apply", df.height())
    } else {
        let iter = apply_iter(df, py, lambda, init_null_count, skip);
        iterator_to_bool(iter, init_null_count, first_value, "apply", df.height())
    }
}

/// Apply a lambda with utf8 output type
pub fn apply_lambda_with_utf8_out_type<'a>(
    df: &'a DataFrame,
    py: Python,
    lambda: &'a PyAny,
    init_null_count: usize,
    first_value: Option<&str>,
) -> Utf8Chunked {
    let skip = usize::from(first_value.is_some());
    if init_null_count == df.height() {
        ChunkedArray::full_null("apply", df.height())
    } else {
        let iter = apply_iter::<&str>(df, py, lambda, init_null_count, skip);
        iterator_to_utf8(iter, init_null_count, first_value, "apply", df.height())
    }
}

/// Apply a lambda with list output type
pub fn apply_lambda_with_list_out_type<'a>(
    df: &'a DataFrame,
    py: Python,
    lambda: &'a PyAny,
    init_null_count: usize,
    first_value: Option<&Series>,
    dt: &DataType,
) -> PyResult<ListChunked> {
    let skip = usize::from(first_value.is_some());
    if init_null_count == df.height() {
        Ok(ChunkedArray::full_null("apply", df.height()))
    } else {
        let mut iters = get_iters_skip(df, init_null_count + skip);
        let iter = ((init_null_count + skip)..df.height()).map(|_| {
            let iter = iters.iter_mut().map(|it| Wrap(it.next().unwrap()));
            let tpl = (PyTuple::new(py, iter),);
            match lambda.call1(tpl) {
                Ok(val) => match val.getattr("_s") {
                    Ok(val) => val.extract::<PySeries>().ok().map(|ps| ps.series),
                    Err(_) => {
                        if val.is_none() {
                            None
                        } else {
                            panic!("should return a Series, got a {val:?}")
                        }
                    }
                },
                Err(e) => panic!("python function failed {e}"),
            }
        });
        iterator_to_list(dt, iter, init_null_count, first_value, "apply", df.height())
    }
}

pub fn apply_lambda_with_rows_output<'a>(
    df: &'a DataFrame,
    py: Python,
    lambda: &'a PyAny,
    init_null_count: usize,
    first_value: Row<'a>,
    inference_size: usize,
) -> PolarsResult<DataFrame> {
    let width = first_value.0.len();
    let null_row = Row::new(vec![AnyValue::Null; width]);

    let mut row_buf = Row::default();

    let skip = 1;
    let mut iters = get_iters_skip(df, init_null_count + skip);
    let mut row_iter = ((init_null_count + skip)..df.height()).map(|_| {
        let iter = iters.iter_mut().map(|it| Wrap(it.next().unwrap()));
        let tpl = (PyTuple::new(py, iter),);
        match lambda.call1(tpl) {
            Ok(val) => {
                match val.downcast::<PyTuple>().ok() {
                    Some(tuple) => {
                        row_buf.0.clear();
                        for v in tuple {
                            let v = v.extract::<Wrap<AnyValue>>().unwrap().0;
                            row_buf.0.push(v);
                        }
                        let ptr = &row_buf as *const Row;
                        // Safety:
                        // we know that row constructor of polars dataframe does not keep a reference
                        // to the row. Before we mutate the row buf again, the reference is dropped.
                        // we only cannot prove it to the compiler.
                        // we still to this because it save a Vec allocation in a hot loop.
                        unsafe { &*ptr }
                    }
                    None => &null_row,
                }
            }
            Err(e) => panic!("python function failed {e}"),
        }
    });

    // first rows for schema inference
    let mut buf = Vec::with_capacity(inference_size);
    buf.push(first_value);
    buf.extend((&mut row_iter).take(inference_size).cloned());
    let schema = rows_to_schema_first_non_null(&buf, Some(50));

    if init_null_count > 0 {
        // Safety: we know the iterators size
        let iter = unsafe {
            (0..init_null_count)
                .map(|_| &null_row)
                .chain(buf.iter())
                .chain(row_iter)
                .trust_my_length(df.height())
        };
        DataFrame::from_rows_iter_and_schema(iter, &schema)
    } else {
        // Safety: we know the iterators size
        let iter = unsafe { buf.iter().chain(row_iter).trust_my_length(df.height()) };
        DataFrame::from_rows_iter_and_schema(iter, &schema)
    }
}
