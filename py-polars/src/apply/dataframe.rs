use super::*;
use crate::conversion::Wrap;
use crate::error::PyPolarsEr;
use crate::series::PySeries;
use crate::PyDataFrame;
use polars::prelude::*;
use polars_core::frame::row::Row;
use polars_core::utils::accumulate_dataframes_vertical;
use pyo3::conversion::{FromPyObject, IntoPy};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PyString, PyTuple};

// the return type is Union[PySeries, PyDataFrame] and a boolean indicating if it is a dataframe or not
pub fn apply_lambda_unknown<'a>(
    df: &'a DataFrame,
    py: Python,
    lambda: &'a PyAny,
    batch_size: usize,
    rechunk: bool,
) -> PyResult<(PyObject, bool)> {
    let columns = df.get_columns();
    let mut null_count = 0;

    for idx in 0..df.height() {
        let iter = columns.iter().map(|s: &Series| Wrap(s.get(idx)));
        let arg = (PyTuple::new(py, iter),);
        let out = lambda.call1(arg)?;

        if out.is_none() {
            null_count += 1;
            continue;
        } else if out.is_instance::<PyBool>().unwrap() {
            let first_value = out.extract::<bool>().ok();
            return Ok((
                PySeries::new(
                    apply_lambda_with_bool_out_type(df, py, lambda, null_count, first_value)
                        .into_series(),
                )
                .into_py(py),
                false,
            ));
        } else if out.is_instance::<PyFloat>().unwrap() {
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
        } else if out.is_instance::<PyInt>().unwrap() {
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
        } else if out.is_instance::<PyString>().unwrap() {
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
                    apply_lambda_with_list_out_type(df, py, lambda, null_count, Some(&series), dt)
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
                        batch_size,
                        rechunk,
                    )
                    .map_err(PyPolarsEr::from)?,
                )
                .into_py(py),
                true,
            ));
        } else if out.is_instance::<PyList>().unwrap() {
            return Err(PyPolarsEr::Other(
                "A list output type is invalid. Do you mean to create polars List Series?\
Then return a Series object."
                    .into(),
            )
            .into());
        } else {
            return Err(PyPolarsEr::Other("Could not determine output type".into()).into());
        }
    }
    Err(PyPolarsEr::Other("Could not determine output type".into()).into())
}

/// Apply a lambda with a primitive output type
pub fn apply_lambda_with_primitive_out_type<'a, D>(
    df: &'a DataFrame,
    py: Python,
    lambda: &'a PyAny,
    init_null_count: usize,
    first_value: Option<D::Native>,
) -> ChunkedArray<D>
where
    D: PyArrowPrimitiveType,
    D::Native: ToPyObject + FromPyObject<'a>,
{
    let columns = df.get_columns();

    let skip = if first_value.is_some() { 1 } else { 0 };
    if init_null_count == df.height() {
        ChunkedArray::full_null("apply", df.height())
    } else {
        let iter = ((init_null_count + skip)..df.height()).map(|idx| {
            let iter = columns.iter().map(|s: &Series| Wrap(s.get(idx)));
            let tpl = (PyTuple::new(py, iter),);
            match lambda.call1(tpl) {
                Ok(val) => val.extract::<D::Native>().ok(),
                Err(e) => panic!("python function failed {}", e),
            }
        });
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
    let columns = df.get_columns();

    let skip = if first_value.is_some() { 1 } else { 0 };
    if init_null_count == df.height() {
        ChunkedArray::full_null("apply", df.height())
    } else {
        let iter = ((init_null_count + skip)..df.height()).map(|idx| {
            let iter = columns.iter().map(|s: &Series| Wrap(s.get(idx)));
            let tpl = (PyTuple::new(py, iter),);
            match lambda.call1(tpl) {
                Ok(val) => val.extract::<bool>().ok(),
                Err(e) => panic!("python function failed {}", e),
            }
        });
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
    let columns = df.get_columns();

    let skip = if first_value.is_some() { 1 } else { 0 };
    if init_null_count == df.height() {
        ChunkedArray::full_null("apply", df.height())
    } else {
        let iter = ((init_null_count + skip)..df.height()).map(|idx| {
            let iter = columns.iter().map(|s: &Series| Wrap(s.get(idx)));
            let tpl = (PyTuple::new(py, iter),);
            match lambda.call1(tpl) {
                Ok(val) => val.extract::<&str>().ok(),
                Err(e) => panic!("python function failed {}", e),
            }
        });
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
) -> ListChunked {
    let columns = df.get_columns();

    let skip = if first_value.is_some() { 1 } else { 0 };
    if init_null_count == df.height() {
        ChunkedArray::full_null("apply", df.height())
    } else {
        let iter = ((init_null_count + skip)..df.height()).map(|idx| {
            let iter = columns.iter().map(|s: &Series| Wrap(s.get(idx)));
            let tpl = (PyTuple::new(py, iter),);
            match lambda.call1(tpl) {
                Ok(val) => match val.getattr("_s") {
                    Ok(val) => val.extract::<PySeries>().ok().map(|ps| ps.series),
                    Err(_) => {
                        if val.is_none() {
                            None
                        } else {
                            panic!("should return a Series, got a {:?}", val)
                        }
                    }
                },
                Err(e) => panic!("python function failed {}", e),
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
    batch_size: usize,
    rechunk: bool,
) -> Result<DataFrame> {
    let columns = df.get_columns();
    let width = first_value.0.len();
    let null_row = Row::new(vec![AnyValue::Null; width]);

    let skip = 1;
    let mut row_iter = ((init_null_count + skip)..df.height()).map(|idx| {
        let iter = columns.iter().map(|s: &Series| Wrap(s.get(idx)));
        let tpl = (PyTuple::new(py, iter),);
        match lambda.call1(tpl) {
            Ok(val) => val
                .extract::<Wrap<Row>>()
                .map(|r| r.0)
                .unwrap_or_else(|_| null_row.clone()),
            Err(e) => panic!("python function failed {}", e),
        }
    });
    let mut buf = Vec::with_capacity(batch_size);
    buf.push(first_value);

    buf.extend((&mut row_iter).take(batch_size));
    let df = DataFrame::from_rows(&buf)?;
    let schema = df.schema();

    let mut dfs = Vec::with_capacity(df.height() / batch_size + 1);
    dfs.push(df);

    loop {
        buf.clear();
        buf.extend((&mut row_iter).take(batch_size));
        if buf.is_empty() {
            break;
        }
        let df = DataFrame::from_rows_and_schema(&buf, &schema)?;
        dfs.push(df);
    }

    let mut df = accumulate_dataframes_vertical(dfs.into_iter())?;
    if rechunk {
        df.rechunk();
    }
    Ok(df)
}
