use pyo3::Python;
use pyo3::prelude::*;
use pyo3::types::{PyNone, PyTuple};

use super::PySeries;
use crate::error::PyPolarsErr;
use crate::map::series::ApplyLambdaGeneric;
use crate::prelude::*;
#[cfg(feature = "object")]
use crate::series::construction::series_from_objects;
use crate::{apply_all_polars_dtypes, raise_err};

#[pymethods]
impl PySeries {
    #[pyo3(signature = (function, return_dtype, skip_nulls))]
    fn map_elements(
        &self,
        function: &Bound<PyAny>,
        return_dtype: Option<Wrap<DataType>>,
        skip_nulls: bool,
    ) -> PyResult<PySeries> {
        let series = self.series.read().clone(); // Clone so we don't deadlock on re-entrance.
        let series = series.to_storage();

        if skip_nulls && (series.null_count() == series.len()) {
            if let Some(return_dtype) = return_dtype {
                return Ok(
                    Series::full_null(series.name().clone(), series.len(), &return_dtype.0).into(),
                );
            }
            let msg = "The output type of the 'map_elements' function cannot be determined.\n\
            The function was never called because 'skip_nulls=True' and all values are null.\n\
            Consider setting 'skip_nulls=False' or setting the 'return_dtype'.";
            raise_err!(msg, ComputeError)
        }

        let return_dtype = return_dtype.map(|dt| dt.0);

        Python::attach(|py| {
            let s = match &return_dtype {
                #[cfg(feature = "object")]
                Some(DataType::Object(_)) => {
                    // If the return dtype is Object we should not go through AnyValue.
                    call_and_collect_objects(
                        py,
                        series.name().clone(),
                        function,
                        series.len(),
                        series.iter().map(|av| av.null_to_none().map(Wrap)),
                        skip_nulls,
                    )
                },
                Some(return_dtype) => {
                    apply_all_polars_dtypes!(
                        series,
                        apply_generic_with_dtype,
                        py,
                        function,
                        return_dtype,
                        skip_nulls
                    )
                },
                None => apply_all_polars_dtypes!(series, apply_generic, py, function, skip_nulls),
            };
            s.map(PySeries::from)
        })
    }
}

#[cfg(feature = "object")]
fn call_and_collect_objects<'py, T, I>(
    py: Python<'py>,
    name: PlSmallStr,
    lambda: &Bound<'py, PyAny>,
    len: usize,
    iter: I,
    skip_nulls: bool,
) -> PyResult<Series>
where
    T: IntoPyObject<'py>,
    I: Iterator<Item = Option<T>>,
{
    let mut objects = Vec::with_capacity(len);
    for opt_val in iter {
        let arg = match opt_val {
            None if skip_nulls => {
                objects.push(ObjectValue {
                    inner: PyNone::get(py).to_owned().unbind().into_any(),
                });
                continue;
            },
            None => PyTuple::new(py, [PyNone::get(py)])?,
            Some(val) => PyTuple::new(py, [val])?,
        };
        let out = lambda.call1(arg)?;
        objects.push(ObjectValue {
            inner: out.unbind(),
        });
    }
    Ok(series_from_objects(py, name, objects))
}
