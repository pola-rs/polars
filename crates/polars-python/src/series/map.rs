use pyo3::Python;
use pyo3::prelude::*;

use super::PySeries;
use crate::error::PyPolarsErr;
use crate::map::series::ApplyLambdaGeneric;
use crate::prelude::*;
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
            let s = if let Some(return_dtype) = &return_dtype {
                apply_all_polars_dtypes!(
                    series,
                    apply_generic_with_dtype,
                    py,
                    function,
                    return_dtype,
                    skip_nulls
                )
            } else {
                apply_all_polars_dtypes!(series, apply_generic, py, function, skip_nulls)
            };
            s.map(PySeries::from)
        })
    }
}
