use pyo3::prelude::*;
use pyo3::types::PyCFunction;
use pyo3::Python;

use super::PySeries;
use crate::error::PyPolarsErr;
use crate::map::series::{call_lambda_and_extract, ApplyLambda};
use crate::prelude::*;
use crate::py_modules::SERIES;
use crate::{apply_method_all_arrow_series2, raise_err};

#[pymethods]
impl PySeries {
    #[pyo3(signature = (function, return_dtype, skip_nulls))]
    fn map_elements(
        &self,
        function: &Bound<PyAny>,
        return_dtype: Option<Wrap<DataType>>,
        skip_nulls: bool,
    ) -> PyResult<PySeries> {
        let series = &self.series;

        if return_dtype.is_none() {
            polars_warn!(
                MapWithoutReturnDtypeWarning,
                "Calling `map_elements` without specifying `return_dtype` can lead to unpredictable results. \
                Specify `return_dtype` to silence this warning.")
        }

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

        macro_rules! dispatch_apply {
            ($self:expr, $method:ident, $($args:expr),*) => {
                match $self.dtype() {
                    #[cfg(feature = "object")]
                    DataType::Object(_, _) => {
                        let ca = $self.0.unpack::<ObjectType<ObjectValue>>().unwrap();
                        ca.$method($($args),*)
                    },
                    _ => {
                        apply_method_all_arrow_series2!(
                            $self,
                            $method,
                            $($args),*
                        )
                    }

                }
            }

        }

        Python::with_gil(|py| {
            if matches!(
                self.series.dtype(),
                DataType::Datetime(_, _)
                    | DataType::Date
                    | DataType::Duration(_)
                    | DataType::Categorical(_, _)
                    | DataType::Enum(_, _)
                    | DataType::Binary
                    | DataType::Array(_, _)
                    | DataType::Time
            ) || !skip_nulls
            {
                let mut avs = Vec::with_capacity(self.series.len());
                let s = self.series.rechunk();
                let iter = s.iter().map(|av| match (skip_nulls, av) {
                    (true, AnyValue::Null) => AnyValue::Null,
                    (_, av) => {
                        let input = Wrap(av);
                        call_lambda_and_extract::<_, Wrap<AnyValue>>(py, function, input)
                            .unwrap()
                            .0
                    },
                });
                avs.extend(iter);
                return Ok(Series::new(self.series.name().clone(), &avs).into());
            }

            let out = match return_dtype {
                Some(DataType::Int8) => {
                    let ca: Int8Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        function,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::Int16) => {
                    let ca: Int16Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        function,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::Int32) => {
                    let ca: Int32Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        function,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::Int64) => {
                    let ca: Int64Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        function,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::UInt8) => {
                    let ca: UInt8Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        function,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::UInt16) => {
                    let ca: UInt16Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        function,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::UInt32) => {
                    let ca: UInt32Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        function,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::UInt64) => {
                    let ca: UInt64Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        function,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::Float32) => {
                    let ca: Float32Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        function,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::Float64) => {
                    let ca: Float64Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        function,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::Boolean) => {
                    let ca: BooleanChunked = dispatch_apply!(
                        series,
                        apply_lambda_with_bool_out_type,
                        py,
                        function,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                Some(DataType::String) => {
                    let ca = dispatch_apply!(
                        series,
                        apply_lambda_with_string_out_type,
                        py,
                        function,
                        0,
                        None
                    )?;

                    ca.into_series()
                },
                Some(DataType::List(inner)) => {
                    // Make sure the function returns a Series of the correct data type.
                    let function_owned = function.to_object(py);
                    let dtype_py = Wrap((*inner).clone()).to_object(py);
                    let function_wrapped =
                        PyCFunction::new_closure_bound(py, None, None, move |args, _kwargs| {
                            Python::with_gil(|py| {
                                let out = function_owned.call1(py, args)?;
                                SERIES.call1(py, ("", out, dtype_py.clone_ref(py)))
                            })
                        })?
                        .to_object(py);

                    let ca = dispatch_apply!(
                        series,
                        apply_lambda_with_list_out_type,
                        py,
                        function_wrapped,
                        0,
                        None,
                        inner.as_ref()
                    )?;

                    ca.into_series()
                },
                #[cfg(feature = "object")]
                Some(DataType::Object(_, _)) => {
                    let ca = dispatch_apply!(
                        series,
                        apply_lambda_with_object_out_type,
                        py,
                        function,
                        0,
                        None
                    )?;
                    ca.into_series()
                },
                None => return dispatch_apply!(series, apply_lambda_unknown, py, function),

                _ => return dispatch_apply!(series, apply_lambda_unknown, py, function),
            };

            Ok(out.into())
        })
    }
}
