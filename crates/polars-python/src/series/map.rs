use pyo3::Python;
use pyo3::prelude::*;
use pyo3::types::PyCFunction;

use super::PySeries;
use crate::error::PyPolarsErr;
use crate::map::check_nested_object;
use crate::map::series::{ApplyLambda, call_lambda_and_extract};
use crate::prelude::*;
use crate::py_modules::pl_series;
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
        let series = &self.series.read().clone(); // Clone so we don't deadlock on re-entrance.

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
                    DataType::Object(_) => {
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

        Python::attach(|py| {
            if matches!(
                series.dtype(),
                DataType::Datetime(_, _)
                    | DataType::Date
                    | DataType::Duration(_)
                    | DataType::Categorical(_, _)
                    | DataType::Enum(_, _)
                    | DataType::Binary
                    | DataType::Array(_, _)
                    | DataType::Time
                    | DataType::Decimal(_, _)
            ) || !skip_nulls
            {
                let mut avs = Vec::with_capacity(series.len());
                let s = series.rechunk();

                for av in s.iter() {
                    let out = match (skip_nulls, av) {
                        (true, AnyValue::Null) => AnyValue::Null,
                        (_, av) => {
                            let av: Option<Wrap<AnyValue>> =
                                call_lambda_and_extract(py, function, Wrap(av))?;
                            match av {
                                None => AnyValue::Null,
                                Some(av) => av.0,
                            }
                        },
                    };
                    avs.push(out)
                }
                let out = Series::new(series.name().clone(), &avs);
                let dtype = out.dtype();
                if dtype.is_nested() {
                    check_nested_object(dtype)?;
                }

                return Ok(out.into());
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
                Some(DataType::Int128) => {
                    let ca: Int128Chunked = dispatch_apply!(
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
                Some(DataType::UInt128) => {
                    let ca: UInt128Chunked = dispatch_apply!(
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
                    check_nested_object(&inner)?;
                    // Make sure the function returns a Series of the correct data type.
                    let function_owned = function.clone().unbind();
                    let dtype_py = Wrap((*inner).clone());
                    let function_wrapped =
                        PyCFunction::new_closure(py, None, None, move |args, _kwargs| {
                            Python::attach(|py| {
                                let out = function_owned.call1(py, args)?;
                                if out.is_none(py) {
                                    Ok(py.None())
                                } else {
                                    pl_series(py).call1(py, ("", out, &dtype_py))
                                }
                            })
                        })?
                        .into_any()
                        .unbind();

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
                Some(DataType::Object(_)) => {
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

            assert!(out.dtype().is_known());
            Ok(out.into())
        })
    }
}
