use std::io::Cursor;
use std::sync::Arc;

use polars_core::datatypes::{DataType, Field};
use polars_core::error::*;
use polars_core::frame::column::Column;
use polars_core::frame::DataFrame;
use polars_core::schema::Schema;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use super::expr_dyn_fn::*;
use crate::constants::MAP_LIST_NAME;
use crate::prelude::*;

// Will be overwritten on Python Polars start up.
pub static mut CALL_COLUMNS_UDF_PYTHON: Option<
    fn(s: Column, lambda: &PyObject) -> PolarsResult<Column>,
> = None;
pub static mut CALL_DF_UDF_PYTHON: Option<
    fn(s: DataFrame, lambda: &PyObject) -> PolarsResult<DataFrame>,
> = None;

pub use polars_utils::python_function::{
    PythonFunction, PYTHON3_VERSION, PYTHON_SERDE_MAGIC_BYTE_MARK,
};

pub struct PythonUdfExpression {
    python_function: PyObject,
    output_type: Option<DataType>,
    is_elementwise: bool,
    returns_scalar: bool,
}

impl PythonUdfExpression {
    pub fn new(
        lambda: PyObject,
        output_type: Option<DataType>,
        is_elementwise: bool,
        returns_scalar: bool,
    ) -> Self {
        Self {
            python_function: lambda,
            output_type,
            is_elementwise,
            returns_scalar,
        }
    }

    #[cfg(feature = "serde")]
    pub(crate) fn try_deserialize(buf: &[u8]) -> PolarsResult<Arc<dyn ColumnsUdf>> {
        // Handle byte mark
        debug_assert!(buf.starts_with(PYTHON_SERDE_MAGIC_BYTE_MARK));
        let buf = &buf[PYTHON_SERDE_MAGIC_BYTE_MARK.len()..];

        // Handle pickle metadata
        let use_cloudpickle = buf[0];
        if use_cloudpickle != 0 {
            let ser_py_version = &buf[1..3];
            let cur_py_version = *PYTHON3_VERSION;
            polars_ensure!(
                ser_py_version == cur_py_version,
                InvalidOperation:
                "current Python version {:?} does not match the Python version used to serialize the UDF {:?}",
                (3, cur_py_version[0], cur_py_version[1]),
                (3, ser_py_version[0], ser_py_version[1] )
            );
        }
        let buf = &buf[3..];

        // Load UDF metadata
        let mut reader = Cursor::new(buf);
        let (output_type, is_elementwise, returns_scalar): (Option<DataType>, bool, bool) =
            ciborium::de::from_reader(&mut reader).map_err(map_err)?;

        let remainder = &buf[reader.position() as usize..];

        // Load UDF
        Python::with_gil(|py| {
            let pickle = PyModule::import_bound(py, "pickle")
                .expect("unable to import 'pickle'")
                .getattr("loads")
                .unwrap();
            let arg = (PyBytes::new_bound(py, remainder),);
            let python_function = pickle.call1(arg).map_err(from_pyerr)?;
            Ok(Arc::new(Self::new(
                python_function.into(),
                output_type,
                is_elementwise,
                returns_scalar,
            )) as Arc<dyn ColumnsUdf>)
        })
    }
}

fn from_pyerr(e: PyErr) -> PolarsError {
    PolarsError::ComputeError(format!("error raised in python: {e}").into())
}

impl DataFrameUdf for polars_utils::python_function::PythonFunction {
    fn call_udf(&self, df: DataFrame) -> PolarsResult<DataFrame> {
        let func = unsafe { CALL_DF_UDF_PYTHON.unwrap() };
        func(df, &self.0)
    }
}

impl ColumnsUdf for PythonUdfExpression {
    fn call_udf(&self, s: &mut [Column]) -> PolarsResult<Option<Column>> {
        let func = unsafe { CALL_COLUMNS_UDF_PYTHON.unwrap() };

        let output_type = self
            .output_type
            .clone()
            .unwrap_or_else(|| DataType::Unknown(Default::default()));
        let mut out = func(s[0].clone(), &self.python_function)?;
        if !matches!(output_type, DataType::Unknown(_)) {
            let must_cast = out.dtype().matches_schema_type(&output_type).map_err(|_| {
                polars_err!(
                    SchemaMismatch: "expected output type '{:?}', got '{:?}'; set `return_dtype` to the proper datatype",
                    output_type, out.dtype(),
                )
            })?;
            if must_cast {
                out = out.cast(&output_type)?;
            }
        }

        Ok(Some(out))
    }

    #[cfg(feature = "serde")]
    fn try_serialize(&self, buf: &mut Vec<u8>) -> PolarsResult<()> {
        // Write byte marks
        buf.extend_from_slice(PYTHON_SERDE_MAGIC_BYTE_MARK);

        Python::with_gil(|py| {
            // Try pickle to serialize the UDF, otherwise fall back to cloudpickle.
            let pickle = PyModule::import_bound(py, "pickle")
                .expect("unable to import 'pickle'")
                .getattr("dumps")
                .unwrap();
            let pickle_result = pickle.call1((self.python_function.clone_ref(py),));
            let (dumped, use_cloudpickle) = match pickle_result {
                Ok(dumped) => (dumped, false),
                Err(_) => {
                    let cloudpickle = PyModule::import_bound(py, "cloudpickle")
                        .map_err(from_pyerr)?
                        .getattr("dumps")
                        .unwrap();
                    let dumped = cloudpickle
                        .call1((self.python_function.clone_ref(py),))
                        .map_err(from_pyerr)?;
                    (dumped, true)
                },
            };

            // Write pickle metadata
            buf.push(use_cloudpickle as u8);
            buf.extend_from_slice(&*PYTHON3_VERSION);

            // Write UDF metadata
            ciborium::ser::into_writer(
                &(
                    self.output_type.clone(),
                    self.is_elementwise,
                    self.returns_scalar,
                ),
                &mut *buf,
            )
            .unwrap();

            // Write UDF
            let dumped = dumped.extract::<PyBackedBytes>().unwrap();
            buf.extend_from_slice(&dumped);
            Ok(())
        })
    }
}

/// Serializable version of [`GetOutput`] for Python UDFs.
pub struct PythonGetOutput {
    return_dtype: Option<DataType>,
}

impl PythonGetOutput {
    pub fn new(return_dtype: Option<DataType>) -> Self {
        Self { return_dtype }
    }

    #[cfg(feature = "serde")]
    pub(crate) fn try_deserialize(buf: &[u8]) -> PolarsResult<Arc<dyn FunctionOutputField>> {
        // Skip header.
        debug_assert!(buf.starts_with(PYTHON_SERDE_MAGIC_BYTE_MARK));
        let buf = &buf[PYTHON_SERDE_MAGIC_BYTE_MARK.len()..];

        let mut reader = Cursor::new(buf);
        let return_dtype: Option<DataType> =
            ciborium::de::from_reader(&mut reader).map_err(map_err)?;

        Ok(Arc::new(Self::new(return_dtype)) as Arc<dyn FunctionOutputField>)
    }
}

impl FunctionOutputField for PythonGetOutput {
    fn get_field(
        &self,
        _input_schema: &Schema,
        _cntxt: Context,
        fields: &[Field],
    ) -> PolarsResult<Field> {
        // Take the name of first field, just like [`GetOutput::map_field`].
        let name = fields[0].name();
        let return_dtype = match self.return_dtype {
            Some(ref dtype) => dtype.clone(),
            None => DataType::Unknown(Default::default()),
        };
        Ok(Field::new(name.clone(), return_dtype))
    }

    #[cfg(feature = "serde")]
    fn try_serialize(&self, buf: &mut Vec<u8>) -> PolarsResult<()> {
        buf.extend_from_slice(PYTHON_SERDE_MAGIC_BYTE_MARK);
        ciborium::ser::into_writer(&self.return_dtype, &mut *buf).unwrap();
        Ok(())
    }
}

impl Expr {
    pub fn map_python(self, func: PythonUdfExpression, agg_list: bool) -> Expr {
        let (collect_groups, name) = if agg_list {
            (ApplyOptions::ApplyList, MAP_LIST_NAME)
        } else if func.is_elementwise {
            (ApplyOptions::ElementWise, "python_udf")
        } else {
            (ApplyOptions::GroupWise, "python_udf")
        };

        let returns_scalar = func.returns_scalar;
        let return_dtype = func.output_type.clone();

        let output_field = PythonGetOutput::new(return_dtype);
        let output_type = SpecialEq::new(Arc::new(output_field) as Arc<dyn FunctionOutputField>);

        let mut flags = FunctionFlags::default() | FunctionFlags::OPTIONAL_RE_ENTRANT;
        if returns_scalar {
            flags |= FunctionFlags::RETURNS_SCALAR;
        }

        Expr::AnonymousFunction {
            input: vec![self],
            function: new_column_udf(func),
            output_type,
            options: FunctionOptions {
                collect_groups,
                fmt_str: name,
                flags,
                ..Default::default()
            },
        }
    }
}
