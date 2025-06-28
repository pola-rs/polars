use std::io::Cursor;
use std::sync::{Arc, OnceLock};

use polars_core::datatypes::{DataType, Field};
use polars_core::error::*;
use polars_core::frame::DataFrame;
use polars_core::frame::column::Column;
use polars_core::prelude::UnknownKind;
use polars_core::schema::Schema;
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use crate::prelude::*;

// Will be overwritten on Python Polars start up.
#[allow(clippy::type_complexity)]
pub static mut CALL_COLUMNS_UDF_PYTHON: Option<
    fn(s: Column, output_dtype: Option<DataType>, lambda: &PyObject) -> PolarsResult<Column>,
> = None;
pub static mut CALL_DF_UDF_PYTHON: Option<
    fn(s: DataFrame, lambda: &PyObject) -> PolarsResult<DataFrame>,
> = None;

pub use polars_utils::python_function::PythonFunction;
#[cfg(feature = "serde")]
pub use polars_utils::python_function::{PYTHON_SERDE_MAGIC_BYTE_MARK, PYTHON3_VERSION};

pub struct PythonUdfExpression {
    python_function: PyObject,
    output_type: Option<DataTypeExpr>,
    materialized_output_type: OnceLock<DataType>,
    is_elementwise: bool,
    returns_scalar: bool,
}

impl PythonUdfExpression {
    pub fn new(
        lambda: PyObject,
        output_type: Option<impl Into<DataTypeExpr>>,
        is_elementwise: bool,
        returns_scalar: bool,
    ) -> Self {
        let output_type = output_type.map(Into::into);
        Self {
            python_function: lambda,
            output_type,
            materialized_output_type: OnceLock::new(),
            is_elementwise,
            returns_scalar,
        }
    }

    #[cfg(feature = "serde")]
    pub(crate) fn try_deserialize(buf: &[u8]) -> PolarsResult<Arc<dyn ColumnsUdf>> {
        // Handle byte mark

        use polars_utils::pl_serialize;
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
        let (output_type, is_elementwise, returns_scalar): (Option<DataTypeExpr>, bool, bool) =
            pl_serialize::deserialize_from_reader::<_, _, true>(&mut reader)?;

        let remainder = &buf[reader.position() as usize..];

        // Load UDF
        Python::with_gil(|py| {
            let pickle = PyModule::import(py, "pickle")
                .expect("unable to import 'pickle'")
                .getattr("loads")
                .unwrap();
            let arg = (PyBytes::new(py, remainder),);
            let python_function = pickle.call1(arg)?;
            Ok(Arc::new(Self::new(
                python_function.into(),
                output_type,
                is_elementwise,
                returns_scalar,
            )) as Arc<dyn ColumnsUdf>)
        })
    }
}

impl DataFrameUdf for polars_utils::python_function::PythonFunction {
    fn call_udf(&self, df: DataFrame) -> PolarsResult<DataFrame> {
        let func = unsafe { CALL_DF_UDF_PYTHON.unwrap() };
        func(df, &self.0)
    }
}

impl ColumnsUdf for PythonUdfExpression {
    fn resolve_dsl(&self, input_schema: &Schema) -> PolarsResult<()> {
        if let Some(output_type) = self.output_type.as_ref() {
            let dtype = output_type.clone().into_datatype(input_schema)?;
            self.materialized_output_type.get_or_init(|| dtype);
        }
        Ok(())
    }

    fn call_udf(&self, s: &mut [Column]) -> PolarsResult<Option<Column>> {
        let func = unsafe { CALL_COLUMNS_UDF_PYTHON.unwrap() };

        let output_type = self
            .materialized_output_type
            .get()
            .map_or_else(|| DataType::Unknown(Default::default()), |dt| dt.clone());
        let mut out = func(
            s[0].clone(),
            self.materialized_output_type.get().cloned(),
            &self.python_function,
        )?;
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

        use polars_utils::pl_serialize;
        buf.extend_from_slice(PYTHON_SERDE_MAGIC_BYTE_MARK);

        Python::with_gil(|py| {
            // Try pickle to serialize the UDF, otherwise fall back to cloudpickle.
            let pickle = PyModule::import(py, "pickle")
                .expect("unable to import 'pickle'")
                .getattr("dumps")
                .unwrap();
            let pickle_result = pickle.call1((self.python_function.clone_ref(py),));
            let (dumped, use_cloudpickle) = match pickle_result {
                Ok(dumped) => (dumped, false),
                Err(_) => {
                    let cloudpickle = PyModule::import(py, "cloudpickle")?
                        .getattr("dumps")
                        .unwrap();
                    let dumped = cloudpickle.call1((self.python_function.clone_ref(py),))?;
                    (dumped, true)
                },
            };

            // Write pickle metadata
            buf.push(use_cloudpickle as u8);
            buf.extend_from_slice(&*PYTHON3_VERSION);

            // Write UDF metadata
            pl_serialize::serialize_into_writer::<_, _, true>(
                &mut *buf,
                &(
                    self.output_type.clone(),
                    self.is_elementwise,
                    self.returns_scalar,
                ),
            )?;

            // Write UDF
            let dumped = dumped.extract::<PyBackedBytes>().unwrap();
            buf.extend_from_slice(&dumped);
            Ok(())
        })
    }
}

/// Serializable version of [`GetOutput`] for Python UDFs.
pub struct PythonGetOutput {
    return_dtype: Option<DataTypeExpr>,
    materialized_output_type: OnceLock<DataType>,
}

impl PythonGetOutput {
    pub fn new(return_dtype: Option<impl Into<DataTypeExpr>>) -> Self {
        Self {
            return_dtype: return_dtype.map(Into::into),
            materialized_output_type: OnceLock::new(),
        }
    }

    #[cfg(feature = "serde")]
    pub(crate) fn try_deserialize(buf: &[u8]) -> PolarsResult<Arc<dyn FunctionOutputField>> {
        // Skip header.

        use polars_utils::pl_serialize;
        debug_assert!(buf.starts_with(PYTHON_SERDE_MAGIC_BYTE_MARK));
        let buf = &buf[PYTHON_SERDE_MAGIC_BYTE_MARK.len()..];

        let mut reader = Cursor::new(buf);
        let return_dtype: Option<DataTypeExpr> =
            pl_serialize::deserialize_from_reader::<_, _, true>(&mut reader)?;

        Ok(Arc::new(Self::new(return_dtype)) as Arc<dyn FunctionOutputField>)
    }
}

impl FunctionOutputField for PythonGetOutput {
    fn get_field(
        &self,
        input_schema: &Schema,
        _cntxt: Context,
        fields: &[Field],
    ) -> PolarsResult<Field> {
        // Take the name of first field, just like [`GetOutput::map_field`].
        let name = fields[0].name();
        let return_dtype = match self.materialized_output_type.get() {
            Some(dtype) => dtype.clone(),
            None => {
                let dtype = if let Some(output_type) = self.return_dtype.as_ref() {
                    output_type.clone().into_datatype(input_schema)?
                } else {
                    DataType::Unknown(UnknownKind::Any)
                };

                self.materialized_output_type.get_or_init(|| dtype.clone());
                dtype
            },
        };
        Ok(Field::new(name.clone(), return_dtype))
    }

    #[cfg(feature = "serde")]
    fn try_serialize(&self, buf: &mut Vec<u8>) -> PolarsResult<()> {
        use polars_utils::pl_serialize;

        buf.extend_from_slice(PYTHON_SERDE_MAGIC_BYTE_MARK);
        pl_serialize::serialize_into_writer::<_, _, true>(&mut *buf, &self.return_dtype)
    }
}

impl Expr {
    pub fn map_python(self, func: PythonUdfExpression) -> Expr {
        const NAME: &str = "python_udf";

        let returns_scalar = func.returns_scalar;
        let return_dtype = func.output_type.clone();

        let output_field = PythonGetOutput::new(return_dtype);
        let output_type = LazySerde::Deserialized(SpecialEq::new(
            Arc::new(output_field) as Arc<dyn FunctionOutputField>
        ));

        let mut flags = FunctionFlags::default() | FunctionFlags::OPTIONAL_RE_ENTRANT;
        if func.is_elementwise {
            flags.set_elementwise();
        }
        if returns_scalar {
            flags |= FunctionFlags::RETURNS_SCALAR;
        }

        Expr::AnonymousFunction {
            input: vec![self],
            function: new_column_udf(func),
            output_type,
            options: FunctionOptions {
                flags,
                ..Default::default()
            },
            fmt_str: Box::new(PlSmallStr::from(NAME)),
        }
    }
}
