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
        use polars_utils::pl_serialize;

        if !buf.starts_with(PYTHON_SERDE_MAGIC_BYTE_MARK) {
            polars_bail!(InvalidOperation: "serialization expected python magic byte mark");
        }
        let buf = &buf[PYTHON_SERDE_MAGIC_BYTE_MARK.len()..];

        // Load UDF metadata
        let mut reader = Cursor::new(buf);
        let (output_type, is_elementwise, returns_scalar): (Option<DataTypeExpr>, bool, bool) =
            pl_serialize::deserialize_from_reader::<_, _, true>(&mut reader)?;

        let buf = &buf[reader.position() as usize..];
        let python_function = pl_serialize::python_object_deserialize(buf)?;

        Ok(Arc::new(Self::new(
            python_function,
            output_type,
            is_elementwise,
            returns_scalar,
        )))
    }
}

impl DataFrameUdf for polars_utils::python_function::PythonFunction {
    fn call_udf(&self, df: DataFrame) -> PolarsResult<DataFrame> {
        let func = unsafe { CALL_DF_UDF_PYTHON.unwrap() };
        func(df, &self.0)
    }
}

impl ColumnsUdf for PythonUdfExpression {
    fn resolve_dsl(
        &self,
        input_schema: &Schema,
        self_dtype: Option<&DataType>,
    ) -> PolarsResult<()> {
        if let Some(output_type) = self.output_type.as_ref() {
            let dtype = output_type
                .clone()
                .into_datatype_with_opt_self(input_schema, self_dtype)?;
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
        use polars_utils::pl_serialize;

        // Write byte marks
        buf.extend_from_slice(PYTHON_SERDE_MAGIC_BYTE_MARK);

        // Write UDF metadata
        pl_serialize::serialize_into_writer::<_, _, true>(
            &mut *buf,
            &(
                self.output_type.clone(),
                self.is_elementwise,
                self.returns_scalar,
            ),
        )?;

        pl_serialize::python_object_serialize(&self.python_function, buf)?;
        Ok(())
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
                    output_type
                        .clone()
                        .into_datatype_with_self(input_schema, fields[0].dtype())?
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
