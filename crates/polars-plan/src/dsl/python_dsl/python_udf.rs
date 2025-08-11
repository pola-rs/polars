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
    fn(s: &[Column], output_dtype: Option<DataType>, lambda: &PyObject) -> PolarsResult<Column>,
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
    materialized_field: OnceLock<Field>,
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
            materialized_field: OnceLock::new(),
            is_elementwise,
            returns_scalar,
        }
    }

    #[cfg(feature = "serde")]
    pub(crate) fn try_deserialize(buf: &[u8]) -> PolarsResult<Arc<dyn AnonymousColumnsUdf>> {
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
    fn call_udf(&self, s: &mut [Column]) -> PolarsResult<Column> {
        let func = unsafe { CALL_COLUMNS_UDF_PYTHON.unwrap() };
        let field = self.materialized_field.get().map_or_else(
            || {
                Field::new(
                    s.first()
                        .map_or(PlSmallStr::from_static("udf"), |s| s.name().clone()),
                    DataType::Unknown(Default::default()),
                )
            },
            |f| f.clone(),
        );
        let mut out = func(
            s,
            self.materialized_field.get().cloned().map(|f| f.dtype),
            &self.python_function,
        )?;

        if !matches!(field.dtype(), DataType::Unknown(_)) {
            let must_cast = out.dtype().matches_schema_type(field.dtype()).map_err(|_| {
                polars_err!(
                    SchemaMismatch: "expected output type '{:?}', got '{:?}'; set `return_dtype` to the proper datatype",
                    field.dtype(), out.dtype(),
                )
            })?;
            if must_cast {
                out = out.cast(field.dtype())?;
            }
        }

        Ok(out)
    }
}

impl AnonymousColumnsUdf for PythonUdfExpression {
    fn as_column_udf(self: Arc<Self>) -> Arc<dyn ColumnsUdf> {
        self as _
    }
    fn deep_clone(self: Arc<Self>) -> Arc<dyn AnonymousColumnsUdf> {
        Arc::new(Self {
            python_function: Python::with_gil(|py| self.python_function.clone_ref(py)),
            output_type: self.output_type.clone(),
            materialized_field: OnceLock::new(),
            is_elementwise: self.is_elementwise,
            returns_scalar: self.returns_scalar,
        }) as _
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

    fn get_field(&self, input_schema: &Schema, fields: &[Field]) -> PolarsResult<Field> {
        let field = match self.materialized_field.get() {
            Some(f) => f.clone(),
            None => {
                let dtype = if let Some(output_type) = self.output_type.as_ref() {
                    output_type
                        .clone()
                        .into_datatype_with_self(input_schema, fields[0].dtype())?
                } else {
                    DataType::Unknown(UnknownKind::Any)
                };

                // Take the name of first field, just like `map_field`.
                let name = fields[0].name();
                let f = Field::new(name.clone(), dtype);
                self.materialized_field.get_or_init(|| f.clone());
                f
            },
        };
        Ok(field)
    }
}

impl Expr {
    pub fn map_python(self, func: PythonUdfExpression) -> Expr {
        Self::map_many_python(vec![self], func)
    }

    pub fn map_many_python(exprs: Vec<Expr>, func: PythonUdfExpression) -> Expr {
        const NAME: &str = "python_udf";

        let returns_scalar = func.returns_scalar;

        let mut flags = FunctionFlags::default() | FunctionFlags::OPTIONAL_RE_ENTRANT;
        if func.is_elementwise {
            flags.set_elementwise();
        }
        if returns_scalar {
            flags |= FunctionFlags::RETURNS_SCALAR;
        }

        Expr::AnonymousFunction {
            input: exprs,
            function: new_column_udf(func),
            options: FunctionOptions {
                flags,
                ..Default::default()
            },
            fmt_str: Box::new(PlSmallStr::from(NAME)),
        }
    }
}
