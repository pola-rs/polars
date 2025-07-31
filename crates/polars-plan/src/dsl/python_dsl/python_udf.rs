use std::io::Cursor;
use std::sync::{Arc, OnceLock};

use polars_core::datatypes::{DataType, Field};
use polars_core::error::*;
use polars_core::frame::DataFrame;
use polars_core::frame::column::Column;
use polars_core::prelude::{IntoColumn, StructChunked};
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

    is_elementwise: bool,
    returns_scalar: bool,

    output_type: Option<DataTypeExpr>,
    materialized_output_type: OnceLock<DataType>,
}

impl PythonUdfExpression {
    pub fn new(
        lambda: PyObject,

        is_elementwise: bool,
        returns_scalar: bool,

        output_type: Option<impl Into<DataTypeExpr>>,
    ) -> Self {
        let output_type = output_type.map(Into::into);
        Self {
            python_function: lambda,

            is_elementwise,
            returns_scalar,

            output_type,
            materialized_output_type: OnceLock::new(),
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
            is_elementwise,
            returns_scalar,
            output_type,
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

        let Some(output_dtype) = self.materialized_output_type.get() else {
            unreachable!("UDF called before output datatype was materialized")
        };

        func(
            s[0].clone(),
            Some(output_dtype.clone()),
            &self.python_function,
        )
    }
}

impl AnonymousColumnsUdf for PythonUdfExpression {
    fn as_column_udf(self: Arc<Self>) -> Arc<dyn ColumnsUdf> {
        self as _
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
        // Take the name of first field, just like [`GetOutput::map_field`].
        let name = fields[0].name();
        let return_dtype = match self.materialized_output_type.get() {
            Some(dtype) => dtype.clone(),
            None => {
                let dtype = match self.output_type.as_ref() {
                    None => {
                        if self.is_elementwise {
                            let columns: Vec<Column> = fields
                                .iter()
                                .map(|f| Column::full_null(f.name().clone(), 0, f.dtype()))
                                .collect();

                            let func = unsafe { CALL_COLUMNS_UDF_PYTHON.unwrap() };
                            let out = func(
                                StructChunked::from_columns(
                                    fields
                                        .get(0)
                                        .map_or(PlSmallStr::EMPTY, |f| f.name().clone()),
                                    0,
                                    &columns,
                                )?
                                .into_column(),
                                None,
                                &self.python_function,
                            )?;

                            if out.dtype().is_known() {
                                return Ok(Field::new(out.name().clone(), out.dtype().clone()));
                            }
                        }

                        polars_bail!(InvalidOperation: "unable to determine output type of map expression. Consider providing an output datatype");
                    },
                    Some(return_dtype) => return_dtype
                        .clone()
                        .into_datatype_with_self(input_schema, fields[0].dtype())?,
                };
                self.materialized_output_type.get_or_init(|| dtype.clone());
                dtype
            },
        };
        Ok(Field::new(name.clone(), return_dtype))
    }
}

impl Expr {
    pub fn map_python(self, func: PythonUdfExpression) -> Expr {
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
            input: vec![self],
            function: new_column_udf(func),
            options: FunctionOptions {
                flags,
                ..Default::default()
            },
            fmt_str: Box::new(PlSmallStr::from(NAME)),
        }
    }
}
