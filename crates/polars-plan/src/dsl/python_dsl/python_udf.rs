use std::io::Cursor;
use std::sync::{Arc, Mutex, OnceLock};

use polars_core::datatypes::{DataType, Field};
use polars_core::error::*;
use polars_core::frame::DataFrame;
use polars_core::frame::column::Column;
use polars_core::schema::Schema;
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;

use crate::dsl::udf::try_infer_udf_output_dtype;
use crate::prelude::*;

// Will be overwritten on Python Polars start up.
type PythonColumnUdf =
    fn(s: &[Column], output_dtype: Option<DataType>, lambda: &Py<PyAny>) -> PolarsResult<Column>;
pub static CALL_PYTHON_COLUMNS_UDF: OnceLock<PythonColumnUdf> = OnceLock::new();

type PythonDfUdf = fn(s: DataFrame, lambda: &Py<PyAny>) -> PolarsResult<DataFrame>;
pub static CALL_PYTHON_DF_UDF: OnceLock<PythonDfUdf> = OnceLock::new();

pub use polars_utils::python_function::PythonFunction;
#[cfg(feature = "serde")]
pub use polars_utils::python_function::{PYTHON_SERDE_MAGIC_BYTE_MARK, PYTHON3_VERSION};

pub struct PythonUdfExpression {
    python_function: Py<PyAny>,
    output_type: Option<DataTypeExpr>,
    output_type_reads_schema: bool,
    materialized: OnceLock<MaterializedDtype>,
    /// Copies resolved for other inputs than `materialized`, shared by uses with those inputs.
    copies: Mutex<Vec<Arc<PythonUdfExpression>>>,
    is_elementwise: bool,
    returns_scalar: bool,
}

const MAX_COPIES: usize = 16;

struct MaterializedDtype {
    dtype: DataType,
    /// Input fields it was resolved for, and the schema if the output type reads it. `None` if
    /// it was deserialized.
    inputs: Option<(Vec<Field>, Option<Schema>)>,
}

impl MaterializedDtype {
    fn is_for(&self, input_schema: &Schema, fields: &[Field], match_names: bool) -> bool {
        self.inputs.as_ref().is_some_and(|(inputs, schema)| {
            inputs.len() == fields.len()
                && inputs
                    .iter()
                    .zip(fields)
                    .all(|(a, b)| a.dtype() == b.dtype() && (!match_names || a.name() == b.name()))
                && schema.as_ref().is_none_or(|s| s == input_schema)
        })
    }
}

fn reads_schema(dt_expr: &DataTypeExpr) -> bool {
    use DataTypeExpr as D;
    match dt_expr {
        D::Literal(_) | D::SelfDtype => false,
        D::OfExpr(_) => true,
        D::InnerDataType { input, .. }
        | D::Int(input, _)
        | D::Struct(input, _)
        | D::WrapInList(input)
        | D::WrapInArray(input, _) => reads_schema(input),
        D::StructWithFields(fields) => fields.iter().any(|(_, dt_expr)| reads_schema(dt_expr)),
    }
}

impl PythonUdfExpression {
    pub fn new(
        lambda: Py<PyAny>,
        output_type: Option<impl Into<DataTypeExpr>>,
        is_elementwise: bool,
        returns_scalar: bool,
    ) -> Self {
        let output_type = output_type.map(Into::into);
        Self {
            python_function: lambda,
            output_type_reads_schema: output_type.as_ref().is_some_and(reads_schema),
            output_type,
            materialized: OnceLock::new(),
            copies: Mutex::new(Vec::new()),
            is_elementwise,
            returns_scalar,
        }
    }

    fn fresh_copy(&self) -> Self {
        Self::new(
            Python::attach(|py| self.python_function.clone_ref(py)),
            self.output_type.clone(),
            self.is_elementwise,
            self.returns_scalar,
        )
    }

    /// Resolves the output dtype if needed. Returns `false` if it was resolved for other inputs.
    fn materialize_for(&self, input_schema: &Schema, fields: &[Field]) -> PolarsResult<bool> {
        let materialized = match self.materialized.get() {
            Some(m) => m,
            None => {
                let dtype = match self.output_type.as_ref() {
                    None => {
                        let func = CALL_PYTHON_COLUMNS_UDF.get().unwrap();
                        let f = |s: &[Column]| func(s, None, &self.python_function);
                        try_infer_udf_output_dtype(&f as _, fields)?
                    },
                    Some(output_type) => output_type
                        .clone()
                        .into_datatype_with_self(input_schema, fields[0].dtype())?,
                };
                let schema = self.output_type_reads_schema.then(|| input_schema.clone());
                self.materialized.get_or_init(|| MaterializedDtype {
                    dtype,
                    inputs: Some((fields.to_vec(), schema)),
                })
            },
        };
        // Inference calls the function on named inputs, so its output can depend on them.
        Ok(materialized.is_for(input_schema, fields, self.output_type.is_none()))
    }

    fn materialized_dtype(&self) -> &DataType {
        &self
            .materialized
            .get()
            .expect("should have been materialized at this point")
            .dtype
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
        let (output_type, materialized, is_elementwise, returns_scalar): (
            Option<DataTypeExpr>,
            Option<Field>,
            bool,
            bool,
        ) = pl_serialize::deserialize_from_reader::<_, _, true>(&mut reader)?;

        let buf = &buf[reader.position() as usize..];
        let python_function = pl_serialize::python_object_deserialize(buf)?;

        let mut udf = Self::new(python_function, output_type, is_elementwise, returns_scalar);
        if let Some(field) = materialized {
            udf.materialized = OnceLock::from(MaterializedDtype {
                dtype: field.dtype,
                inputs: None,
            });
        }

        Ok(Arc::new(udf))
    }
}

impl DataFrameUdf for polars_utils::python_function::PythonFunction {
    fn call_udf(&self, df: DataFrame) -> PolarsResult<DataFrame> {
        let func = CALL_PYTHON_DF_UDF.get().unwrap();
        func(df, &self.0)
    }

    fn display_str(&self) -> PlSmallStr {
        pyo3::Python::attach(|py| {
            use polars_utils::format_pl_smallstr;
            use pyo3::intern;
            use pyo3::pybacked::PyBackedStr;

            let class_name: PyBackedStr = self
                .0
                .getattr(py, intern!(py, "__class__"))
                .unwrap()
                .extract(py)
                .unwrap();

            format_pl_smallstr!("PythonUdf({class_name})")
        })
    }
}

impl ColumnsUdf for PythonUdfExpression {
    fn call_udf(&self, s: &mut [Column]) -> PolarsResult<Column> {
        let func = CALL_PYTHON_COLUMNS_UDF.get().unwrap();
        let dtype = self.materialized_dtype();
        let mut out = func(s, Some(dtype.clone()), &self.python_function)?;

        let must_cast = out.dtype().matches_schema_type(dtype).map_err(|_| {
            polars_err!(
                SchemaMismatch: "expected output type '{:?}', got '{:?}'; set `return_dtype` to the proper datatype",
                dtype, out.dtype(),
            )
        })?;
        if must_cast {
            out = out.cast(dtype)?;
        }

        Ok(out)
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
                self.materialized
                    .get()
                    .map(|m| Field::new(PlSmallStr::EMPTY, m.dtype.clone())),
                self.is_elementwise,
                self.returns_scalar,
            ),
        )?;

        pl_serialize::python_object_serialize(&self.python_function, buf)?;
        Ok(())
    }

    fn get_field(&self, input_schema: &Schema, fields: &[Field]) -> PolarsResult<Field> {
        if self.materialized.get().is_none() {
            self.materialize_for(input_schema, fields)?;
        }
        // Take the name of first field, just like `map_field`.
        Ok(Field::new(
            fields[0].name().clone(),
            self.materialized_dtype().clone(),
        ))
    }

    fn resolve(
        self: Arc<Self>,
        input_schema: &Schema,
        fields: &[Field],
    ) -> PolarsResult<(Arc<dyn AnonymousColumnsUdf>, Field)> {
        if self.materialize_for(input_schema, fields)? {
            let field = self.get_field(input_schema, fields)?;
            return Ok((self, field));
        }

        let copies = self.copies.lock().unwrap().clone();
        for copy in copies {
            if copy.materialize_for(input_schema, fields)? {
                let field = copy.get_field(input_schema, fields)?;
                return Ok((copy, field));
            }
        }

        let copy = Arc::new(self.fresh_copy());
        let field = copy.get_field(input_schema, fields)?;
        let mut copies = self.copies.lock().unwrap();
        if copies.len() == MAX_COPIES {
            copies.remove(0);
        }
        copies.push(copy.clone());
        Ok((copy, field))
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
