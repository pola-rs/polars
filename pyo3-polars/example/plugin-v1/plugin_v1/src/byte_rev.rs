use polars::error::{polars_ensure, PolarsResult};
use polars::prelude::{
    ChunkedBuilder, DataType, Field, Int64Type, PrimitiveChunkedBuilder, Schema, SchemaExt,
};
use polars::series::{IntoSeries, Series};
use pyo3_polars::polars_plugin_expr_info;
use pyo3_polars::v1::{self, PolarsPluginExprInfo};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct ByteRev;

impl v1::elementwise::PolarsElementwisePlugin for ByteRev {
    fn to_field(&self, fields: &Schema) -> PolarsResult<Field> {
        assert_eq!(fields.len(), 1);
        let field = fields.iter_fields().next().unwrap();
        polars_ensure!(
            field.dtype() == &DataType::Int64,
            InvalidOperation: "`byte_rev` can only be performed on i64"
        );
        Ok(field)
    }

    fn evaluate(&self, inputs: &[Series]) -> PolarsResult<Series> {
        assert_eq!(inputs.len(), 1);
        let s = inputs[0].i64()?;
        let mut builder = PrimitiveChunkedBuilder::<Int64Type>::new(s.name().clone(), s.len());
        for v in s.iter() {
            let Some(v) = v else {
                builder.append_null();
                continue;
            };
            builder.append_value(v.swap_bytes());
        }
        Ok(builder.finish().into_series())
    }
}

#[pyo3::pyfunction]
pub fn byte_rev() -> PolarsPluginExprInfo {
    polars_plugin_expr_info!(
        "byte_rev",
        v1::elementwise::Plugin(ByteRev),
        v1::elementwise::Plugin<ByteRev>
    )
}
