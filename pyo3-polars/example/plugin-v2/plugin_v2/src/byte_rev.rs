use polars::error::{polars_ensure, PolarsResult};
use polars::prelude::{
    ChunkedBuilder, DataType, Field, Int64Type, PrimitiveChunkedBuilder, Schema, SchemaExt,
};
use polars::series::{IntoSeries, Series};
use pyo3_polars::export::polars_ffi::version_1::PolarsPlugin;
use pyo3_polars::polars_plugin_expr_info;
use pyo3_polars::v1::PolarsPluginExprInfo;

struct ByteRev;

impl PolarsPlugin for ByteRev {
    type State = ();

    fn serialize(&self) -> PolarsResult<Box<[u8]>> {
        Ok(Default::default())
    }

    fn deserialize(_buff: &[u8]) -> PolarsResult<Self> {
        Ok(ByteRev)
    }

    fn serialize_state(&self, _state: &Self::State) -> PolarsResult<Box<[u8]>> {
        Ok(Default::default())
    }

    fn deserialize_state(&self, _buff: &[u8]) -> PolarsResult<Self::State> {
        Ok(())
    }

    fn to_field(&self, fields: &Schema) -> PolarsResult<Field> {
        assert_eq!(fields.len(), 1);
        let field = fields.iter_fields().next().unwrap();
        polars_ensure!(
            field.dtype() == &DataType::Int64,
            InvalidOperation: "`byte_rev` can only be performed on i64"
        );
        Ok(field)
    }

    fn new_state(&self, fields: &Schema) -> PolarsResult<Self::State> {
        assert_eq!(fields.len(), 1);
        Ok(())
    }

    fn step(&self, _state: &mut Self::State, inputs: &[Series]) -> PolarsResult<Option<Series>> {
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
        Ok(Some(builder.finish().into_series()))
    }

    fn finalize(&self, _state: &mut Self::State) -> PolarsResult<Option<Series>> {
        unreachable!()
    }

    fn new_empty(&self, _state: &Self::State) -> PolarsResult<Self::State> {
        Ok(())
    }

    fn reset(&self, _state: &mut Self::State) -> PolarsResult<()> {
        Ok(())
    }

    fn combine(&self, _state: &mut Self::State, _other: &Self::State) -> PolarsResult<()> {
        unreachable!()
    }
}

#[pyo3::pyfunction]
pub fn byte_rev() -> PolarsPluginExprInfo {
    polars_plugin_expr_info!("byte_rev", ByteRev, ByteRev)
}
