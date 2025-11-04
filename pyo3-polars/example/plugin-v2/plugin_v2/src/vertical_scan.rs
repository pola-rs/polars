use polars::error::{polars_ensure, polars_err, PolarsResult};
use polars::prelude::{
    ChunkedBuilder, DataType, Field, Int64Type, PrimitiveChunkedBuilder, Schema, SchemaExt,
};
use polars::series::{IntoSeries, Series};
use pyo3_polars::export::polars_ffi::version_1::PolarsPlugin;
use pyo3_polars::polars_plugin_expr_info;
use pyo3_polars::v1::PolarsPluginExprInfo;
use serde::{Deserialize, Serialize};

// Implementation of https://github.com/pola-rs/polars/issues/12165#issuecomment-2766352413
//
// y[t] = (y[t-1] + 1) % x[t]
#[derive(Serialize, Deserialize)]
struct VerticalScan {
    init: i64,
}
#[derive(Serialize, Deserialize, Clone)]
struct VerticalScanState {
    n: i64,
}

impl PolarsPlugin for VerticalScan {
    type State = VerticalScanState;

    fn serialize(&self) -> PolarsResult<Box<[u8]>> {
        Ok(
            bincode::serde::encode_to_vec(self, bincode::config::standard())
                .map_err(|err| polars_err!(InvalidOperation: "failed to serialize: {err}"))?
                .into(),
        )
    }

    fn deserialize(buff: &[u8]) -> PolarsResult<Self> {
        let (data, num_bytes) =
            bincode::serde::decode_from_slice(buff, bincode::config::standard())
                .map_err(|err| polars_err!(InvalidOperation: "failed to deserialize: {err}"))?;
        assert_eq!(num_bytes, buff.len());
        Ok(data)
    }

    fn serialize_state(&self, state: &Self::State) -> PolarsResult<Box<[u8]>> {
        Ok(
            bincode::serde::encode_to_vec(state, bincode::config::standard())
                .map_err(|err| polars_err!(InvalidOperation: "failed to serialize: {err}"))?
                .into(),
        )
    }

    fn deserialize_state(&self, buff: &[u8]) -> PolarsResult<Self::State> {
        let (state, num_bytes) =
            bincode::serde::decode_from_slice(buff, bincode::config::standard())
                .map_err(|err| polars_err!(InvalidOperation: "failed to deserialize: {err}"))?;
        assert_eq!(num_bytes, buff.len());
        Ok(state)
    }

    fn to_field(&self, fields: &Schema) -> PolarsResult<Field> {
        assert_eq!(fields.len(), 1);
        let field = fields.iter_fields().next().unwrap();
        polars_ensure!(
            field.dtype().is_integer(),
            InvalidOperation: "`vertical_scan` can only be performed on integers"
        );
        Ok(field)
    }

    fn initialize(&self, fields: &Schema) -> PolarsResult<Self::State> {
        assert_eq!(fields.len(), 1);
        Ok(VerticalScanState { n: self.init })
    }

    fn insert(&self, state: &mut Self::State, inputs: &[Series]) -> PolarsResult<Option<Series>> {
        assert_eq!(inputs.len(), 1);

        let x = inputs[0].cast(&DataType::Int64)?;
        let x = x.i64()?;

        let mut builder = PrimitiveChunkedBuilder::<Int64Type>::new(x.name().clone(), x.len());
        for x in x.iter() {
            let Some(x) = x else {
                builder.append_null();
                continue;
            };

            let value = (state.n + 1) % x;
            state.n = value;
            builder.append_value(value);
        }
        Ok(Some(builder.finish().into_series()))
    }

    fn finalize(&self, _state: &mut Self::State) -> PolarsResult<Option<Series>> {
        Ok(None)
    }

    fn new_empty(&self, state: &Self::State) -> PolarsResult<Self::State> {
        let mut state = state.clone();
        self.reset(&mut state)?;
        Ok(state)
    }

    fn reset(&self, state: &mut Self::State) -> PolarsResult<()> {
        state.n = self.init;
        Ok(())
    }
}

#[pyo3::pyfunction]
pub fn vertical_scan(init: i64) -> PolarsPluginExprInfo {
    polars_plugin_expr_info!("vertical_scan", VerticalScan { init }, VerticalScan)
}
