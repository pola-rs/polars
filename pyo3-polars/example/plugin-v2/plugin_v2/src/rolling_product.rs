use std::collections::VecDeque;

use polars::error::{polars_ensure, polars_err, PolarsResult};
use polars::prelude::{
    ChunkedBuilder, DataType, Field, Int64Type, PrimitiveChunkedBuilder, Schema, SchemaExt,
};
use polars::series::{IntoSeries, Series};
use pyo3_polars::export::polars_ffi::version_1::PolarsPlugin;
use pyo3_polars::polars_plugin_expr_info;
use pyo3_polars::v1::PolarsPluginExprInfo;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct RollingProduct {
    n: usize,
}
#[derive(Serialize, Deserialize, Clone)]
struct RollingProductState {
    product: i64,
    values: VecDeque<i64>,
}

impl PolarsPlugin for RollingProduct {
    type State = RollingProductState;

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
            field.dtype() == &DataType::Int64,
            InvalidOperation: "rolling_product can only be performed on i64"
        );
        Ok(field)
    }

    fn new_state(&self, fields: &Schema) -> PolarsResult<Self::State> {
        assert_eq!(fields.len(), 1);
        Ok(RollingProductState {
            product: 1,
            values: VecDeque::with_capacity(self.n.clone()),
        })
    }

    fn step(&self, state: &mut Self::State, inputs: &[Series]) -> PolarsResult<Option<Series>> {
        assert_eq!(inputs.len(), 1);
        let s = inputs[0].i64()?;
        let mut builder = PrimitiveChunkedBuilder::<Int64Type>::new(s.name().clone(), s.len());
        for v in s.iter() {
            let Some(v) = v else {
                builder.append_null();
                continue;
            };

            if state.values.len() >= self.n {
                state.product /= state.values.pop_front().unwrap();
            }
            state.values.push_back(v);
            state.product *= v;
            builder.append_value(state.product);
        }
        Ok(Some(builder.finish().into_series()))
    }

    fn finalize(&self, _state: &mut Self::State) -> PolarsResult<Option<Series>> {
        unreachable!()
    }

    fn new_empty(&self, state: &Self::State) -> PolarsResult<Self::State> {
        let mut state = state.clone();
        self.reset(&mut state)?;
        Ok(state)
    }

    fn reset(&self, state: &mut Self::State) -> PolarsResult<()> {
        state.product = 1;
        state.values.clear();
        Ok(())
    }

    fn combine(&self, _state: &mut Self::State, _other: &Self::State) -> PolarsResult<()> {
        unreachable!()
    }
}

#[pyo3::pyfunction]
pub fn rolling_product(n: usize) -> PolarsPluginExprInfo {
    assert!(n > 0);
    polars_plugin_expr_info!("rolling_product", RollingProduct { n }, RollingProduct)
}
