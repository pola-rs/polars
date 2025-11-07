use std::borrow::Cow;

use polars::error::{polars_bail, PolarsResult};
use polars::prelude::{ArithmeticChunked, ChunkCast, DataType, Field, IdxCa, Schema};
use polars::series::{IntoSeries, Series};
use pyo3_polars::export::polars_ffi::version_1::{GroupPositions, PolarsPlugin};
use pyo3_polars::polars_plugin_expr_info;
use pyo3_polars::v1::PolarsPluginExprInfo;

struct HorizontalCount;

impl PolarsPlugin for HorizontalCount {
    type State = ();

    fn serialize(&self) -> PolarsResult<Box<[u8]>> {
        Ok(Default::default())
    }

    fn deserialize(_data: &[u8]) -> PolarsResult<Self> {
        Ok(HorizontalCount)
    }

    fn serialize_state(&self, _state: &Self::State) -> PolarsResult<Box<[u8]>> {
        Ok(Default::default())
    }

    fn deserialize_state(&self, _data: &[u8]) -> PolarsResult<Self::State> {
        Ok(())
    }

    fn to_field(&self, _fields: &Schema) -> PolarsResult<Field> {
        Ok(Field::new("horizontal_count".into(), DataType::IDX_DTYPE))
    }

    fn new_state(&self, _fields: &Schema) -> PolarsResult<Self::State> {
        Ok(())
    }

    fn step(&self, _state: &mut Self::State, inputs: &[Series]) -> PolarsResult<Option<Series>> {
        let mut length = 1;
        for i in inputs {
            if i.len() == 1 {
                continue;
            }
            if length != 1 && i.len() != length {
                polars_bail!(length_mismatch = "horizontal_count", length, i.len());
            }
            length = i.len();
        }

        let mut acc = IdxCa::new_vec("horizontal_count".into(), vec![0; length]);
        for i in inputs {
            let i = i.is_not_null();
            let i = i.cast(&DataType::IDX_DTYPE).unwrap();
            let i = i.idx()?;
            acc = acc.wrapping_add(i.clone());
        }

        Ok(Some(acc.into_series()))
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

    fn evaluate_on_groups<'a>(
        &self,
        inputs: &[(Series, &'a GroupPositions)],
    ) -> PolarsResult<(Series, Cow<'a, GroupPositions>)> {
        _ = inputs;
        unreachable!()
    }
}

#[pyo3::pyfunction]
pub fn horizontal_count() -> PolarsPluginExprInfo {
    polars_plugin_expr_info!("horizontal_count", HorizontalCount, HorizontalCount)
}
