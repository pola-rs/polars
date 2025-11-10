use std::borrow::Cow;

use arrow::bitmap::bitmask::BitMask;
use polars::error::{polars_err, PolarsResult};
use polars::prelude::{
    ChunkCast, ChunkedArray, Column, DataType, Field, PlSmallStr, Scalar, Schema, UInt64Type,
};
use polars::series::{IntoSeries, Series};
use pyo3_polars::export::polars_ffi::version_1::{GroupPositions, PolarsPlugin};
use pyo3_polars::polars_plugin_expr_info;
use pyo3_polars::v1::PolarsPluginExprInfo;
use serde::{Deserialize, Serialize};

struct Count;
#[derive(Serialize, Deserialize)]
struct CountState(PlSmallStr, u64);

impl PolarsPlugin for Count {
    type State = CountState;

    fn serialize(&self) -> PolarsResult<Box<[u8]>> {
        Ok(Box::default())
    }

    fn deserialize(_buffer: &[u8]) -> PolarsResult<Self> {
        Ok(Self)
    }

    fn serialize_state(&self, state: &Self::State) -> PolarsResult<Box<[u8]>> {
        Ok(
            bincode::serde::encode_to_vec(&state, bincode::config::standard())
                .map_err(|err| polars_err!(InvalidOperation: "failed to serialize: {err}"))?
                .into(),
        )
    }

    fn deserialize_state(&self, buffer: &[u8]) -> PolarsResult<Self::State> {
        let (data, num_bytes) =
            bincode::serde::decode_from_slice(buffer, bincode::config::standard())
                .map_err(|err| polars_err!(InvalidOperation: "failed to deserialize: {err}"))?;
        assert_eq!(num_bytes, buffer.len());
        Ok(data)
    }

    fn to_field(&self, fields: &Schema) -> PolarsResult<Field> {
        assert_eq!(fields.len(), 1);
        let field = fields.get_at_index(0).unwrap();
        Ok(Field::new(field.0.clone(), DataType::UInt64))
    }

    fn combine(&self, state: &mut Self::State, other: &Self::State) -> PolarsResult<()> {
        state.1 += other.1;
        Ok(())
    }

    fn step(&self, state: &mut Self::State, inputs: &[Series]) -> PolarsResult<Option<Series>> {
        assert_eq!(inputs.len(), 1);
        state.1 += (inputs[0].len() - inputs[0].null_count()) as u64;
        Ok(None)
    }

    fn finalize(&self, state: &mut Self::State) -> PolarsResult<Option<Series>> {
        let data = Scalar::from(state.1 as u64);
        Ok(Some(data.into_series(state.0.clone())))
    }
    fn new_state(&self, fields: &Schema) -> PolarsResult<Self::State> {
        let field = fields.get_at_index(0).unwrap();
        Ok(CountState(field.0.clone(), 0))
    }

    fn new_empty(&self, state: &Self::State) -> PolarsResult<Self::State> {
        Ok(CountState(state.0.clone(), 0))
    }

    fn reset(&self, state: &mut Self::State) -> PolarsResult<()> {
        state.1 = 0;
        Ok(())
    }

    unsafe fn evaluate_on_groups<'a>(
        &self,
        inputs: &[(Series, &'a GroupPositions)],
    ) -> PolarsResult<(Series, Cow<'a, GroupPositions>)> {
        assert_eq!(inputs.len(), 1);
        let (data, positions) = &inputs[0];
        let name = data.name().clone();

        match positions {
            GroupPositions::SharedAcrossGroups { num_groups: _ } => {
                let data = data.len() - data.null_count();
                let data = Scalar::from(data as u64);
                let data = data.into_series(name);
                Ok((data, Cow::Borrowed(positions)))
            },
            GroupPositions::ScalarPerGroup => {
                let data = if data.has_nulls() {
                    data.is_not_null().cast(&DataType::UInt64)?
                } else {
                    Column::new_scalar(name, Scalar::from(1u64), data.len())
                        .take_materialized_series()
                };
                Ok((data, Cow::Borrowed(positions)))
            },
            GroupPositions::Slice(slice_groups) => {
                let data: Vec<u64> = match data.rechunk_validity() {
                    None => slice_groups.lengths().map(|v| v as u64).collect(),
                    Some(validity) => {
                        let validity = BitMask::from_bitmap(&validity);
                        slice_groups
                            .iter()
                            .map(|g| {
                                validity
                                    .sliced(g.offset as usize, g.length as usize)
                                    .set_bits() as u64
                            })
                            .collect()
                    },
                };
                let data = <ChunkedArray<UInt64Type>>::from_vec(name, data);
                let data = data.into_series();
                Ok((data, Cow::Owned(GroupPositions::ScalarPerGroup)))
            },
            GroupPositions::Index(index_groups) => {
                let data: Vec<u64> = match data.rechunk_validity() {
                    None => index_groups.lengths().map(|v| v as u64).collect(),
                    Some(validity) => {
                        let validity = BitMask::from_bitmap(&validity);
                        index_groups
                            .iter()
                            .map(|g| {
                                g.iter()
                                    .map(|i| {
                                        u64::from(unsafe {
                                            validity.get_bit_unchecked(*i as usize)
                                        })
                                    })
                                    .sum::<u64>()
                            })
                            .collect()
                    },
                };
                let data = <ChunkedArray<UInt64Type>>::from_vec(name, data);
                let data = data.into_series();
                Ok((data, Cow::Owned(GroupPositions::ScalarPerGroup)))
            },
        }
    }
}

#[pyo3::pyfunction]
pub fn count() -> PolarsPluginExprInfo {
    polars_plugin_expr_info!("count", Count, Count)
}
