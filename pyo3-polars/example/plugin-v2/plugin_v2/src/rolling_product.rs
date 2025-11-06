use std::borrow::Cow;
use std::collections::VecDeque;

use arrow::array::{Array, FixedSizeListArray, ListArray, PrimitiveArray};
use arrow::bitmap::bitmask::BitMask;
use arrow::bitmap::BitmapBuilder;
use arrow::io::ipc::format::ipc::FixedSizeList;
use arrow::types::NativeType;
use polars::error::{polars_ensure, polars_err, PolarsResult};
use polars::prelude::{
    ArrowDataType, ChunkedArray, ChunkedBuilder, DataType, Field, Int64Type, PlSmallStr,
    PrimitiveChunkedBuilder, Schema, SchemaExt,
};
use polars::series::{IntoSeries, Series};
use pyo3_polars::export::polars_ffi::version_1::{
    GroupPositions, IndexGroups, PolarsPlugin, SliceGroup,
};
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

    fn evaluate_on_groups<'a>(
        &self,
        inputs: &[(Series, &'a GroupPositions)],
    ) -> PolarsResult<(Series, Cow<'a, GroupPositions>)> {
        assert_eq!(inputs.len(), 1);
        let (data, groups) = &inputs[0];

        if let GroupPositions::SharedAcrossGroups = groups {
            let mut state = self.new_state(&Schema::from_iter([data.field().into_owned()]))?;
            let data = self.step(&mut state, &[data.clone()])?.unwrap();
            return Ok((data, Cow::Borrowed(groups)));
        };

        if let GroupPositions::ScalarPerGroup = groups {
            return Ok((data.clone(), Cow::Borrowed(groups)));
        };

        let data = data.i64()?;
        let name = data.name().clone();
        let data = data.rechunk();
        let data = data.downcast_as_array();

        let num_groups = match groups {
            GroupPositions::SharedAcrossGroups | GroupPositions::ScalarPerGroup => unreachable!(),
            GroupPositions::Slice(slice_groups) => slice_groups.len(),
            GroupPositions::Index(index_groups) => index_groups.ends.len(),
        };
        let num_values = match groups {
            GroupPositions::SharedAcrossGroups | GroupPositions::ScalarPerGroup => unreachable!(),
            GroupPositions::Slice(slice_groups) => {
                slice_groups.iter().map(|s| s.length as usize).sum()
            },
            GroupPositions::Index(index_groups) => {
                index_groups.ends.last().copied().unwrap_or(0) as usize
            },
        };

        let mut out = Vec::with_capacity(num_values as usize);
        let mut out_bitmap = BitmapBuilder::with_capacity(num_values as usize);
        let mut out_slices = Vec::with_capacity(num_groups);

        let data_values = data.values().as_slice();
        let data_validity = data.validity().map(BitMask::from_bitmap);
        let mut past = VecDeque::with_capacity(self.n);

        match groups {
            GroupPositions::ScalarPerGroup | GroupPositions::SharedAcrossGroups => unreachable!(),
            GroupPositions::Slice(groups) => {
                for slice in groups {
                    past.clear();
                    let offset = out.len() as u64;
                    rolling_product_values(
                        &data_values[slice.offset as usize..][..slice.length as usize],
                        data_validity,
                        &mut 1,
                        &mut past,
                        &mut out,
                        &mut out_bitmap,
                        self.n,
                    );
                    let length = out.len() as u64 - offset;
                    out_slices.push(SliceGroup { offset, length });
                }
            },
            GroupPositions::Index(indices) => {
                for i in indices.iter() {
                    past.clear();
                    let offset = out.len();
                    rolling_product_indices(
                        data_values,
                        data_validity,
                        i,
                        &mut past,
                        &mut out,
                        &mut out_bitmap,
                        self.n,
                    );
                    let length = out.len() - offset;
                    out_slices.push(SliceGroup {
                        offset: offset as u64,
                        length: length as u64,
                    });
                }
            },
        }

        let out = PrimitiveArray::new(
            ArrowDataType::Int64,
            out.into(),
            out_bitmap.into_opt_validity(),
        );
        let data = unsafe {
            ChunkedArray::<Int64Type>::from_chunks_and_dtype(
                name,
                vec![out.boxed()],
                DataType::Int64,
            )
        }
        .into_series();

        let groups = GroupPositions::Slice(out_slices.into());
        Ok((data, Cow::Owned(groups)))
    }
}

fn rolling_product_values(
    values: &[i64],
    validity: Option<BitMask>,
    init: &mut i64,
    past: &mut VecDeque<i64>,
    out: &mut Vec<i64>,
    out_bitmap: &mut BitmapBuilder,
    n: usize,
) {
    match validity {
        None => {
            out_bitmap.extend_constant(values.len(), true);
            out.extend(values.iter().map(|v| {
                if past.len() >= n {
                    *init /= past.pop_front().unwrap();
                }
                past.push_back(*v);
                *init *= v;
                *init
            }));
        },
        Some(validity) => {
            out_bitmap.reserve(values.len());
            out.extend(values.iter().zip(validity.iter()).map(|(v, is_valid)| {
                if !is_valid {
                    out_bitmap.push(false);
                    return 0;
                }

                if past.len() >= n {
                    *init /= past.pop_front().unwrap();
                }
                out_bitmap.push(true);
                past.push_back(*v);
                *init *= v;
                *init
            }));
        },
    }
}

fn rolling_product_indices(
    values: &[i64],
    validity: Option<BitMask>,
    indices: &[u64],
    past: &mut VecDeque<i64>,
    out: &mut Vec<i64>,
    out_bitmap: &mut BitmapBuilder,
    n: usize,
) {
    past.clear();

    let mut product = 1;
    match validity {
        None => {
            out_bitmap.extend_constant(values.len(), true);
            out.extend(indices.iter().map(|i| {
                let v = values[*i as usize];
                if past.len() >= n {
                    product /= past.pop_front().unwrap();
                }
                past.push_back(v);
                product *= v;
                product
            }));
        },
        Some(validity) => {
            out_bitmap.reserve(values.len());
            out.extend(indices.iter().map(|i| {
                let v = values[*i as usize];
                let is_valid = validity.get(*i as usize);

                if !is_valid {
                    out_bitmap.push(false);
                    return 0;
                }

                if past.len() >= n {
                    product /= past.pop_front().unwrap();
                }
                out_bitmap.push(true);
                past.push_back(v);
                product *= v;
                product
            }));
        },
    }
}

#[pyo3::pyfunction]
pub fn rolling_product(n: usize) -> PolarsPluginExprInfo {
    assert!(n > 0);
    polars_plugin_expr_info!("rolling_product", RollingProduct { n }, RollingProduct)
}
