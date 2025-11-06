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

    // data: Series
    // groups:
    //     None
    //     List<u32>     -> [offset_1, ..., offset_n] 
    //     List<u64>     -> [offset_1, ..., offset_n] 
    //     Array<u32, 2> -> [offset, length]
    //     Array<u64, 2> -> [offset, length]
    fn evaluate_on_groups(
        &self,
        inputs: &[(Series, Option<Box<dyn Array>>)],
    ) -> PolarsResult<(Series, Option<Box<dyn Array>>)> {
        assert_eq!(inputs.len(), 1);
        let (data, groups) = &inputs[0];

        fn indices<T: NativeType + Into<u64>>(
            name: PlSmallStr,
            data: &PrimitiveArray<i64>,
            groups: &ListArray<i64>,
            n: usize,
        ) -> (Series, Option<Box<dyn Array>>) {
            let mut out = Vec::with_capacity(groups.values().len());
            let mut out_bitmap = BitmapBuilder::with_capacity(groups.values().len());
            let mut slices = Vec::with_capacity(groups.len());
            assert_eq!(groups.validity(), None);
            let indices = groups.values();
            let indices = indices
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .unwrap();
            assert_eq!(indices.validity(), None);
            let indices = indices.values().as_slice();
            let data_values = data.values().as_slice();
            let data_validity = data.validity().map(BitMask::from_bitmap);
            let mut past = VecDeque::with_capacity(n);
            for (offset, length) in groups.offsets().offset_and_length_iter() {
                past.clear();
                slices.push(offset as u64);
                rolling_product_indices::<T>(
                    data_values,
                    data_validity,
                    &indices[offset..][..length],
                    &mut past,
                    &mut out,
                    &mut out_bitmap,
                    n,
                );
                slices.push(length as u64);
            }

            let out = PrimitiveArray::new(
                ArrowDataType::Int64,
                out.into(),
                out_bitmap.into_opt_validity(),
            );
            let slices = PrimitiveArray::new(ArrowDataType::UInt64, slices.into(), None);
            let slices = FixedSizeListArray::new(
                ArrowDataType::UInt64.to_fixed_size_list(2, false),
                groups.len(),
                slices.boxed(),
                None,
            );

            let data = unsafe {
                ChunkedArray::<Int64Type>::from_chunks_and_dtype(
                    name,
                    vec![out.boxed()],
                    DataType::Int64,
                )
            }
            .into_series();
            let groups = slices.boxed();

            (data, Some(groups))
        }

        fn slices<T: NativeType + Into<u64>>(
            name: PlSmallStr,
            data: &PrimitiveArray<i64>,
            groups: &FixedSizeListArray,
            n: usize,
        ) -> (Series, Option<Box<dyn Array>>) {
            let mut out = Vec::with_capacity(groups.values().len());
            let mut out_bitmap = BitmapBuilder::with_capacity(groups.values().len());
            let mut out_slices = Vec::with_capacity(groups.len());
            assert_eq!(groups.validity(), None);
            let indices = groups.values();
            let indices = indices
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .unwrap();
            assert_eq!(indices.validity(), None);
            let indices = indices.values().as_slice();
            let data_values = data.values().as_slice();
            let data_validity = data.validity().map(BitMask::from_bitmap);
            let mut past = VecDeque::with_capacity(n);
            for i in 0..groups.len() {
                let offset = indices[i * 2].into() as usize;
                let length = indices[i * 2 + 1].into() as usize;

                past.clear();
                out_slices.push(out.len() as u64);
                rolling_product_values(
                    &data_values[offset..][..length],
                    data_validity,
                    &mut 1,
                    &mut past,
                    &mut out,
                    &mut out_bitmap,
                    n,
                );
                out_slices.push((out.len() as u64 - *out_slices.last().unwrap()) as u64);
            }

            let out = PrimitiveArray::new(
                ArrowDataType::Int64,
                out.into(),
                out_bitmap.into_opt_validity(),
            );
            let out_slices = PrimitiveArray::new(ArrowDataType::UInt64, out_slices.into(), None);
            let out_slices = FixedSizeListArray::new(
                ArrowDataType::UInt64.to_fixed_size_list(2, false),
                groups.len(),
                out_slices.boxed(),
                None,
            );

            let data = unsafe {
                ChunkedArray::<Int64Type>::from_chunks_and_dtype(
                    name,
                    vec![out.boxed()],
                    DataType::Int64,
                )
            }
            .into_series();
            let groups = out_slices.boxed();

            (data, Some(groups))
        }

        match groups {
            None => Ok((data.clone(), groups.clone())),
            Some(groups) => {
                let data = data.i64()?;
                let name = data.name().clone();
                let data = data.rechunk();
                let data = data.downcast_as_array();

                use ArrowDataType as D;
                match groups.dtype() {
                    D::LargeList(f) if matches!(f.dtype(), D::UInt32) => Ok(indices::<u32>(
                        name,
                        data,
                        groups.as_any().downcast_ref().unwrap(),
                        self.n,
                    )),
                    D::LargeList(f) if matches!(f.dtype(), D::UInt64) => Ok(indices::<u64>(
                        name,
                        data,
                        groups.as_any().downcast_ref().unwrap(),
                        self.n,
                    )),
                    D::FixedSizeList(f, 2) if matches!(f.dtype(), D::UInt32) => Ok(slices::<u32>(
                        name,
                        data,
                        groups.as_any().downcast_ref().unwrap(),
                        self.n,
                    )),
                    D::FixedSizeList(f, 2) if matches!(f.dtype(), D::UInt64) => Ok(slices::<u64>(
                        name,
                        data,
                        groups.as_any().downcast_ref().unwrap(),
                        self.n,
                    )),
                    _ => unreachable!(),
                }
            },
        }
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

fn rolling_product_indices<T: Copy + Into<u64>>(
    values: &[i64],
    validity: Option<BitMask>,
    indices: &[T],
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
                let v = values[(*i).into() as usize];
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
                let v = values[(*i).into() as usize];
                let is_valid = validity.get((*i).into() as usize);

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
