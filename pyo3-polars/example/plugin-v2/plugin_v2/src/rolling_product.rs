use std::borrow::Cow;
use std::collections::VecDeque;

use arrow::array::PrimitiveArray;
use arrow::bitmap::bitmask::BitMask;
use arrow::bitmap::BitmapBuilder;
use polars::error::{polars_ensure, polars_err, PolarsResult};
use polars::prelude::{
    ArrowDataType, ChunkedArray, ChunkedBuilder, DataType, Field, Int64Type,
    PrimitiveChunkedBuilder, Schema, SchemaExt,
};
use polars::series::{IntoSeries, Series};
use pyo3_polars::export::polars_ffi::version_1::{
    GroupPositions, PolarsPlugin, SliceGroup, SliceGroups,
};
use pyo3_polars::v1::PolarsPluginExprInfo;
use pyo3_polars::{polars_plugin_expr_info, v1};
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

impl v1::scan::PolarsScanPlugin for RollingProduct {
    type State = RollingProductState;

    fn to_field(&self, fields: &Schema) -> PolarsResult<Field> {
        assert_eq!(fields.len(), 1);
        let field = fields.iter_fields().next().unwrap();
        polars_ensure!(
            field.dtype() == &DataType::Int64,
            InvalidOperation: "rolling_product can only be performed on i64"
        );
        Ok(field)
    }

    fn new_state(&self, _fields: &Schema) -> PolarsResult<Self::State> {
        Ok(Self::State {
            product: 1,
            values: VecDeque::with_capacity(self.n),
        })
    }

    fn reset(&self, state: &mut Self::State) -> PolarsResult<()> {
        state.product = 1;
        state.values.clear();
        Ok(())
    }

    fn step(&self, state: &mut Self::State, inputs: &[Series]) -> PolarsResult<Series> {
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
        Ok(builder.finish().into_series())
    }

    // fn evaluate_on_groups<'a>(
    //     &self,
    //     inputs: &[(Series, &'a GroupPositions)],
    // ) -> PolarsResult<(Series, Cow<'a, GroupPositions>)> {
    //     assert_eq!(inputs.len(), 1);
    //     let (data, groups) = &inputs[0];
    //
    //     if let GroupPositions::SharedAcrossGroups { .. } = groups {
    //         let mut state = self.new_state(&Schema::from_iter([data.field().into_owned()]))?;
    //         let data = self.step(&mut state, &[data.clone()])?.unwrap();
    //         return Ok((data, Cow::Borrowed(groups)));
    //     };
    //
    //     if let GroupPositions::ScalarPerGroup = groups {
    //         return Ok((data.clone(), Cow::Borrowed(groups)));
    //     };
    //
    //     let data = data.i64()?;
    //     let name = data.name().clone();
    //     let data = data.rechunk();
    //     let data = data.downcast_as_array();
    //
    //     let num_groups = match groups {
    //         GroupPositions::SharedAcrossGroups { .. } | GroupPositions::ScalarPerGroup => {
    //             unreachable!()
    //         },
    //         GroupPositions::Slice(slice_groups) => slice_groups.len(),
    //         GroupPositions::Index(index_groups) => index_groups.ends.len(),
    //     };
    //     let num_values = match groups {
    //         GroupPositions::SharedAcrossGroups { .. } | GroupPositions::ScalarPerGroup => {
    //             unreachable!()
    //         },
    //         GroupPositions::Slice(slice_groups) => slice_groups.lengths().sum(),
    //         GroupPositions::Index(index_groups) => {
    //             index_groups.ends.last().copied().unwrap_or(0) as usize
    //         },
    //     };
    //
    //     let mut out = Vec::with_capacity(num_values as usize);
    //     let mut out_bitmap = BitmapBuilder::with_capacity(num_values as usize);
    //     let mut out_slices = Vec::with_capacity(num_groups);
    //
    //     let data_values = data.values().as_slice();
    //     let data_validity = data.validity().map(BitMask::from_bitmap);
    //     let mut past = VecDeque::with_capacity(self.n);
    //
    //     match groups {
    //         GroupPositions::ScalarPerGroup | GroupPositions::SharedAcrossGroups { .. } => {
    //             unreachable!()
    //         },
    //         GroupPositions::Slice(groups) => {
    //             for slice in &groups.0 {
    //                 past.clear();
    //                 let offset = out.len() as u64;
    //                 rolling_product_values(
    //                     &data_values[slice.offset as usize..][..slice.length as usize],
    //                     data_validity,
    //                     &mut 1,
    //                     &mut past,
    //                     &mut out,
    //                     &mut out_bitmap,
    //                     self.n,
    //                 );
    //                 let length = out.len() as u64 - offset;
    //                 out_slices.push(SliceGroup { offset, length });
    //             }
    //         },
    //         GroupPositions::Index(indices) => {
    //             for i in indices.iter() {
    //                 past.clear();
    //                 let offset = out.len();
    //                 rolling_product_indices(
    //                     data_values,
    //                     data_validity,
    //                     i,
    //                     &mut past,
    //                     &mut out,
    //                     &mut out_bitmap,
    //                     self.n,
    //                 );
    //                 let length = out.len() - offset;
    //                 out_slices.push(SliceGroup {
    //                     offset: offset as u64,
    //                     length: length as u64,
    //                 });
    //             }
    //         },
    //     }
    //
    //     let out = PrimitiveArray::new(
    //         ArrowDataType::Int64,
    //         out.into(),
    //         out_bitmap.into_opt_validity(),
    //     );
    //     let data = unsafe {
    //         ChunkedArray::<Int64Type>::from_chunks_and_dtype(
    //             name,
    //             vec![out.boxed()],
    //             DataType::Int64,
    //         )
    //     }
    //     .into_series();
    //
    //     let groups = GroupPositions::Slice(SliceGroups(out_slices.into()));
    //     Ok((data, Cow::Owned(groups)))
    // }
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
    polars_plugin_expr_info!(
        "rolling_product",
        v1::scan::Plugin(RollingProduct { n }),
        v1::scan::Plugin<RollingProduct>
    )
}
