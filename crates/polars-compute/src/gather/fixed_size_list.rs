// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::mem::ManuallyDrop;

use polars_arrow::array::{Array, ArrayRef, FixedSizeListArray, PrimitiveArray, StaticArray};
use polars_arrow::bitmap::{Bitmap, BitmapBuilder, MutableBitmap};
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::datatypes::reshape::{Dimension, ReshapeDimension};
use polars_arrow::datatypes::{ArrowDataType, IdxArr, PhysicalType};
use polars_arrow::legacy::prelude::FromData;
use polars_arrow::with_match_primitive_type;
use polars_utils::itertools::Itertools;

use super::Index;
use crate::gather::bitmap::{take_bitmap_nulls_unchecked, take_bitmap_unchecked};

fn get_stride_and_leaf_type(dtype: &ArrowDataType, size: usize) -> (usize, &ArrowDataType) {
    if let ArrowDataType::FixedSizeList(inner, size_inner) = dtype {
        get_stride_and_leaf_type(inner.dtype(), *size_inner * size)
    } else {
        (size, dtype)
    }
}

fn get_leaves(array: &FixedSizeListArray) -> &dyn Array {
    if let Some(array) = array.values().as_any().downcast_ref::<FixedSizeListArray>() {
        get_leaves(array)
    } else {
        &**array.values()
    }
}

fn get_buffer_and_size(array: &dyn Array) -> (&[u8], usize) {
    match array.dtype().to_physical_type() {
        PhysicalType::Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {

            let arr = array.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
            let values = arr.values();
            (bytemuck::cast_slice(values), size_of::<$T>())

        }),
        _ => {
            unimplemented!()
        },
    }
}

unsafe fn from_buffer(
    mut buf: ManuallyDrop<Vec<u8>>,
    dtype: &ArrowDataType,
    validity: Option<Bitmap>,
) -> ArrayRef {
    match dtype.to_physical_type() {
        PhysicalType::Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {
            let ptr = buf.as_mut_ptr();
            let len_units = buf.len();
            let cap_units = buf.capacity();

            let buf = Vec::from_raw_parts(
                ptr as *mut $T,
                len_units / size_of::<$T>(),
                cap_units / size_of::<$T>(),
            );

            PrimitiveArray::from_data_default(buf.into(), validity).boxed()

        }),
        _ => {
            unimplemented!()
        },
    }
}

unsafe fn aligned_vec(dt: &ArrowDataType, n_bytes: usize) -> Vec<u8> {
    match dt.to_physical_type() {
        PhysicalType::Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {

        let n_units = (n_bytes / size_of::<$T>()) + 1;

        let mut aligned: Vec<$T> = Vec::with_capacity(n_units);

        let ptr = aligned.as_mut_ptr();
        let len_units = aligned.len();
        let cap_units = aligned.capacity();

        std::mem::forget(aligned);

        Vec::from_raw_parts(
            ptr as *mut u8,
            len_units * size_of::<$T>(),
            cap_units * size_of::<$T>(),
        )

            }),
        _ => {
            unimplemented!()
        },
    }
}

/// Whether no level between `arr` and the leaf it nests carries a validity mask.
///
/// The leaf's own mask is gathered alongside its values, in the same runs of `stride` slots; a
/// mask on a level in between would have to be gathered in a shape of its own, so an array that
/// holds one takes the generic path.
fn no_validity_above_leaf(arr: &dyn Array) -> bool {
    arr.as_any()
        .downcast_ref::<FixedSizeListArray>()
        .is_none_or(|arr| arr.validity().is_none() && no_validity_above_leaf(arr.values().as_ref()))
}

/// `take` implementation for FixedSizeListArrays
pub(super) unsafe fn take_unchecked(values: &FixedSizeListArray, indices: &IdxArr) -> ArrayRef {
    let (stride, leaf_type) = get_stride_and_leaf_type(values.dtype(), 1);
    if leaf_type.to_physical_type().is_primitive()
        && no_validity_above_leaf(values.values().as_ref())
    {
        let leaves = get_leaves(values);

        let (leaves_buf, leave_size) = get_buffer_and_size(leaves);
        let bytes_per_element = leave_size * stride;

        // The leaf reads `stride` of its slots per element, so its mask is gathered in runs of
        // that many bits, next to the run of bytes the values are copied from.
        let leaf_validity = leaves.validity();
        let mut new_leaf_validity =
            leaf_validity.map(|_| BitmapBuilder::with_capacity(indices.len() * stride));

        let n_idx = indices.len();
        let total_bytes = bytes_per_element * n_idx;

        let mut buf = ManuallyDrop::new(aligned_vec(leaves.dtype(), total_bytes));
        let dst = buf.spare_capacity_mut();

        let take_leaf_mask = |builder: &mut BitmapBuilder, i: usize| {
            builder.subslice_extend_from_bitmap(leaf_validity.unwrap(), i * stride, stride);
        };

        let mut count = 0;
        let outer_validity = if indices.null_count() == 0 {
            for i in indices.values().iter() {
                let i = i.to_usize();

                std::ptr::copy_nonoverlapping(
                    leaves_buf.as_ptr().add(i * bytes_per_element),
                    dst.as_mut_ptr().add(count * bytes_per_element) as *mut _,
                    bytes_per_element,
                );
                if let Some(builder) = new_leaf_validity.as_mut() {
                    take_leaf_mask(builder, i);
                }
                count += 1;
            }
            None
        } else {
            let mut new_validity = MutableBitmap::with_capacity(indices.len());
            new_validity.extend_constant(indices.len(), true);
            for i in indices.iter() {
                if let Some(i) = i {
                    let i = i.to_usize();
                    std::ptr::copy_nonoverlapping(
                        leaves_buf.as_ptr().add(i * bytes_per_element),
                        dst.as_mut_ptr().add(count * bytes_per_element) as *mut _,
                        bytes_per_element,
                    );
                    if let Some(builder) = new_leaf_validity.as_mut() {
                        take_leaf_mask(builder, i);
                    }
                } else {
                    new_validity.set_unchecked(count, false);
                    std::ptr::write_bytes(
                        dst.as_mut_ptr().add(count * bytes_per_element) as *mut _,
                        0,
                        bytes_per_element,
                    );
                    if let Some(builder) = new_leaf_validity.as_mut() {
                        builder.extend_constant(stride, false);
                    }
                }

                count += 1;
            }
            Some(new_validity.freeze())
        };

        assert_eq!(count * bytes_per_element, total_bytes);
        buf.set_len(total_bytes);

        let outer_validity = combine_validities_and(
            outer_validity.as_ref(),
            values
                .validity()
                .map(|x| {
                    if indices.has_nulls() {
                        take_bitmap_nulls_unchecked(x, indices)
                    } else {
                        take_bitmap_unchecked(x, indices.as_slice().unwrap())
                    }
                })
                .as_ref(),
        );

        let leaves = from_buffer(
            buf,
            leaves.dtype(),
            new_leaf_validity.and_then(BitmapBuilder::into_opt_validity),
        );
        let mut shape = values.get_dims();
        shape[0] = Dimension::new(indices.len() as _);
        let shape = shape
            .into_iter()
            .map(ReshapeDimension::Specified)
            .collect_vec();

        FixedSizeListArray::from_shape(leaves.clone(), &shape)
            .unwrap()
            .with_validity(outer_validity)
    } else {
        super::take_unchecked_impl_generic(values, indices, &FixedSizeListArray::new_null).boxed()
    }
}

#[cfg(test)]
mod tests {
    use polars_arrow::array::StaticArray;
    use polars_arrow::bitmap::Bitmap;
    use polars_arrow::datatypes::ArrowDataType;

    /// Test gather for FixedSizeListArray with outer validity but no inner validities.
    #[test]
    fn test_arr_gather_nulls_outer_validity_19482() {
        use polars_arrow::array::{FixedSizeListArray, Int64Array, PrimitiveArray};
        use polars_arrow::bitmap::Bitmap;
        use polars_arrow::datatypes::reshape::{Dimension, ReshapeDimension};
        use polars_utils::IdxSize;

        use super::take_unchecked;

        unsafe {
            let dyn_arr = FixedSizeListArray::from_shape(
                Box::new(Int64Array::from_slice([1, 2, 3, 4])),
                &[
                    ReshapeDimension::Specified(Dimension::new(2)),
                    ReshapeDimension::Specified(Dimension::new(2)),
                ],
            )
            .unwrap()
            .with_validity(Some(Bitmap::from_iter([true, false]))); // FixedSizeListArray[[1, 2], None]

            let arr = dyn_arr
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap();

            assert_eq!(
                [arr.validity().is_some(), arr.values().validity().is_some()],
                [true, false]
            );

            assert_eq!(
                take_unchecked(arr, &PrimitiveArray::<IdxSize>::from_slice([0, 1])),
                dyn_arr
            )
        }
    }

    /// A leaf mask is gathered in runs of `stride` bits, alongside the values it covers -- and a
    /// null index reads as a null element whose slots are null too.
    #[test]
    fn gather_leaf_validity_in_runs() {
        use polars_arrow::array::{FixedSizeListArray, Int64Array, PrimitiveArray};
        use polars_arrow::datatypes::reshape::{Dimension, ReshapeDimension};
        use polars_utils::IdxSize;

        use super::take_unchecked;

        unsafe {
            // FixedSizeListArray[[1, None], [None, 4], [5, 6]]
            let dyn_arr = FixedSizeListArray::from_shape(
                Box::new(Int64Array::from([
                    Some(1),
                    None,
                    None,
                    Some(4),
                    Some(5),
                    Some(6),
                ])),
                &[
                    ReshapeDimension::Specified(Dimension::new(3)),
                    ReshapeDimension::Specified(Dimension::new(2)),
                ],
            )
            .unwrap();

            let arr = dyn_arr
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap();

            let taken = take_unchecked(arr, &PrimitiveArray::<IdxSize>::from_slice([2, 0, 1]));
            let expected = FixedSizeListArray::from_shape(
                Box::new(Int64Array::from([
                    Some(5),
                    Some(6),
                    Some(1),
                    None,
                    None,
                    Some(4),
                ])),
                &[
                    ReshapeDimension::Specified(Dimension::new(3)),
                    ReshapeDimension::Specified(Dimension::new(2)),
                ],
            )
            .unwrap();
            assert_eq!(taken, expected);

            let taken = take_unchecked(arr, &PrimitiveArray::<IdxSize>::from([Some(1), None]));
            let expected = FixedSizeListArray::from_shape(
                Box::new(Int64Array::from([None, Some(4), None, None])),
                &[
                    ReshapeDimension::Specified(Dimension::new(2)),
                    ReshapeDimension::Specified(Dimension::new(2)),
                ],
            )
            .unwrap()
            .with_validity(Some(Bitmap::from_iter([true, false])));
            assert_eq!(taken, expected);
        }
    }

    #[test]
    fn test_arr_gather_nulls_inner_validity() {
        use polars_arrow::array::{FixedSizeListArray, Int64Array, PrimitiveArray};
        use polars_arrow::datatypes::reshape::{Dimension, ReshapeDimension};
        use polars_utils::IdxSize;

        use super::take_unchecked;

        unsafe {
            let dyn_arr = FixedSizeListArray::from_shape(
                Box::new(Int64Array::full_null(4, ArrowDataType::Int64)),
                &[
                    ReshapeDimension::Specified(Dimension::new(2)),
                    ReshapeDimension::Specified(Dimension::new(2)),
                ],
            )
            .unwrap(); // FixedSizeListArray[[None, None], [None, None]]

            let arr = dyn_arr
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap();

            assert_eq!(
                [arr.validity().is_some(), arr.values().validity().is_some()],
                [false, true]
            );

            assert_eq!(
                take_unchecked(arr, &PrimitiveArray::<IdxSize>::from_slice([0, 1])),
                dyn_arr
            )
        }
    }
}
