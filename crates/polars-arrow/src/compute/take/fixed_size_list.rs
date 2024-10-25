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

use polars_utils::itertools::Itertools;

use super::Index;
use crate::array::growable::{Growable, GrowableFixedSizeList};
use crate::array::{Array, ArrayRef, FixedSizeListArray, PrimitiveArray};
use crate::bitmap::MutableBitmap;
use crate::datatypes::reshape::{Dimension, ReshapeDimension};
use crate::datatypes::{ArrowDataType, PhysicalType};
use crate::legacy::prelude::FromData;
use crate::with_match_primitive_type;

pub(super) unsafe fn take_unchecked_slow<O: Index>(
    values: &FixedSizeListArray,
    indices: &PrimitiveArray<O>,
) -> FixedSizeListArray {
    let take_len = std::cmp::min(values.len(), 1);
    let mut capacity = 0;
    let arrays = indices
        .values()
        .iter()
        .map(|index| {
            let index = index.to_usize();
            let slice = values.clone().sliced_unchecked(index, take_len);
            capacity += slice.len();
            slice
        })
        .collect::<Vec<FixedSizeListArray>>();

    let arrays = arrays.iter().collect();

    if let Some(validity) = indices.validity() {
        let mut growable: GrowableFixedSizeList =
            GrowableFixedSizeList::new(arrays, true, capacity);

        for index in 0..indices.len() {
            if validity.get_bit_unchecked(index) {
                growable.extend(index, 0, 1);
            } else {
                growable.extend_validity(1)
            }
        }

        growable.into()
    } else {
        let mut growable: GrowableFixedSizeList =
            GrowableFixedSizeList::new(arrays, false, capacity);
        for index in 0..indices.len() {
            growable.extend(index, 0, 1);
        }

        growable.into()
    }
}

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

unsafe fn from_buffer(mut buf: ManuallyDrop<Vec<u8>>, dtype: &ArrowDataType) -> ArrayRef {
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

            PrimitiveArray::from_data_default(buf.into(), None).boxed()

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

fn no_inner_validities(values: &ArrayRef) -> bool {
    if let Some(arr) = values.as_any().downcast_ref::<FixedSizeListArray>() {
        arr.validity().is_none() && no_inner_validities(arr.values())
    } else {
        values.validity().is_none()
    }
}

/// `take` implementation for FixedSizeListArrays
pub(super) unsafe fn take_unchecked<O: Index>(
    values: &FixedSizeListArray,
    indices: &PrimitiveArray<O>,
) -> ArrayRef {
    let (stride, leaf_type) = get_stride_and_leaf_type(values.dtype(), 1);
    if leaf_type.to_physical_type().is_primitive() && no_inner_validities(values.values()) {
        let leaves = get_leaves(values);

        let (leaves_buf, leave_size) = get_buffer_and_size(leaves);
        let bytes_per_element = leave_size * stride;

        let n_idx = indices.len();
        let total_bytes = bytes_per_element * n_idx;

        let mut buf = ManuallyDrop::new(aligned_vec(leaves.dtype(), total_bytes));
        let dst = buf.spare_capacity_mut();

        let mut count = 0;
        let validity = if indices.null_count() == 0 {
            for i in indices.values().iter() {
                let i = i.to_usize();

                std::ptr::copy_nonoverlapping(
                    leaves_buf.as_ptr().add(i * bytes_per_element),
                    dst.as_mut_ptr().add(count * bytes_per_element) as *mut _,
                    bytes_per_element,
                );
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
                } else {
                    new_validity.set_unchecked(count, false);
                    std::ptr::write_bytes(
                        dst.as_mut_ptr().add(count * bytes_per_element) as *mut _,
                        0,
                        bytes_per_element,
                    );
                }

                count += 1;
            }
            Some(new_validity.freeze())
        };
        assert_eq!(count * bytes_per_element, total_bytes);

        buf.set_len(total_bytes);

        let leaves = from_buffer(buf, leaves.dtype());
        let mut shape = values.get_dims();
        shape[0] = Dimension::new(indices.len() as _);
        let shape = shape
            .into_iter()
            .map(ReshapeDimension::Specified)
            .collect_vec();

        FixedSizeListArray::from_shape(leaves.clone(), &shape)
            .unwrap()
            .with_validity(validity)
    } else {
        take_unchecked_slow(values, indices).boxed()
    }
}
