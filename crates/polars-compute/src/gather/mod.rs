#![allow(unsafe_op_in_unsafe_fn)]
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

//! Defines take kernel for [`Array`]

use arrow::array::{
    self, Array, ArrayCollectIterExt, ArrayFromIterDtype, BinaryViewArray, NullArray, StaticArray,
    Utf8ViewArray, new_empty_array,
};
use arrow::datatypes::{ArrowDataType, IdxArr};
use arrow::types::Index;

pub mod binary;
pub mod binview;
pub mod bitmap;
pub mod boolean;
pub mod fixed_size_list;
pub mod generic_binary;
pub mod list;
pub mod primitive;
pub mod structure;
pub mod sublist;

use arrow::with_match_primitive_type_full;

/// Returns a new [`Array`] with only indices at `indices`. Null indices are taken as nulls.
/// The returned array has a length equal to `indices.len()`.
/// # Safety
/// Doesn't do bound checks
pub unsafe fn take_unchecked(values: &dyn Array, indices: &IdxArr) -> Box<dyn Array> {
    if indices.len() == 0 {
        return new_empty_array(values.dtype().clone());
    }

    use arrow::datatypes::PhysicalType::*;
    match values.dtype().to_physical_type() {
        Null => Box::new(NullArray::new(values.dtype().clone(), indices.len())),
        Boolean => {
            let values = values.as_any().downcast_ref().unwrap();
            Box::new(boolean::take_unchecked(values, indices))
        },
        Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            let values = values.as_any().downcast_ref().unwrap();
            Box::new(primitive::take_primitive_unchecked::<$T>(&values, indices))
        }),
        LargeBinary => {
            let values = values.as_any().downcast_ref().unwrap();
            Box::new(binary::take_unchecked::<i64, _>(values, indices))
        },
        Struct => {
            let array = values.as_any().downcast_ref().unwrap();
            structure::take_unchecked(array, indices).boxed()
        },
        LargeList => {
            let array = values.as_any().downcast_ref().unwrap();
            Box::new(list::take_unchecked::<i64>(array, indices))
        },
        FixedSizeList => {
            let array = values.as_any().downcast_ref().unwrap();
            fixed_size_list::take_unchecked(array, indices)
        },
        BinaryView => {
            let array: &BinaryViewArray = values.as_any().downcast_ref().unwrap();
            binview::take_binview_unchecked(array, indices).boxed()
        },
        Utf8View => {
            let array: &Utf8ViewArray = values.as_any().downcast_ref().unwrap();
            binview::take_binview_unchecked(array, indices).boxed()
        },
        t => unimplemented!("Take not supported for data type {:?}", t),
    }
}

/// Naive default implementation
unsafe fn take_unchecked_impl_generic<T>(
    values: &T,
    indices: &IdxArr,
    new_null_func: &dyn Fn(ArrowDataType, usize) -> T,
) -> T
where
    T: StaticArray + ArrayFromIterDtype<std::option::Option<Box<dyn array::Array>>>,
{
    if values.null_count() == values.len() || indices.null_count() == indices.len() {
        return new_null_func(values.dtype().clone(), indices.len());
    }

    match (indices.has_nulls(), values.has_nulls()) {
        (true, true) => {
            let values_validity = values.validity().unwrap();

            indices
                .iter()
                .map(|i| {
                    if let Some(i) = i {
                        let i = *i as usize;
                        if values_validity.get_bit_unchecked(i) {
                            return Some(values.value_unchecked(i));
                        }
                    }
                    None
                })
                .collect_arr_trusted_with_dtype(values.dtype().clone())
        },
        (true, false) => indices
            .iter()
            .map(|i| {
                if let Some(i) = i {
                    let i = *i as usize;
                    return Some(values.value_unchecked(i));
                }
                None
            })
            .collect_arr_trusted_with_dtype(values.dtype().clone()),
        (false, true) => {
            let values_validity = values.validity().unwrap();

            indices
                .values_iter()
                .map(|i| {
                    let i = *i as usize;
                    if values_validity.get_bit_unchecked(i) {
                        return Some(values.value_unchecked(i));
                    }
                    None
                })
                .collect_arr_trusted_with_dtype(values.dtype().clone())
        },
        (false, false) => indices
            .values_iter()
            .map(|i| {
                let i = *i as usize;
                Some(values.value_unchecked(i))
            })
            .collect_arr_trusted_with_dtype(values.dtype().clone()),
    }
}
