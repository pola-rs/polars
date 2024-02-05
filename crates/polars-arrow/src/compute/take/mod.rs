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

use crate::array::{new_empty_array, Array, NullArray, Utf8ViewArray};
use crate::compute::take::binview::take_binview_unchecked;
use crate::datatypes::IdxArr;
use crate::types::Index;

mod binary;
mod binview;
mod bitmap;
mod boolean;
mod fixed_size_list;
mod generic_binary;
mod list;
mod primitive;
mod structure;

use crate::with_match_primitive_type_full;

/// Returns a new [`Array`] with only indices at `indices`. Null indices are taken as nulls.
/// The returned array has a length equal to `indices.len()`.
/// # Safety
/// Doesn't do bound checks
pub unsafe fn take_unchecked(values: &dyn Array, indices: &IdxArr) -> Box<dyn Array> {
    if indices.len() == 0 {
        return new_empty_array(values.data_type().clone());
    }

    use crate::datatypes::PhysicalType::*;
    match values.data_type().to_physical_type() {
        Null => Box::new(NullArray::new(values.data_type().clone(), indices.len())),
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
            Box::new(fixed_size_list::take_unchecked(array, indices))
        },
        BinaryView => {
            take_binview_unchecked(values.as_any().downcast_ref().unwrap(), indices).boxed()
        },
        Utf8View => {
            let arr: &Utf8ViewArray = values.as_any().downcast_ref().unwrap();
            take_binview_unchecked(&arr.to_binview(), indices)
                .to_utf8view_unchecked()
                .boxed()
        },
        t => unimplemented!("Take not supported for data type {:?}", t),
    }
}
