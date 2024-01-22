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

use crate::array::{new_empty_array, Array, NullArray, PrimitiveArray};
use crate::types::Index;

mod binary;
mod boolean;
mod dict;
mod fixed_size_list;
mod generic_binary;
mod list;
mod primitive;
mod structure;
mod utf8;

use crate::{match_integer_type, with_match_primitive_type};

/// Returns a new [`Array`] with only indices at `indices`. Null indices are taken as nulls.
/// The returned array has a length equal to `indices.len()`.
/// # Safety
/// Doesn't do bound checks
pub unsafe fn take_unchecked<O: Index>(
    values: &dyn Array,
    indices: &PrimitiveArray<O>,
) -> Box<dyn Array> {
    if indices.len() == 0 {
        return new_empty_array(values.data_type().clone());
    }

    use crate::datatypes::PhysicalType::*;
    match values.data_type().to_physical_type() {
        Null => Box::new(NullArray::new(values.data_type().clone(), indices.len())),
        Boolean => {
            let values = values.as_any().downcast_ref().unwrap();
            Box::new(boolean::take_unchecked::<O>(values, indices))
        },
        Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {
            let values = values.as_any().downcast_ref().unwrap();
            Box::new(primitive::take_unchecked::<$T, _>(&values, indices))
        }),
        LargeUtf8 => {
            let values = values.as_any().downcast_ref().unwrap();
            Box::new(utf8::take_unchecked::<i64, _>(values, indices))
        },
        LargeBinary => {
            let values = values.as_any().downcast_ref().unwrap();
            Box::new(binary::take_unchecked::<i64, _>(values, indices))
        },
        Dictionary(key_type) => {
            match_integer_type!(key_type, |$T| {
                let values = values.as_any().downcast_ref().unwrap();
                Box::new(dict::take_unchecked::<$T, _>(&values, indices))
            })
        },
        Struct => {
            let array = values.as_any().downcast_ref().unwrap();
            structure::take_unchecked::<_>(array, indices).boxed()
        },
        LargeList => {
            let array = values.as_any().downcast_ref().unwrap();
            Box::new(list::take_unchecked::<i64, O>(array, indices))
        },
        FixedSizeList => {
            let array = values.as_any().downcast_ref().unwrap();
            Box::new(fixed_size_list::take_unchecked::<O>(array, indices))
        },
        t => unimplemented!("Take not supported for data type {:?}", t),
    }
}
