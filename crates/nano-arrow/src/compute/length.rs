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

//! Defines kernel for length of composite arrays

use crate::{
    array::*,
    datatypes::DataType,
    error::{Error, Result},
    offset::Offset,
    types::NativeType,
};

fn unary_offsets_string<O, F>(array: &Utf8Array<O>, op: F) -> PrimitiveArray<O>
where
    O: Offset + NativeType,
    F: Fn(O) -> O,
{
    let values = array
        .offsets()
        .buffer()
        .windows(2)
        .map(|offset| op(offset[1] - offset[0]))
        .collect::<Vec<_>>();

    let data_type = if O::IS_LARGE {
        DataType::Int64
    } else {
        DataType::Int32
    };

    PrimitiveArray::<O>::new(data_type, values.into(), array.validity().cloned())
}

/// Returns an array of integers with the number of bytes on each string of the array.
pub fn length(array: &dyn Array) -> Result<Box<dyn Array>> {
    match array.data_type() {
        DataType::Utf8 => {
            let array = array.as_any().downcast_ref::<Utf8Array<i32>>().unwrap();
            Ok(Box::new(unary_offsets_string::<i32, _>(array, |x| x)))
        }
        DataType::LargeUtf8 => {
            let array = array.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
            Ok(Box::new(unary_offsets_string::<i64, _>(array, |x| x)))
        }
        _ => Err(Error::InvalidArgumentError(format!(
            "length not supported for {:?}",
            array.data_type()
        ))),
    }
}

/// Checks if an array of type `datatype` can perform length operation
///
/// # Examples
/// ```
/// use arrow2::compute::length::can_length;
/// use arrow2::datatypes::{DataType};
///
/// let data_type = DataType::Utf8;
/// assert_eq!(can_length(&data_type), true);
///
/// let data_type = DataType::Int8;
/// assert_eq!(can_length(&data_type), false);
/// ```
pub fn can_length(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8 | DataType::LargeUtf8)
}
