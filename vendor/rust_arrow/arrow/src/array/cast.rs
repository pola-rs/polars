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

//! Defines helper functions for force Array type downcast

use crate::array::*;
use crate::datatypes::*;

/// Force downcast ArrayRef to PrimitiveArray<T>
pub fn as_primitive_array<T>(arr: &ArrayRef) -> &PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
{
    arr.as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .expect("Unable to downcast to primitive array")
}

macro_rules! array_downcast_fn {
    ($name: ident, $arrty: ty, $arrty_str:expr) => {
        #[doc = "Force downcast ArrayRef to "]
        #[doc = $arrty_str]
        pub fn $name(arr: &ArrayRef) -> &$arrty {
            arr.as_any().downcast_ref::<$arrty>().expect(concat!(
                "Unable to downcast to typed array through ",
                stringify!($name)
            ))
        }
    };

    // use recursive macro to generate dynamic doc string for a given array type
    ($name: ident, $arrty: ty) => {
        array_downcast_fn!($name, $arrty, stringify!($arrty));
    };
}

array_downcast_fn!(as_string_array, StringArray);
array_downcast_fn!(as_boolean_array, BooleanArray);
array_downcast_fn!(as_null_array, NullArray);
