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

use crate::array::{Array, StructArray};
use crate::compute::utils::combine_validities_and;
use crate::datatypes::IdxArr;

pub(super) unsafe fn take_unchecked(array: &StructArray, indices: &IdxArr) -> StructArray {
    let values: Vec<Box<dyn Array>> = array
        .values()
        .iter()
        .map(|a| super::take_unchecked(a.as_ref(), indices))
        .collect();

    let validity = array
        .validity()
        .map(|b| super::bitmap::take_bitmap_nulls_unchecked(b, indices));
    let validity = combine_validities_and(validity.as_ref(), indices.validity());
    StructArray::new(array.data_type().clone(), values, validity)
}
