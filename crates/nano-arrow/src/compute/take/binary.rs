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

use crate::array::{Array, BinaryArray, PrimitiveArray};
use crate::offset::Offset;

use super::generic_binary::*;
use super::Index;

/// `take` implementation for utf8 arrays
pub fn take<O: Offset, I: Index>(
    values: &BinaryArray<O>,
    indices: &PrimitiveArray<I>,
) -> BinaryArray<O> {
    let data_type = values.data_type().clone();
    let indices_has_validity = indices.null_count() > 0;
    let values_has_validity = values.null_count() > 0;

    let (offsets, values, validity) = match (values_has_validity, indices_has_validity) {
        (false, false) => {
            take_no_validity::<O, I>(values.offsets(), values.values(), indices.values())
        }
        (true, false) => take_values_validity(values, indices.values()),
        (false, true) => take_indices_validity(values.offsets(), values.values(), indices),
        (true, true) => take_values_indices_validity(values, indices),
    };
    BinaryArray::<O>::new(data_type, offsets, values, validity)
}
