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

use super::Index;
use crate::array::growable::{Growable, GrowableFixedSizeList};
use crate::array::{FixedSizeListArray, PrimitiveArray};

/// `take` implementation for FixedSizeListArrays
pub(super) unsafe fn take_unchecked<O: Index>(
    values: &FixedSizeListArray,
    indices: &PrimitiveArray<O>,
) -> FixedSizeListArray {
    let mut capacity = 0;
    let arrays = indices
        .values()
        .iter()
        .map(|index| {
            let index = index.to_usize();
            let slice = values.clone().sliced(index, 1);
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
