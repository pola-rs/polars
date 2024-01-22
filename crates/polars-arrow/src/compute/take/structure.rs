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
use crate::array::{Array, PrimitiveArray, StructArray};
use crate::bitmap::{Bitmap, MutableBitmap};

#[inline]
unsafe fn take_validity<I: Index>(
    validity: Option<&Bitmap>,
    indices: &PrimitiveArray<I>,
) -> Option<Bitmap> {
    let indices_validity = indices.validity();
    match (validity, indices_validity) {
        (None, _) => indices_validity.cloned(),
        (Some(validity), None) => {
            let iter = indices.values().iter().map(|index| {
                let index = index.to_usize();
                validity.get_bit_unchecked(index)
            });
            MutableBitmap::from_trusted_len_iter(iter).into()
        },
        (Some(validity), _) => {
            let iter = indices.iter().map(|x| match x {
                Some(index) => {
                    let index = index.to_usize();
                    validity.get_bit_unchecked(index)
                },
                None => false,
            });
            MutableBitmap::from_trusted_len_iter(iter).into()
        },
    }
}

pub(super) unsafe fn take_unchecked<I: Index>(
    array: &StructArray,
    indices: &PrimitiveArray<I>,
) -> StructArray {
    let values: Vec<Box<dyn Array>> = array
        .values()
        .iter()
        .map(|a| super::take_unchecked(a.as_ref(), indices))
        .collect();
    let validity = take_validity(array.validity(), indices);
    StructArray::new(array.data_type().clone(), values, validity)
}
