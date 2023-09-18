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

use crate::array::{Array, PrimitiveArray, Utf8Array};
use crate::offset::Offset;

use super::generic_binary::*;
use super::Index;

/// `take` implementation for utf8 arrays
pub fn take<O: Offset, I: Index>(
    values: &Utf8Array<O>,
    indices: &PrimitiveArray<I>,
) -> Utf8Array<O> {
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
    unsafe { Utf8Array::<O>::new_unchecked(data_type, offsets, values, validity) }
}

#[cfg(test)]
mod tests {
    use crate::array::Int32Array;

    use super::*;

    fn _all_cases<O: Offset>() -> Vec<(Int32Array, Utf8Array<O>, Utf8Array<O>)> {
        vec![
            (
                Int32Array::from(&[Some(1), Some(0)]),
                Utf8Array::<O>::from(vec![Some("one"), Some("two")]),
                Utf8Array::<O>::from(vec![Some("two"), Some("one")]),
            ),
            (
                Int32Array::from(&[Some(1), None]),
                Utf8Array::<O>::from(vec![Some("one"), Some("two")]),
                Utf8Array::<O>::from(vec![Some("two"), None]),
            ),
            (
                Int32Array::from(&[Some(1), Some(0)]),
                Utf8Array::<O>::from(vec![None, Some("two")]),
                Utf8Array::<O>::from(vec![Some("two"), None]),
            ),
            (
                Int32Array::from(&[Some(1), None, Some(0)]),
                Utf8Array::<O>::from(vec![None, Some("two")]),
                Utf8Array::<O>::from(vec![Some("two"), None, None]),
            ),
        ]
    }

    #[test]
    fn all_cases() {
        let cases = _all_cases::<i32>();
        for (indices, input, expected) in cases {
            let output = take(&input, &indices);
            assert_eq!(expected, output);
        }
        let cases = _all_cases::<i64>();
        for (indices, input, expected) in cases {
            let output = take(&input, &indices);
            assert_eq!(expected, output);
        }
    }
}
