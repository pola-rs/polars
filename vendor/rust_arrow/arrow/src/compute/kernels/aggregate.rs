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

//! Defines aggregations over Arrow arrays.

use std::ops::Add;

use crate::array::{Array, PrimitiveArray};
use crate::datatypes::ArrowNumericType;

/// Returns the minimum value in the array, according to the natural order.
pub fn min<T>(array: &PrimitiveArray<T>) -> Option<T::Native>
where
    T: ArrowNumericType,
{
    min_max_helper(array, |a, b| a < b)
}

/// Returns the maximum value in the array, according to the natural order.
pub fn max<T>(array: &PrimitiveArray<T>) -> Option<T::Native>
where
    T: ArrowNumericType,
{
    min_max_helper(array, |a, b| a > b)
}

/// Helper function to perform min/max lambda function on values from a numeric array.
fn min_max_helper<T, F>(array: &PrimitiveArray<T>, cmp: F) -> Option<T::Native>
where
    T: ArrowNumericType,
    F: Fn(T::Native, T::Native) -> bool,
{
    let mut n: Option<T::Native> = None;
    let data = array.data();
    for i in 0..data.len() {
        if data.is_null(i) {
            continue;
        }
        let m = array.value(i);
        match n {
            None => n = Some(m),
            Some(nn) => {
                if cmp(m, nn) {
                    n = Some(m)
                }
            }
        }
    }
    n
}

/// Returns the sum of values in the array.
///
/// Returns `None` if the array is empty or only contains null values.
pub fn sum<T>(array: &PrimitiveArray<T>) -> Option<T::Native>
where
    T: ArrowNumericType,
    T::Native: Add<Output = T::Native>,
{
    let null_count = array.null_count();

    if null_count == array.len() {
        None
    } else if null_count == 0 {
        // optimized path for arrays without null values
        let mut n: T::Native = T::default_value();
        let data = array.data();
        let m = array.value_slice(0, data.len());
        for item in m.iter().take(data.len()) {
            n = n + *item;
        }
        Some(n)
    } else {
        let mut n: T::Native = T::default_value();
        let data = array.data();
        let m = array.value_slice(0, data.len());
        for (i, item) in m.iter().enumerate() {
            if data.is_valid(i) {
                n = n + *item;
            }
        }
        Some(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::*;

    #[test]
    fn test_primitive_array_sum() {
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        assert_eq!(15, sum(&a).unwrap());
    }

    #[test]
    fn test_primitive_array_float_sum() {
        let a = Float64Array::from(vec![1.1, 2.2, 3.3, 4.4, 5.5]);
        assert_eq!(16.5, sum(&a).unwrap());
    }

    #[test]
    fn test_primitive_array_sum_with_nulls() {
        let a = Int32Array::from(vec![None, Some(2), Some(3), None, Some(5)]);
        assert_eq!(10, sum(&a).unwrap());
    }

    #[test]
    fn test_primitive_array_sum_all_nulls() {
        let a = Int32Array::from(vec![None, None, None]);
        assert_eq!(None, sum(&a));
    }

    #[test]
    fn test_buffer_array_min_max() {
        let a = Int32Array::from(vec![5, 6, 7, 8, 9]);
        assert_eq!(5, min(&a).unwrap());
        assert_eq!(9, max(&a).unwrap());
    }

    #[test]
    fn test_buffer_array_min_max_with_nulls() {
        let a = Int32Array::from(vec![Some(5), None, None, Some(8), Some(9)]);
        assert_eq!(5, min(&a).unwrap());
        assert_eq!(9, max(&a).unwrap());
    }
}
