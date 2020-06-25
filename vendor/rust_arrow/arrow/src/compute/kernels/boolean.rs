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

//! Defines boolean kernels on Arrow `BooleanArray`'s, e.g. `AND`, `OR` and `NOT`.
//!
//! These kernels can leverage SIMD if available on your system.  Currently no runtime
//! detection is provided, you should enable the specific SIMD intrinsics using
//! `RUSTFLAGS="-C target-feature=+avx2"` for example.  See the documentation
//! [here](https://doc.rust-lang.org/stable/core/arch/) for more information.

use std::sync::Arc;

use crate::array::{Array, ArrayData, BooleanArray};
use crate::buffer::Buffer;
use crate::compute::util::apply_bin_op_to_option_bitmap;
use crate::datatypes::DataType;
use crate::error::{ArrowError, Result};

/// Helper function to implement binary kernels
fn binary_boolean_kernel<F>(
    left: &BooleanArray,
    right: &BooleanArray,
    op: F,
) -> Result<BooleanArray>
where
    F: Fn(&Buffer, &Buffer) -> Result<Buffer>,
{
    if left.offset() != right.offset() {
        return Err(ArrowError::ComputeError(
            "Cannot apply Bitwise binary op when arrays have different offsets."
                .to_string(),
        ));
    }

    let left_data = left.data();
    let right_data = right.data();
    let null_bit_buffer = apply_bin_op_to_option_bitmap(
        left_data.null_bitmap(),
        right_data.null_bitmap(),
        |a, b| a & b,
    )?;
    let values = op(&left_data.buffers()[0], &right_data.buffers()[0])?;
    let data = ArrayData::new(
        DataType::Boolean,
        left.len(),
        None,
        null_bit_buffer,
        left.offset(),
        vec![values],
        vec![],
    );
    Ok(BooleanArray::from(Arc::new(data)))
}

/// Performs `AND` operation on two arrays. If either left or right value is null then the
/// result is also null.
pub fn and(left: &BooleanArray, right: &BooleanArray) -> Result<BooleanArray> {
    binary_boolean_kernel(&left, &right, |a, b| a & b)
}

/// Performs `OR` operation on two arrays. If either left or right value is null then the
/// result is also null.
pub fn or(left: &BooleanArray, right: &BooleanArray) -> Result<BooleanArray> {
    binary_boolean_kernel(&left, &right, |a, b| a | b)
}

/// Performs unary `NOT` operation on an arrays. If value is null then the result is also
/// null.
pub fn not(left: &BooleanArray) -> Result<BooleanArray> {
    let data = left.data();
    let null_bit_buffer = data.null_bitmap().as_ref().map(|b| b.bits.clone());

    let values = !&data.buffers()[0];
    let data = ArrayData::new(
        DataType::Boolean,
        left.len(),
        None,
        null_bit_buffer,
        left.offset(),
        vec![values],
        vec![],
    );
    Ok(BooleanArray::from(Arc::new(data)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_array_and() {
        let a = BooleanArray::from(vec![false, false, true, true]);
        let b = BooleanArray::from(vec![false, true, false, true]);
        let c = and(&a, &b).unwrap();
        assert_eq!(false, c.value(0));
        assert_eq!(false, c.value(1));
        assert_eq!(false, c.value(2));
        assert_eq!(true, c.value(3));
    }

    #[test]
    fn test_bool_array_or() {
        let a = BooleanArray::from(vec![false, false, true, true]);
        let b = BooleanArray::from(vec![false, true, false, true]);
        let c = or(&a, &b).unwrap();
        assert_eq!(false, c.value(0));
        assert_eq!(true, c.value(1));
        assert_eq!(true, c.value(2));
        assert_eq!(true, c.value(3));
    }

    #[test]
    fn test_bool_array_or_nulls() {
        let a = BooleanArray::from(vec![None, Some(false), None, Some(false)]);
        let b = BooleanArray::from(vec![None, None, Some(false), Some(false)]);
        let c = or(&a, &b).unwrap();
        assert_eq!(true, c.is_null(0));
        assert_eq!(true, c.is_null(1));
        assert_eq!(true, c.is_null(2));
        assert_eq!(false, c.is_null(3));
    }

    #[test]
    fn test_bool_array_not() {
        let a = BooleanArray::from(vec![false, false, true, true]);
        let c = not(&a).unwrap();
        assert_eq!(true, c.value(0));
        assert_eq!(true, c.value(1));
        assert_eq!(false, c.value(2));
        assert_eq!(false, c.value(3));
    }

    #[test]
    fn test_bool_array_and_nulls() {
        let a = BooleanArray::from(vec![None, Some(false), None, Some(false)]);
        let b = BooleanArray::from(vec![None, None, Some(false), Some(false)]);
        let c = and(&a, &b).unwrap();
        assert_eq!(true, c.is_null(0));
        assert_eq!(true, c.is_null(1));
        assert_eq!(true, c.is_null(2));
        assert_eq!(false, c.is_null(3));
    }
}
