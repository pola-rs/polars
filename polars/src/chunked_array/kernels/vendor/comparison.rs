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

//! Defines basic comparison kernels for `PrimitiveArrays`.
//!
//! These kernels can leverage SIMD if available on your system.  Currently no runtime
//! detection is provided, you should enable the specific SIMD intrinsics using
//! `RUSTFLAGS="-C target-feature=+avx2"` for example.  See the documentation
//! [here](https://doc.rust-lang.org/stable/core/arch/) for more information.

use std::sync::Arc;

use super::utils::apply_bin_op_to_option_bitmap;
use crate::datatypes::PolarsNumericType;
use arrow::array::*;
use arrow::datatypes::{BooleanType, DataType};
use arrow::error::{ArrowError, Result};

/// Helper function to perform boolean lambda function on values from two arrays, this
/// version does not attempt to use SIMD.
macro_rules! compare_op {
    ($left: expr, $right:expr, $op:expr) => {{
        if $left.len() != $right.len() {
            return Err(ArrowError::ComputeError(
                "Cannot perform comparison operation on arrays of different length".to_string(),
            ));
        }

        let null_bit_buffer = apply_bin_op_to_option_bitmap(
            $left.data().null_bitmap(),
            $right.data().null_bitmap(),
            |a, b| a & b,
        )?;

        let mut result = BooleanBufferBuilder::new($left.len());
        for i in 0..$left.len() {
            result.append($op($left.value(i), $right.value(i)))?;
        }

        let data = ArrayData::new(
            DataType::Boolean,
            $left.len(),
            None,
            null_bit_buffer,
            0,
            vec![result.finish()],
            vec![],
        );
        Ok(PrimitiveArray::<BooleanType>::from(Arc::new(data)))
    }};
}

macro_rules! compare_op_scalar {
    ($left: expr, $right:expr, $op:expr) => {{
        let null_bit_buffer = $left.data().null_buffer().cloned();
        let mut result = BooleanBufferBuilder::new($left.len());
        for i in 0..$left.len() {
            result.append($op($left.value(i), $right))?;
        }

        let data = ArrayData::new(
            DataType::Boolean,
            $left.len(),
            None,
            null_bit_buffer,
            0,
            vec![result.finish()],
            vec![],
        );
        Ok(PrimitiveArray::<BooleanType>::from(Arc::new(data)))
    }};
}

pub fn no_simd_compare_op<T, F>(
    left: &PrimitiveArray<T>,
    right: &PrimitiveArray<T>,
    op: F,
) -> Result<BooleanArray>
where
    T: PolarsNumericType,
    F: Fn(T::Native, T::Native) -> bool,
{
    compare_op!(left, right, op)
}

pub fn no_simd_compare_op_scalar<T, F>(
    left: &PrimitiveArray<T>,
    right: T::Native,
    op: F,
) -> Result<BooleanArray>
where
    T: PolarsNumericType,
    F: Fn(T::Native, T::Native) -> bool,
{
    compare_op_scalar!(left, right, op)
}

pub fn eq_utf8(left: &StringArray, right: &StringArray) -> Result<BooleanArray> {
    compare_op!(left, right, |a, b| a == b)
}

pub fn eq_utf8_scalar(left: &StringArray, right: &str) -> Result<BooleanArray> {
    compare_op_scalar!(left, right, |a, b| a == b)
}

pub fn neq_utf8(left: &StringArray, right: &StringArray) -> Result<BooleanArray> {
    compare_op!(left, right, |a, b| a != b)
}

pub fn neq_utf8_scalar(left: &StringArray, right: &str) -> Result<BooleanArray> {
    compare_op_scalar!(left, right, |a, b| a != b)
}

pub fn lt_utf8(left: &StringArray, right: &StringArray) -> Result<BooleanArray> {
    compare_op!(left, right, |a, b| a < b)
}

pub fn lt_utf8_scalar(left: &StringArray, right: &str) -> Result<BooleanArray> {
    compare_op_scalar!(left, right, |a, b| a < b)
}

pub fn lt_eq_utf8(left: &StringArray, right: &StringArray) -> Result<BooleanArray> {
    compare_op!(left, right, |a, b| a <= b)
}

pub fn lt_eq_utf8_scalar(left: &StringArray, right: &str) -> Result<BooleanArray> {
    compare_op_scalar!(left, right, |a, b| a <= b)
}

pub fn gt_utf8(left: &StringArray, right: &StringArray) -> Result<BooleanArray> {
    compare_op!(left, right, |a, b| a > b)
}

pub fn gt_utf8_scalar(left: &StringArray, right: &str) -> Result<BooleanArray> {
    compare_op_scalar!(left, right, |a, b| a > b)
}

pub fn gt_eq_utf8(left: &StringArray, right: &StringArray) -> Result<BooleanArray> {
    compare_op!(left, right, |a, b| a >= b)
}

pub fn gt_eq_utf8_scalar(left: &StringArray, right: &str) -> Result<BooleanArray> {
    compare_op_scalar!(left, right, |a, b| a >= b)
}

/// Helper function to perform boolean lambda function on values from two arrays using
/// SIMD.
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
fn simd_compare_op<T, F>(
    left: &PrimitiveArray<T>,
    right: &PrimitiveArray<T>,
    op: F,
) -> Result<BooleanArray>
where
    T: PolarsNumericType,
    F: Fn(T::Simd, T::Simd) -> T::SimdMask,
{
    use arrow::buffer::MutableBuffer;
    use std::io::Write;
    use std::mem;

    let len = left.len();
    if len != right.len() {
        return Err(ArrowError::ComputeError(
            "Cannot perform comparison operation on arrays of different length".to_string(),
        ));
    }

    let null_bit_buffer = apply_bin_op_to_option_bitmap(
        left.data().null_bitmap(),
        right.data().null_bitmap(),
        |a, b| a & b,
    )?;

    let lanes = T::lanes();
    let mut result = MutableBuffer::new(left.len() * mem::size_of::<bool>());

    let rem = len % lanes;

    for i in (0..len - rem).step_by(lanes) {
        let simd_left = T::load(left.value_slice(i, lanes));
        let simd_right = T::load(right.value_slice(i, lanes));
        let simd_result = op(simd_left, simd_right);
        T::bitmask(&simd_result, |b| {
            result.write(b).unwrap();
        });
    }

    if rem > 0 {
        let simd_left = T::load(left.value_slice(len - rem, lanes));
        let simd_right = T::load(right.value_slice(len - rem, lanes));
        let simd_result = op(simd_left, simd_right);
        let rem_buffer_size = (rem as f32 / 8f32).ceil() as usize;
        T::bitmask(&simd_result, |b| {
            result.write(&b[0..rem_buffer_size]).unwrap();
        });
    }

    let data = ArrayData::new(
        DataType::Boolean,
        left.len(),
        None,
        null_bit_buffer,
        0,
        vec![result.freeze()],
        vec![],
    );
    Ok(PrimitiveArray::<BooleanType>::from(Arc::new(data)))
}

/// Helper function to perform boolean lambda function on values from an array and a scalar value using
/// SIMD.
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
fn simd_compare_op_scalar<T, F>(
    left: &PrimitiveArray<T>,
    right: T::Native,
    op: F,
) -> Result<BooleanArray>
where
    T: PolarsNumericType,
    F: Fn(T::Simd, T::Simd) -> T::SimdMask,
{
    use arrow::buffer::MutableBuffer;
    use std::io::Write;
    use std::mem;

    let len = left.len();
    let null_bit_buffer = left.data().null_buffer().cloned();
    let lanes = T::lanes();
    let mut result = MutableBuffer::new(left.len() * mem::size_of::<bool>());
    let simd_right = T::init(right);

    let rem = len % lanes;

    for i in (0..len - rem).step_by(lanes) {
        let simd_left = T::load(left.value_slice(i, lanes));
        let simd_result = op(simd_left, simd_right);
        T::bitmask(&simd_result, |b| {
            result.write(b).unwrap();
        });
    }

    if rem > 0 {
        let simd_left = T::load(left.value_slice(len - rem, lanes));
        let simd_result = op(simd_left, simd_right);
        let rem_buffer_size = (rem as f32 / 8f32).ceil() as usize;
        T::bitmask(&simd_result, |b| {
            result.write(&b[0..rem_buffer_size]).unwrap();
        });
    }

    let data = ArrayData::new(
        DataType::Boolean,
        left.len(),
        None,
        null_bit_buffer,
        0,
        vec![result.freeze()],
        vec![],
    );
    Ok(PrimitiveArray::<BooleanType>::from(Arc::new(data)))
}

/// Perform `left == right` operation on two arrays.
pub fn eq<T>(left: &PrimitiveArray<T>, right: &PrimitiveArray<T>) -> Result<BooleanArray>
where
    T: PolarsNumericType,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    return simd_compare_op(left, right, T::eq);

    #[cfg(any(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        not(feature = "simd")
    ))]
    compare_op!(left, right, |a, b| a == b)
}

/// Perform `left == right` operation on an array and a scalar value.
pub fn eq_scalar<T>(left: &PrimitiveArray<T>, right: T::Native) -> Result<BooleanArray>
where
    T: PolarsNumericType,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    return simd_compare_op_scalar(left, right, T::eq);

    #[cfg(any(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        not(feature = "simd")
    ))]
    compare_op_scalar!(left, right, |a, b| a == b)
}

/// Perform `left != right` operation on two arrays.
pub fn neq<T>(left: &PrimitiveArray<T>, right: &PrimitiveArray<T>) -> Result<BooleanArray>
where
    T: PolarsNumericType,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    return simd_compare_op(left, right, T::ne);

    #[cfg(any(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        not(feature = "simd")
    ))]
    compare_op!(left, right, |a, b| a != b)
}

/// Perform `left != right` operation on an array and a scalar value.
pub fn neq_scalar<T>(left: &PrimitiveArray<T>, right: T::Native) -> Result<BooleanArray>
where
    T: PolarsNumericType,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    return simd_compare_op_scalar(left, right, T::ne);

    #[cfg(any(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        not(feature = "simd")
    ))]
    compare_op_scalar!(left, right, |a, b| a != b)
}

/// Perform `left < right` operation on two arrays. Null values are less than non-null
/// values.
pub fn lt<T>(left: &PrimitiveArray<T>, right: &PrimitiveArray<T>) -> Result<BooleanArray>
where
    T: PolarsNumericType,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    return simd_compare_op(left, right, T::lt);

    #[cfg(any(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        not(feature = "simd")
    ))]
    compare_op!(left, right, |a, b| a < b)
}

/// Perform `left < right` operation on an array and a scalar value.
/// Null values are less than non-null values.
pub fn lt_scalar<T>(left: &PrimitiveArray<T>, right: T::Native) -> Result<BooleanArray>
where
    T: PolarsNumericType,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    return simd_compare_op_scalar(left, right, T::lt);

    #[cfg(any(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        not(feature = "simd")
    ))]
    compare_op_scalar!(left, right, |a, b| a < b)
}

/// Perform `left <= right` operation on two arrays. Null values are less than non-null
/// values.
pub fn lt_eq<T>(left: &PrimitiveArray<T>, right: &PrimitiveArray<T>) -> Result<BooleanArray>
where
    T: PolarsNumericType,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    return simd_compare_op(left, right, T::le);

    #[cfg(any(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        not(feature = "simd")
    ))]
    compare_op!(left, right, |a, b| a <= b)
}

/// Perform `left <= right` operation on an array and a scalar value.
/// Null values are less than non-null values.
pub fn lt_eq_scalar<T>(left: &PrimitiveArray<T>, right: T::Native) -> Result<BooleanArray>
where
    T: PolarsNumericType,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    return simd_compare_op_scalar(left, right, T::le);

    #[cfg(any(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        not(feature = "simd")
    ))]
    compare_op_scalar!(left, right, |a, b| a <= b)
}

/// Perform `left > right` operation on two arrays. Non-null values are greater than null
/// values.
pub fn gt<T>(left: &PrimitiveArray<T>, right: &PrimitiveArray<T>) -> Result<BooleanArray>
where
    T: PolarsNumericType,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    return simd_compare_op(left, right, T::gt);

    #[cfg(any(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        not(feature = "simd")
    ))]
    compare_op!(left, right, |a, b| a > b)
}

/// Perform `left > right` operation on an array and a scalar value.
/// Non-null values are greater than null values.
pub fn gt_scalar<T>(left: &PrimitiveArray<T>, right: T::Native) -> Result<BooleanArray>
where
    T: PolarsNumericType,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    return simd_compare_op_scalar(left, right, T::gt);

    #[cfg(any(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        not(feature = "simd")
    ))]
    compare_op_scalar!(left, right, |a, b| a > b)
}

/// Perform `left >= right` operation on two arrays. Non-null values are greater than null
/// values.
pub fn gt_eq<T>(left: &PrimitiveArray<T>, right: &PrimitiveArray<T>) -> Result<BooleanArray>
where
    T: PolarsNumericType,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    return simd_compare_op(left, right, T::ge);

    #[cfg(any(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        not(feature = "simd")
    ))]
    compare_op!(left, right, |a, b| a >= b)
}

/// Perform `left >= right` operation on an array and a scalar value.
/// Non-null values are greater than null values.
pub fn gt_eq_scalar<T>(left: &PrimitiveArray<T>, right: T::Native) -> Result<BooleanArray>
where
    T: PolarsNumericType,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    return simd_compare_op_scalar(left, right, T::ge);

    #[cfg(any(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        not(feature = "simd")
    ))]
    compare_op_scalar!(left, right, |a, b| a >= b)
}
