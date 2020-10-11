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

//! Defines basic arithmetic kernels for `PrimitiveArrays`.
//!
//! These kernels can leverage SIMD if available on your system.  Currently no runtime
//! detection is provided, you should enable the specific SIMD intrinsics using
//! `RUSTFLAGS="-C target-feature=+avx2"` for example.  See the documentation
//! [here](https://doc.rust-lang.org/stable/core/arch/) for more information.

use num::{One, Zero};
#[cfg(feature = "simd")]
use std::mem;
use std::ops::{Add, Div, Mul, Sub};
#[cfg(feature = "simd")]
use std::slice::from_raw_parts_mut;
#[cfg(feature = "simd")]
use std::sync::Arc;

use super::utils::apply_bin_op_to_option_bitmap;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
use super::utils::simd_load_set_invalid;
use crate::chunked_array::builder::{aligned_vec_to_primitive_array, AlignedVec};
use crate::datatypes::PolarsNumericType;
use arrow::array::*;
#[cfg(feature = "simd")]
use arrow::bitmap::Bitmap;
#[cfg(feature = "simd")]
use arrow::buffer::MutableBuffer;
use arrow::error::{ArrowError, Result};

/// Helper function to perform math lambda function on values from two arrays. If either
/// left or right value is null then the output value is also null, so `1 + null` is
/// `null`.
pub fn math_op<T, F>(
    left: &PrimitiveArray<T>,
    right: &PrimitiveArray<T>,
    op: F,
) -> Result<PrimitiveArray<T>>
where
    T: PolarsNumericType,
    F: Fn(T::Native, T::Native) -> Result<T::Native>,
{
    if left.len() != right.len() {
        return Err(ArrowError::ComputeError(
            "Cannot perform math operation on arrays of different length".to_string(),
        ));
    }

    let null_bit_buffer = apply_bin_op_to_option_bitmap(
        left.data().null_bitmap(),
        right.data().null_bitmap(),
        |a, b| a & b,
    )?;

    let mut values = AlignedVec::with_capacity_aligned(left.len());
    for i in 0..left.len() {
        values.push(op(left.value(i), right.value(i))?);
    }
    let arr = aligned_vec_to_primitive_array::<T>(values, null_bit_buffer, None);
    Ok(arr)
}

/// SIMD vectorized version of `math_op` above.
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
fn simd_math_op<T, F>(
    left: &PrimitiveArray<T>,
    right: &PrimitiveArray<T>,
    op: F,
) -> Result<PrimitiveArray<T>>
where
    T: PolarsNumericType,
    T::Simd: Add<Output = T::Simd>
        + Sub<Output = T::Simd>
        + Mul<Output = T::Simd>
        + Div<Output = T::Simd>,
    F: Fn(T::Simd, T::Simd) -> T::Simd,
{
    if left.len() != right.len() {
        return Err(ArrowError::ComputeError(
            "Cannot perform math operation on arrays of different length".to_string(),
        ));
    }

    let null_bit_buffer = apply_bin_op_to_option_bitmap(
        left.data().null_bitmap(),
        right.data().null_bitmap(),
        |a, b| a & b,
    )?;

    let lanes = T::lanes();
    let buffer_size = left.len() * mem::size_of::<T::Native>();
    let mut result = MutableBuffer::new(buffer_size).with_bitset(buffer_size, false);

    for i in (0..left.len()).step_by(lanes) {
        let simd_left = T::load(left.value_slice(i, lanes));
        let simd_right = T::load(right.value_slice(i, lanes));
        let simd_result = T::bin_op(simd_left, simd_right, &op);

        let result_slice: &mut [T::Native] = unsafe {
            from_raw_parts_mut(
                (result.data_mut().as_mut_ptr() as *mut T::Native).add(i),
                lanes,
            )
        };
        T::write(simd_result, result_slice);
    }

    let data = ArrayData::new(
        T::get_data_type(),
        left.len(),
        None,
        null_bit_buffer,
        left.offset(),
        vec![result.freeze()],
        vec![],
    );
    Ok(PrimitiveArray::<T>::from(Arc::new(data)))
}

/// SIMD vectorized version of `divide`, the divide kernel needs it's own implementation as there
/// is a need to handle situations where a divide by `0` occurs.  This is complicated by `NULL`
/// slots and padding.
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
fn simd_divide<T>(left: &PrimitiveArray<T>, right: &PrimitiveArray<T>) -> Result<PrimitiveArray<T>>
where
    T: PolarsNumericType,
    T::Native: One + Zero,
{
    if left.len() != right.len() {
        return Err(ArrowError::ComputeError(
            "Cannot perform math operation on arrays of different length".to_string(),
        ));
    }

    // Create the combined `Bitmap`
    let null_bit_buffer = apply_bin_op_to_option_bitmap(
        left.data().null_bitmap(),
        right.data().null_bitmap(),
        |a, b| a & b,
    )?;
    let bitmap = null_bit_buffer.map(Bitmap::from);

    let lanes = T::lanes();
    let buffer_size = left.len() * mem::size_of::<T::Native>();
    let mut result = MutableBuffer::new(buffer_size).with_bitset(buffer_size, false);

    for i in (0..left.len()).step_by(lanes) {
        let right_no_invalid_zeros =
            unsafe { simd_load_set_invalid(right, &bitmap, i, lanes, T::Native::one()) };
        let is_zero = T::eq(T::init(T::Native::zero()), right_no_invalid_zeros);
        if T::mask_any(is_zero) {
            return Err(ArrowError::DivideByZero);
        }
        let right_no_invalid_zeros =
            unsafe { simd_load_set_invalid(right, &bitmap, i, lanes, T::Native::one()) };
        let simd_left = T::load(left.value_slice(i, lanes));
        let simd_result = T::bin_op(simd_left, right_no_invalid_zeros, |a, b| a / b);

        let result_slice: &mut [T::Native] = unsafe {
            from_raw_parts_mut(
                (result.data_mut().as_mut_ptr() as *mut T::Native).add(i),
                lanes,
            )
        };
        T::write(simd_result, result_slice);
    }

    let null_bit_buffer = bitmap.map(|b| b.into_buffer());

    let data = ArrayData::new(
        T::get_data_type(),
        left.len(),
        None,
        null_bit_buffer,
        left.offset(),
        vec![result.freeze()],
        vec![],
    );
    Ok(PrimitiveArray::<T>::from(Arc::new(data)))
}

/// Perform `left + right` operation on two arrays. If either left or right value is null
/// then the result is also null.
pub fn add<T>(left: &PrimitiveArray<T>, right: &PrimitiveArray<T>) -> Result<PrimitiveArray<T>>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + Zero,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    return simd_math_op(&left, &right, |a, b| a + b);

    #[allow(unreachable_code)]
    math_op(left, right, |a, b| Ok(a + b))
}

/// Perform `left - right` operation on two arrays. If either left or right value is null
/// then the result is also null.
pub fn subtract<T>(left: &PrimitiveArray<T>, right: &PrimitiveArray<T>) -> Result<PrimitiveArray<T>>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + Zero,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    return simd_math_op(&left, &right, |a, b| a - b);

    #[allow(unreachable_code)]
    math_op(left, right, |a, b| Ok(a - b))
}

/// Perform `left * right` operation on two arrays. If either left or right value is null
/// then the result is also null.
pub fn multiply<T>(left: &PrimitiveArray<T>, right: &PrimitiveArray<T>) -> Result<PrimitiveArray<T>>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + Zero,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    return simd_math_op(&left, &right, |a, b| a * b);

    #[allow(unreachable_code)]
    math_op(left, right, |a, b| Ok(a * b))
}

/// Perform `left / right` operation on two arrays. If either left or right value is null
/// then the result is also null. If any right hand value is zero then the result of this
/// operation will be `Err(ArrowError::DivideByZero)`.
pub fn divide<T>(left: &PrimitiveArray<T>, right: &PrimitiveArray<T>) -> Result<PrimitiveArray<T>>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + Zero
        + One,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    return simd_divide(&left, &right);

    #[allow(unreachable_code)]
    math_op(left, right, |a, b| {
        if b.is_zero() {
            Err(ArrowError::DivideByZero)
        } else {
            Ok(a / b)
        }
    })
}
