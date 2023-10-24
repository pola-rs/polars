//! Definition of basic add operations with primitive arrays
use std::ops::Add;

use num_traits::ops::overflowing::OverflowingAdd;
use num_traits::{CheckedAdd, SaturatingAdd, WrappingAdd};

use super::NativeArithmetics;
use crate::array::PrimitiveArray;
use crate::bitmap::Bitmap;
use crate::compute::arithmetics::{
    ArrayAdd, ArrayCheckedAdd, ArrayOverflowingAdd, ArraySaturatingAdd, ArrayWrappingAdd,
};
use crate::compute::arity::{
    binary, binary_checked, binary_with_bitmap, unary, unary_checked, unary_with_bitmap,
};

/// Adds two primitive arrays with the same type.
/// Panics if the sum of one pair of values overflows.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::add;
/// use polars_arrow::array::PrimitiveArray;
///
/// let a = PrimitiveArray::from([None, Some(6), None, Some(6)]);
/// let b = PrimitiveArray::from([Some(5), None, None, Some(6)]);
/// let result = add(&a, &b);
/// let expected = PrimitiveArray::from([None, None, None, Some(12)]);
/// assert_eq!(result, expected)
/// ```
pub fn add<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeArithmetics + Add<Output = T>,
{
    binary(lhs, rhs, lhs.data_type().clone(), |a, b| a + b)
}

/// Wrapping addition of two [`PrimitiveArray`]s.
/// It wraps around at the boundary of the type if the result overflows.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::wrapping_add;
/// use polars_arrow::array::PrimitiveArray;
///
/// let a = PrimitiveArray::from([Some(-100i8), Some(100i8), Some(100i8)]);
/// let b = PrimitiveArray::from([Some(0i8), Some(100i8), Some(0i8)]);
/// let result = wrapping_add(&a, &b);
/// let expected = PrimitiveArray::from([Some(-100i8), Some(-56i8), Some(100i8)]);
/// assert_eq!(result, expected);
/// ```
pub fn wrapping_add<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeArithmetics + WrappingAdd<Output = T>,
{
    let op = move |a: T, b: T| a.wrapping_add(&b);

    binary(lhs, rhs, lhs.data_type().clone(), op)
}

/// Checked addition of two primitive arrays. If the result from the sum
/// overflows, the validity for that index is changed to None
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::checked_add;
/// use polars_arrow::array::PrimitiveArray;
///
/// let a = PrimitiveArray::from([Some(100i8), Some(100i8), Some(100i8)]);
/// let b = PrimitiveArray::from([Some(0i8), Some(100i8), Some(0i8)]);
/// let result = checked_add(&a, &b);
/// let expected = PrimitiveArray::from([Some(100i8), None, Some(100i8)]);
/// assert_eq!(result, expected);
/// ```
pub fn checked_add<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeArithmetics + CheckedAdd<Output = T>,
{
    let op = move |a: T, b: T| a.checked_add(&b);

    binary_checked(lhs, rhs, lhs.data_type().clone(), op)
}

/// Saturating addition of two primitive arrays. If the result from the sum is
/// larger than the possible number for this type, the result for the operation
/// will be the saturated value.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::saturating_add;
/// use polars_arrow::array::PrimitiveArray;
///
/// let a = PrimitiveArray::from([Some(100i8)]);
/// let b = PrimitiveArray::from([Some(100i8)]);
/// let result = saturating_add(&a, &b);
/// let expected = PrimitiveArray::from([Some(127)]);
/// assert_eq!(result, expected);
/// ```
pub fn saturating_add<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeArithmetics + SaturatingAdd<Output = T>,
{
    let op = move |a: T, b: T| a.saturating_add(&b);

    binary(lhs, rhs, lhs.data_type().clone(), op)
}

/// Overflowing addition of two primitive arrays. If the result from the sum is
/// larger than the possible number for this type, the result for the operation
/// will be an array with overflowed values and a  validity array indicating
/// the overflowing elements from the array.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::overflowing_add;
/// use polars_arrow::array::PrimitiveArray;
///
/// let a = PrimitiveArray::from([Some(1i8), Some(100i8)]);
/// let b = PrimitiveArray::from([Some(1i8), Some(100i8)]);
/// let (result, overflow) = overflowing_add(&a, &b);
/// let expected = PrimitiveArray::from([Some(2i8), Some(-56i8)]);
/// assert_eq!(result, expected);
/// ```
pub fn overflowing_add<T>(
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
) -> (PrimitiveArray<T>, Bitmap)
where
    T: NativeArithmetics + OverflowingAdd<Output = T>,
{
    let op = move |a: T, b: T| a.overflowing_add(&b);

    binary_with_bitmap(lhs, rhs, lhs.data_type().clone(), op)
}

// Implementation of ArrayAdd trait for PrimitiveArrays
impl<T> ArrayAdd<PrimitiveArray<T>> for PrimitiveArray<T>
where
    T: NativeArithmetics + Add<Output = T>,
{
    fn add(&self, rhs: &PrimitiveArray<T>) -> Self {
        add(self, rhs)
    }
}

impl<T> ArrayWrappingAdd<PrimitiveArray<T>> for PrimitiveArray<T>
where
    T: NativeArithmetics + WrappingAdd<Output = T>,
{
    fn wrapping_add(&self, rhs: &PrimitiveArray<T>) -> Self {
        wrapping_add(self, rhs)
    }
}

// Implementation of ArrayCheckedAdd trait for PrimitiveArrays
impl<T> ArrayCheckedAdd<PrimitiveArray<T>> for PrimitiveArray<T>
where
    T: NativeArithmetics + CheckedAdd<Output = T>,
{
    fn checked_add(&self, rhs: &PrimitiveArray<T>) -> Self {
        checked_add(self, rhs)
    }
}

// Implementation of ArraySaturatingAdd trait for PrimitiveArrays
impl<T> ArraySaturatingAdd<PrimitiveArray<T>> for PrimitiveArray<T>
where
    T: NativeArithmetics + SaturatingAdd<Output = T>,
{
    fn saturating_add(&self, rhs: &PrimitiveArray<T>) -> Self {
        saturating_add(self, rhs)
    }
}

// Implementation of ArraySaturatingAdd trait for PrimitiveArrays
impl<T> ArrayOverflowingAdd<PrimitiveArray<T>> for PrimitiveArray<T>
where
    T: NativeArithmetics + OverflowingAdd<Output = T>,
{
    fn overflowing_add(&self, rhs: &PrimitiveArray<T>) -> (Self, Bitmap) {
        overflowing_add(self, rhs)
    }
}

/// Adds a scalar T to a primitive array of type T.
/// Panics if the sum of the values overflows.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::add_scalar;
/// use polars_arrow::array::PrimitiveArray;
///
/// let a = PrimitiveArray::from([None, Some(6), None, Some(6)]);
/// let result = add_scalar(&a, &1i32);
/// let expected = PrimitiveArray::from([None, Some(7), None, Some(7)]);
/// assert_eq!(result, expected)
/// ```
pub fn add_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> PrimitiveArray<T>
where
    T: NativeArithmetics + Add<Output = T>,
{
    let rhs = *rhs;
    unary(lhs, |a| a + rhs, lhs.data_type().clone())
}

/// Wrapping addition of a scalar T to a [`PrimitiveArray`] of type T.
/// It do nothing if the result overflows.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::wrapping_add_scalar;
/// use polars_arrow::array::Int8Array;
///
/// let a = Int8Array::from(&[None, Some(100)]);
/// let result = wrapping_add_scalar(&a, &100i8);
/// let expected = Int8Array::from(&[None, Some(-56)]);
/// assert_eq!(result, expected);
/// ```
pub fn wrapping_add_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> PrimitiveArray<T>
where
    T: NativeArithmetics + WrappingAdd<Output = T>,
{
    unary(lhs, |a| a.wrapping_add(rhs), lhs.data_type().clone())
}

/// Checked addition of a scalar T to a primitive array of type T. If the
/// result from the sum overflows then the validity index for that value is
/// changed to None
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::checked_add_scalar;
/// use polars_arrow::array::Int8Array;
///
/// let a = Int8Array::from(&[None, Some(100), None, Some(100)]);
/// let result = checked_add_scalar(&a, &100i8);
/// let expected = Int8Array::from(&[None, None, None, None]);
/// assert_eq!(result, expected);
/// ```
pub fn checked_add_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> PrimitiveArray<T>
where
    T: NativeArithmetics + CheckedAdd<Output = T>,
{
    let rhs = *rhs;
    let op = move |a: T| a.checked_add(&rhs);

    unary_checked(lhs, op, lhs.data_type().clone())
}

/// Saturated addition of a scalar T to a primitive array of type T. If the
/// result from the sum is larger than the possible number for this type, then
/// the result will be saturated
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::saturating_add_scalar;
/// use polars_arrow::array::PrimitiveArray;
///
/// let a = PrimitiveArray::from([Some(100i8)]);
/// let result = saturating_add_scalar(&a, &100i8);
/// let expected = PrimitiveArray::from([Some(127)]);
/// assert_eq!(result, expected);
/// ```
pub fn saturating_add_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> PrimitiveArray<T>
where
    T: NativeArithmetics + SaturatingAdd<Output = T>,
{
    let rhs = *rhs;
    let op = move |a: T| a.saturating_add(&rhs);

    unary(lhs, op, lhs.data_type().clone())
}

/// Overflowing addition of a scalar T to a primitive array of type T. If the
/// result from the sum is larger than the possible number for this type, then
/// the result will be an array with overflowed values and a validity array
/// indicating the overflowing elements from the array
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::overflowing_add_scalar;
/// use polars_arrow::array::PrimitiveArray;
///
/// let a = PrimitiveArray::from([Some(1i8), Some(100i8)]);
/// let (result, overflow) = overflowing_add_scalar(&a, &100i8);
/// let expected = PrimitiveArray::from([Some(101i8), Some(-56i8)]);
/// assert_eq!(result, expected);
/// ```
pub fn overflowing_add_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> (PrimitiveArray<T>, Bitmap)
where
    T: NativeArithmetics + OverflowingAdd<Output = T>,
{
    let rhs = *rhs;
    let op = move |a: T| a.overflowing_add(&rhs);

    unary_with_bitmap(lhs, op, lhs.data_type().clone())
}

// Implementation of ArrayAdd trait for PrimitiveArrays with a scalar
impl<T> ArrayAdd<T> for PrimitiveArray<T>
where
    T: NativeArithmetics + Add<Output = T>,
{
    fn add(&self, rhs: &T) -> Self {
        add_scalar(self, rhs)
    }
}

// Implementation of ArrayCheckedAdd trait for PrimitiveArrays with a scalar
impl<T> ArrayCheckedAdd<T> for PrimitiveArray<T>
where
    T: NativeArithmetics + CheckedAdd<Output = T>,
{
    fn checked_add(&self, rhs: &T) -> Self {
        checked_add_scalar(self, rhs)
    }
}

// Implementation of ArraySaturatingAdd trait for PrimitiveArrays with a scalar
impl<T> ArraySaturatingAdd<T> for PrimitiveArray<T>
where
    T: NativeArithmetics + SaturatingAdd<Output = T>,
{
    fn saturating_add(&self, rhs: &T) -> Self {
        saturating_add_scalar(self, rhs)
    }
}

// Implementation of ArraySaturatingAdd trait for PrimitiveArrays with a scalar
impl<T> ArrayOverflowingAdd<T> for PrimitiveArray<T>
where
    T: NativeArithmetics + OverflowingAdd<Output = T>,
{
    fn overflowing_add(&self, rhs: &T) -> (Self, Bitmap) {
        overflowing_add_scalar(self, rhs)
    }
}
