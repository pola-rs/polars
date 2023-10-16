//! Definition of basic sub operations with primitive arrays
use std::ops::Sub;

use num_traits::ops::overflowing::OverflowingSub;
use num_traits::{CheckedSub, SaturatingSub, WrappingSub};

use super::NativeArithmetics;
use crate::array::PrimitiveArray;
use crate::bitmap::Bitmap;
use crate::compute::arithmetics::{
    ArrayCheckedSub, ArrayOverflowingSub, ArraySaturatingSub, ArraySub, ArrayWrappingSub,
};
use crate::compute::arity::{
    binary, binary_checked, binary_with_bitmap, unary, unary_checked, unary_with_bitmap,
};

/// Subtracts two primitive arrays with the same type.
/// Panics if the subtraction of one pair of values overflows.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::sub;
/// use polars_arrow::array::Int32Array;
///
/// let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
/// let b = Int32Array::from(&[Some(5), None, None, Some(6)]);
/// let result = sub(&a, &b);
/// let expected = Int32Array::from(&[None, None, None, Some(0)]);
/// assert_eq!(result, expected)
/// ```
pub fn sub<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeArithmetics + Sub<Output = T>,
{
    binary(lhs, rhs, lhs.data_type().clone(), |a, b| a - b)
}

/// Wrapping subtraction of two [`PrimitiveArray`]s.
///  It wraps around at the boundary of the type if the result overflows.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::wrapping_sub;
/// use polars_arrow::array::PrimitiveArray;
///
/// let a = PrimitiveArray::from([Some(-100i8), Some(-100i8), Some(100i8)]);
/// let b = PrimitiveArray::from([Some(0i8), Some(100i8), Some(0i8)]);
/// let result = wrapping_sub(&a, &b);
/// let expected = PrimitiveArray::from([Some(-100i8), Some(56i8), Some(100i8)]);
/// assert_eq!(result, expected);
/// ```
pub fn wrapping_sub<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeArithmetics + WrappingSub<Output = T>,
{
    let op = move |a: T, b: T| a.wrapping_sub(&b);

    binary(lhs, rhs, lhs.data_type().clone(), op)
}

/// Checked subtraction of two primitive arrays. If the result from the
/// subtraction overflow, the validity for that index is changed
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::checked_sub;
/// use polars_arrow::array::Int8Array;
///
/// let a = Int8Array::from(&[Some(100i8), Some(-100i8), Some(100i8)]);
/// let b = Int8Array::from(&[Some(1i8), Some(100i8), Some(0i8)]);
/// let result = checked_sub(&a, &b);
/// let expected = Int8Array::from(&[Some(99i8), None, Some(100i8)]);
/// assert_eq!(result, expected);
/// ```
pub fn checked_sub<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeArithmetics + CheckedSub<Output = T>,
{
    let op = move |a: T, b: T| a.checked_sub(&b);

    binary_checked(lhs, rhs, lhs.data_type().clone(), op)
}

/// Saturating subtraction of two primitive arrays. If the result from the sub
/// is smaller than the possible number for this type, the result for the
/// operation will be the saturated value.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::saturating_sub;
/// use polars_arrow::array::Int8Array;
///
/// let a = Int8Array::from(&[Some(-100i8)]);
/// let b = Int8Array::from(&[Some(100i8)]);
/// let result = saturating_sub(&a, &b);
/// let expected = Int8Array::from(&[Some(-128)]);
/// assert_eq!(result, expected);
/// ```
pub fn saturating_sub<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeArithmetics + SaturatingSub<Output = T>,
{
    let op = move |a: T, b: T| a.saturating_sub(&b);

    binary(lhs, rhs, lhs.data_type().clone(), op)
}

/// Overflowing subtraction of two primitive arrays. If the result from the sub
/// is smaller than the possible number for this type, the result for the
/// operation will be an array with overflowed values and a validity array
/// indicating the overflowing elements from the array.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::overflowing_sub;
/// use polars_arrow::array::Int8Array;
///
/// let a = Int8Array::from(&[Some(1i8), Some(-100i8)]);
/// let b = Int8Array::from(&[Some(1i8), Some(100i8)]);
/// let (result, overflow) = overflowing_sub(&a, &b);
/// let expected = Int8Array::from(&[Some(0i8), Some(56i8)]);
/// assert_eq!(result, expected);
/// ```
pub fn overflowing_sub<T>(
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
) -> (PrimitiveArray<T>, Bitmap)
where
    T: NativeArithmetics + OverflowingSub<Output = T>,
{
    let op = move |a: T, b: T| a.overflowing_sub(&b);

    binary_with_bitmap(lhs, rhs, lhs.data_type().clone(), op)
}

// Implementation of ArraySub trait for PrimitiveArrays
impl<T> ArraySub<PrimitiveArray<T>> for PrimitiveArray<T>
where
    T: NativeArithmetics + Sub<Output = T>,
{
    fn sub(&self, rhs: &PrimitiveArray<T>) -> Self {
        sub(self, rhs)
    }
}

impl<T> ArrayWrappingSub<PrimitiveArray<T>> for PrimitiveArray<T>
where
    T: NativeArithmetics + WrappingSub<Output = T>,
{
    fn wrapping_sub(&self, rhs: &PrimitiveArray<T>) -> Self {
        wrapping_sub(self, rhs)
    }
}

// Implementation of ArrayCheckedSub trait for PrimitiveArrays
impl<T> ArrayCheckedSub<PrimitiveArray<T>> for PrimitiveArray<T>
where
    T: NativeArithmetics + CheckedSub<Output = T>,
{
    fn checked_sub(&self, rhs: &PrimitiveArray<T>) -> Self {
        checked_sub(self, rhs)
    }
}

// Implementation of ArraySaturatingSub trait for PrimitiveArrays
impl<T> ArraySaturatingSub<PrimitiveArray<T>> for PrimitiveArray<T>
where
    T: NativeArithmetics + SaturatingSub<Output = T>,
{
    fn saturating_sub(&self, rhs: &PrimitiveArray<T>) -> Self {
        saturating_sub(self, rhs)
    }
}

// Implementation of ArraySaturatingSub trait for PrimitiveArrays
impl<T> ArrayOverflowingSub<PrimitiveArray<T>> for PrimitiveArray<T>
where
    T: NativeArithmetics + OverflowingSub<Output = T>,
{
    fn overflowing_sub(&self, rhs: &PrimitiveArray<T>) -> (Self, Bitmap) {
        overflowing_sub(self, rhs)
    }
}

/// Subtract a scalar T to a primitive array of type T.
/// Panics if the subtraction of the values overflows.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::sub_scalar;
/// use polars_arrow::array::Int32Array;
///
/// let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
/// let result = sub_scalar(&a, &1i32);
/// let expected = Int32Array::from(&[None, Some(5), None, Some(5)]);
/// assert_eq!(result, expected)
/// ```
pub fn sub_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> PrimitiveArray<T>
where
    T: NativeArithmetics + Sub<Output = T>,
{
    let rhs = *rhs;
    unary(lhs, |a| a - rhs, lhs.data_type().clone())
}

/// Wrapping subtraction of a scalar T to a [`PrimitiveArray`] of type T.
/// It do nothing if the result overflows.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::wrapping_sub_scalar;
/// use polars_arrow::array::Int8Array;
///
/// let a = Int8Array::from(&[None, Some(-100)]);
/// let result = wrapping_sub_scalar(&a, &100i8);
/// let expected = Int8Array::from(&[None, Some(56)]);
/// assert_eq!(result, expected);
/// ```
pub fn wrapping_sub_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> PrimitiveArray<T>
where
    T: NativeArithmetics + WrappingSub<Output = T>,
{
    unary(lhs, |a| a.wrapping_sub(rhs), lhs.data_type().clone())
}

/// Checked subtraction of a scalar T to a primitive array of type T. If the
/// result from the subtraction overflows, then the validity for that index
/// is changed to None
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::checked_sub_scalar;
/// use polars_arrow::array::Int8Array;
///
/// let a = Int8Array::from(&[None, Some(-100), None, Some(-100)]);
/// let result = checked_sub_scalar(&a, &100i8);
/// let expected = Int8Array::from(&[None, None, None, None]);
/// assert_eq!(result, expected);
/// ```
pub fn checked_sub_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> PrimitiveArray<T>
where
    T: NativeArithmetics + CheckedSub<Output = T>,
{
    let rhs = *rhs;
    let op = move |a: T| a.checked_sub(&rhs);

    unary_checked(lhs, op, lhs.data_type().clone())
}

/// Saturated subtraction of a scalar T to a primitive array of type T. If the
/// result from the sub is smaller than the possible number for this type, then
/// the result will be saturated
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::saturating_sub_scalar;
/// use polars_arrow::array::Int8Array;
///
/// let a = Int8Array::from(&[Some(-100i8)]);
/// let result = saturating_sub_scalar(&a, &100i8);
/// let expected = Int8Array::from(&[Some(-128i8)]);
/// assert_eq!(result, expected);
/// ```
pub fn saturating_sub_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> PrimitiveArray<T>
where
    T: NativeArithmetics + SaturatingSub<Output = T>,
{
    let rhs = *rhs;
    let op = move |a: T| a.saturating_sub(&rhs);

    unary(lhs, op, lhs.data_type().clone())
}

/// Overflowing subtraction of a scalar T to a primitive array of type T. If
/// the result from the sub is smaller than the possible number for this type,
/// then the result will be an array with overflowed values and a validity
/// array indicating the overflowing elements from the array
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::overflowing_sub_scalar;
/// use polars_arrow::array::Int8Array;
///
/// let a = Int8Array::from(&[Some(1i8), Some(-100i8)]);
/// let (result, overflow) = overflowing_sub_scalar(&a, &100i8);
/// let expected = Int8Array::from(&[Some(-99i8), Some(56i8)]);
/// assert_eq!(result, expected);
/// ```
pub fn overflowing_sub_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> (PrimitiveArray<T>, Bitmap)
where
    T: NativeArithmetics + OverflowingSub<Output = T>,
{
    let rhs = *rhs;
    let op = move |a: T| a.overflowing_sub(&rhs);

    unary_with_bitmap(lhs, op, lhs.data_type().clone())
}

// Implementation of ArraySub trait for PrimitiveArrays with a scalar
impl<T> ArraySub<T> for PrimitiveArray<T>
where
    T: NativeArithmetics + Sub<Output = T>,
{
    fn sub(&self, rhs: &T) -> Self {
        sub_scalar(self, rhs)
    }
}

// Implementation of ArrayCheckedSub trait for PrimitiveArrays with a scalar
impl<T> ArrayCheckedSub<T> for PrimitiveArray<T>
where
    T: NativeArithmetics + CheckedSub<Output = T>,
{
    fn checked_sub(&self, rhs: &T) -> Self {
        checked_sub_scalar(self, rhs)
    }
}

// Implementation of ArraySaturatingSub trait for PrimitiveArrays with a scalar
impl<T> ArraySaturatingSub<T> for PrimitiveArray<T>
where
    T: NativeArithmetics + SaturatingSub<Output = T>,
{
    fn saturating_sub(&self, rhs: &T) -> Self {
        saturating_sub_scalar(self, rhs)
    }
}

// Implementation of ArraySaturatingSub trait for PrimitiveArrays with a scalar
impl<T> ArrayOverflowingSub<T> for PrimitiveArray<T>
where
    T: NativeArithmetics + OverflowingSub<Output = T>,
{
    fn overflowing_sub(&self, rhs: &T) -> (Self, Bitmap) {
        overflowing_sub_scalar(self, rhs)
    }
}
