//! Definition of basic div operations with primitive arrays
use std::ops::Div;

use num_traits::{CheckedDiv, NumCast};
use strength_reduce::{
    StrengthReducedU16, StrengthReducedU32, StrengthReducedU64, StrengthReducedU8,
};

use super::NativeArithmetics;
use crate::array::{Array, PrimitiveArray};
use crate::compute::arithmetics::{ArrayCheckedDiv, ArrayDiv};
use crate::compute::arity::{binary, binary_checked, unary, unary_checked};
use crate::compute::utils::check_same_len;
use crate::datatypes::PrimitiveType;

/// Divides two primitive arrays with the same type.
/// Panics if the divisor is zero of one pair of values overflows.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::div;
/// use polars_arrow::array::Int32Array;
///
/// let a = Int32Array::from(&[Some(10), Some(1), Some(6)]);
/// let b = Int32Array::from(&[Some(5), None, Some(6)]);
/// let result = div(&a, &b);
/// let expected = Int32Array::from(&[Some(2), None, Some(1)]);
/// assert_eq!(result, expected)
/// ```
pub fn div<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeArithmetics + Div<Output = T>,
{
    if rhs.null_count() == 0 {
        binary(lhs, rhs, lhs.data_type().clone(), |a, b| a / b)
    } else {
        check_same_len(lhs, rhs).unwrap();
        let values = lhs.iter().zip(rhs.iter()).map(|(l, r)| match (l, r) {
            (Some(l), Some(r)) => Some(*l / *r),
            _ => None,
        });

        PrimitiveArray::from_trusted_len_iter(values).to(lhs.data_type().clone())
    }
}

/// Checked division of two primitive arrays. If the result from the division
/// overflows, the result for the operation will change the validity array
/// making this operation None
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::checked_div;
/// use polars_arrow::array::Int8Array;
///
/// let a = Int8Array::from(&[Some(-100i8), Some(10i8)]);
/// let b = Int8Array::from(&[Some(100i8), Some(0i8)]);
/// let result = checked_div(&a, &b);
/// let expected = Int8Array::from(&[Some(-1i8), None]);
/// assert_eq!(result, expected);
/// ```
pub fn checked_div<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeArithmetics + CheckedDiv<Output = T>,
{
    let op = move |a: T, b: T| a.checked_div(&b);
    binary_checked(lhs, rhs, lhs.data_type().clone(), op)
}

// Implementation of ArrayDiv trait for PrimitiveArrays
impl<T> ArrayDiv<PrimitiveArray<T>> for PrimitiveArray<T>
where
    T: NativeArithmetics + Div<Output = T>,
{
    fn div(&self, rhs: &PrimitiveArray<T>) -> Self {
        div(self, rhs)
    }
}

// Implementation of ArrayCheckedDiv trait for PrimitiveArrays
impl<T> ArrayCheckedDiv<PrimitiveArray<T>> for PrimitiveArray<T>
where
    T: NativeArithmetics + CheckedDiv<Output = T>,
{
    fn checked_div(&self, rhs: &PrimitiveArray<T>) -> Self {
        checked_div(self, rhs)
    }
}

/// Divide a primitive array of type T by a scalar T.
/// Panics if the divisor is zero.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::div_scalar;
/// use polars_arrow::array::Int32Array;
///
/// let a = Int32Array::from(&[None, Some(6), None, Some(6)]);
/// let result = div_scalar(&a, &2i32);
/// let expected = Int32Array::from(&[None, Some(3), None, Some(3)]);
/// assert_eq!(result, expected)
/// ```
pub fn div_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> PrimitiveArray<T>
where
    T: NativeArithmetics + Div<Output = T> + NumCast,
{
    let rhs = *rhs;
    match T::PRIMITIVE {
        PrimitiveType::UInt64 => {
            let lhs = lhs.as_any().downcast_ref::<PrimitiveArray<u64>>().unwrap();
            let rhs = rhs.to_u64().unwrap();

            let reduced_div = StrengthReducedU64::new(rhs);
            let r = unary(lhs, |a| a / reduced_div, lhs.data_type().clone());
            (&r as &dyn Array)
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .unwrap()
                .clone()
        },
        PrimitiveType::UInt32 => {
            let lhs = lhs.as_any().downcast_ref::<PrimitiveArray<u32>>().unwrap();
            let rhs = rhs.to_u32().unwrap();

            let reduced_div = StrengthReducedU32::new(rhs);
            let r = unary(lhs, |a| a / reduced_div, lhs.data_type().clone());
            (&r as &dyn Array)
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .unwrap()
                .clone()
        },
        PrimitiveType::UInt16 => {
            let lhs = lhs.as_any().downcast_ref::<PrimitiveArray<u16>>().unwrap();
            let rhs = rhs.to_u16().unwrap();

            let reduced_div = StrengthReducedU16::new(rhs);

            let r = unary(lhs, |a| a / reduced_div, lhs.data_type().clone());
            (&r as &dyn Array)
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .unwrap()
                .clone()
        },
        PrimitiveType::UInt8 => {
            let lhs = lhs.as_any().downcast_ref::<PrimitiveArray<u8>>().unwrap();
            let rhs = rhs.to_u8().unwrap();

            let reduced_div = StrengthReducedU8::new(rhs);
            let r = unary(lhs, |a| a / reduced_div, lhs.data_type().clone());
            (&r as &dyn Array)
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .unwrap()
                .clone()
        },
        _ => unary(lhs, |a| a / rhs, lhs.data_type().clone()),
    }
}

/// Checked division of a primitive array of type T by a scalar T. If the
/// divisor is zero then the validity array is changed to None.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::checked_div_scalar;
/// use polars_arrow::array::Int8Array;
///
/// let a = Int8Array::from(&[Some(-100i8)]);
/// let result = checked_div_scalar(&a, &100i8);
/// let expected = Int8Array::from(&[Some(-1i8)]);
/// assert_eq!(result, expected);
/// ```
pub fn checked_div_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> PrimitiveArray<T>
where
    T: NativeArithmetics + CheckedDiv<Output = T>,
{
    let rhs = *rhs;
    let op = move |a: T| a.checked_div(&rhs);

    unary_checked(lhs, op, lhs.data_type().clone())
}

// Implementation of ArrayDiv trait for PrimitiveArrays with a scalar
impl<T> ArrayDiv<T> for PrimitiveArray<T>
where
    T: NativeArithmetics + Div<Output = T> + NumCast,
{
    fn div(&self, rhs: &T) -> Self {
        div_scalar(self, rhs)
    }
}

// Implementation of ArrayCheckedDiv trait for PrimitiveArrays with a scalar
impl<T> ArrayCheckedDiv<T> for PrimitiveArray<T>
where
    T: NativeArithmetics + CheckedDiv<Output = T>,
{
    fn checked_div(&self, rhs: &T) -> Self {
        checked_div_scalar(self, rhs)
    }
}
