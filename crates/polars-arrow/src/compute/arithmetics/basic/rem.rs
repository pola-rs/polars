use std::ops::Rem;

use num_traits::{CheckedRem, NumCast};
use strength_reduce::{
    StrengthReducedU16, StrengthReducedU32, StrengthReducedU64, StrengthReducedU8,
};

use super::NativeArithmetics;
use crate::array::{Array, PrimitiveArray};
use crate::compute::arithmetics::{ArrayCheckedRem, ArrayRem};
use crate::compute::arity::{binary, binary_checked, unary, unary_checked};
use crate::datatypes::PrimitiveType;

/// Remainder of two primitive arrays with the same type.
/// Panics if the divisor is zero of one pair of values overflows.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::rem;
/// use polars_arrow::array::Int32Array;
///
/// let a = Int32Array::from(&[Some(10), Some(7)]);
/// let b = Int32Array::from(&[Some(5), Some(6)]);
/// let result = rem(&a, &b);
/// let expected = Int32Array::from(&[Some(0), Some(1)]);
/// assert_eq!(result, expected)
/// ```
pub fn rem<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeArithmetics + Rem<Output = T>,
{
    binary(lhs, rhs, lhs.data_type().clone(), |a, b| a % b)
}

/// Checked remainder of two primitive arrays. If the result from the remainder
/// overflows, the result for the operation will change the validity array
/// making this operation None
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::checked_rem;
/// use polars_arrow::array::Int8Array;
///
/// let a = Int8Array::from(&[Some(-100i8), Some(10i8)]);
/// let b = Int8Array::from(&[Some(100i8), Some(0i8)]);
/// let result = checked_rem(&a, &b);
/// let expected = Int8Array::from(&[Some(-0i8), None]);
/// assert_eq!(result, expected);
/// ```
pub fn checked_rem<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeArithmetics + CheckedRem<Output = T>,
{
    let op = move |a: T, b: T| a.checked_rem(&b);

    binary_checked(lhs, rhs, lhs.data_type().clone(), op)
}

impl<T> ArrayRem<PrimitiveArray<T>> for PrimitiveArray<T>
where
    T: NativeArithmetics + Rem<Output = T>,
{
    fn rem(&self, rhs: &PrimitiveArray<T>) -> Self {
        rem(self, rhs)
    }
}

impl<T> ArrayCheckedRem<PrimitiveArray<T>> for PrimitiveArray<T>
where
    T: NativeArithmetics + CheckedRem<Output = T>,
{
    fn checked_rem(&self, rhs: &PrimitiveArray<T>) -> Self {
        checked_rem(self, rhs)
    }
}

/// Remainder a primitive array of type T by a scalar T.
/// Panics if the divisor is zero.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::rem_scalar;
/// use polars_arrow::array::Int32Array;
///
/// let a = Int32Array::from(&[None, Some(6), None, Some(7)]);
/// let result = rem_scalar(&a, &2i32);
/// let expected = Int32Array::from(&[None, Some(0), None, Some(1)]);
/// assert_eq!(result, expected)
/// ```
pub fn rem_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> PrimitiveArray<T>
where
    T: NativeArithmetics + Rem<Output = T> + NumCast,
{
    let rhs = *rhs;

    match T::PRIMITIVE {
        PrimitiveType::UInt64 => {
            let lhs = lhs.as_any().downcast_ref::<PrimitiveArray<u64>>().unwrap();
            let rhs = rhs.to_u64().unwrap();

            let reduced_rem = StrengthReducedU64::new(rhs);

            // small hack to avoid a transmute of `PrimitiveArray<u64>` to `PrimitiveArray<T>`
            let r = unary(lhs, |a| a % reduced_rem, lhs.data_type().clone());
            (&r as &dyn Array)
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .unwrap()
                .clone()
        },
        PrimitiveType::UInt32 => {
            let lhs = lhs.as_any().downcast_ref::<PrimitiveArray<u32>>().unwrap();
            let rhs = rhs.to_u32().unwrap();

            let reduced_rem = StrengthReducedU32::new(rhs);

            let r = unary(lhs, |a| a % reduced_rem, lhs.data_type().clone());
            // small hack to avoid an unsafe transmute of `PrimitiveArray<u64>` to `PrimitiveArray<T>`
            (&r as &dyn Array)
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .unwrap()
                .clone()
        },
        PrimitiveType::UInt16 => {
            let lhs = lhs.as_any().downcast_ref::<PrimitiveArray<u16>>().unwrap();
            let rhs = rhs.to_u16().unwrap();

            let reduced_rem = StrengthReducedU16::new(rhs);

            let r = unary(lhs, |a| a % reduced_rem, lhs.data_type().clone());
            // small hack to avoid an unsafe transmute of `PrimitiveArray<u16>` to `PrimitiveArray<T>`
            (&r as &dyn Array)
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .unwrap()
                .clone()
        },
        PrimitiveType::UInt8 => {
            let lhs = lhs.as_any().downcast_ref::<PrimitiveArray<u8>>().unwrap();
            let rhs = rhs.to_u8().unwrap();

            let reduced_rem = StrengthReducedU8::new(rhs);

            let r = unary(lhs, |a| a % reduced_rem, lhs.data_type().clone());
            // small hack to avoid an unsafe transmute of `PrimitiveArray<u8>` to `PrimitiveArray<T>`
            (&r as &dyn Array)
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .unwrap()
                .clone()
        },
        _ => unary(lhs, |a| a % rhs, lhs.data_type().clone()),
    }
}

/// Checked remainder of a primitive array of type T by a scalar T. If the
/// divisor is zero then the validity array is changed to None.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::checked_rem_scalar;
/// use polars_arrow::array::Int8Array;
///
/// let a = Int8Array::from(&[Some(-100i8)]);
/// let result = checked_rem_scalar(&a, &100i8);
/// let expected = Int8Array::from(&[Some(0i8)]);
/// assert_eq!(result, expected);
/// ```
pub fn checked_rem_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> PrimitiveArray<T>
where
    T: NativeArithmetics + CheckedRem<Output = T>,
{
    let rhs = *rhs;
    let op = move |a: T| a.checked_rem(&rhs);

    unary_checked(lhs, op, lhs.data_type().clone())
}

impl<T> ArrayRem<T> for PrimitiveArray<T>
where
    T: NativeArithmetics + Rem<Output = T> + NumCast,
{
    fn rem(&self, rhs: &T) -> Self {
        rem_scalar(self, rhs)
    }
}

impl<T> ArrayCheckedRem<T> for PrimitiveArray<T>
where
    T: NativeArithmetics + CheckedRem<Output = T>,
{
    fn checked_rem(&self, rhs: &T) -> Self {
        checked_rem_scalar(self, rhs)
    }
}
