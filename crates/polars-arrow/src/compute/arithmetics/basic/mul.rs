//! Definition of basic mul operations with primitive arrays
use std::ops::Mul;

use super::NativeArithmetics;
use crate::array::PrimitiveArray;
use crate::compute::arity::{binary, unary};

/// Multiplies two primitive arrays with the same type.
/// Panics if the multiplication of one pair of values overflows.
pub fn mul<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeArithmetics + Mul<Output = T>,
{
    binary(lhs, rhs, lhs.data_type().clone(), |a, b| a * b)
}

/// Multiply a scalar T to a primitive array of type T.
/// Panics if the multiplication of the values overflows.
pub fn mul_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> PrimitiveArray<T>
where
    T: NativeArithmetics + Mul<Output = T>,
{
    let rhs = *rhs;
    unary(lhs, |a| a * rhs, lhs.data_type().clone())
}
