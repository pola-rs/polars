//! Definition of basic sub operations with primitive arrays
use std::ops::Sub;

use super::NativeArithmetics;
use crate::array::PrimitiveArray;
use crate::compute::arity::{binary, unary};

/// Subtracts two primitive arrays with the same type.
/// Panics if the subtraction of one pair of values overflows.
pub fn sub<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeArithmetics + Sub<Output = T>,
{
    binary(lhs, rhs, lhs.data_type().clone(), |a, b| a - b)
}

/// Subtract a scalar T to a primitive array of type T.
/// Panics if the subtraction of the values overflows.
pub fn sub_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> PrimitiveArray<T>
where
    T: NativeArithmetics + Sub<Output = T>,
{
    let rhs = *rhs;
    unary(lhs, |a| a - rhs, lhs.data_type().clone())
}
