//! Definition of basic add operations with primitive arrays
use std::ops::Add;

use super::NativeArithmetics;
use crate::array::PrimitiveArray;
use crate::compute::arity::{binary, unary};

/// Adds two primitive arrays with the same type.
/// Panics if the sum of one pair of values overflows.
pub fn add<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeArithmetics + Add<Output = T>,
{
    binary(lhs, rhs, lhs.data_type().clone(), |a, b| a + b)
}

/// Adds a scalar T to a primitive array of type T.
/// Panics if the sum of the values overflows.
pub fn add_scalar<T>(lhs: &PrimitiveArray<T>, rhs: &T) -> PrimitiveArray<T>
where
    T: NativeArithmetics + Add<Output = T>,
{
    let rhs = *rhs;
    unary(lhs, |a| a + rhs, lhs.data_type().clone())
}
