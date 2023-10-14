//! Definition of basic pow operations with primitive arrays
use num_traits::{checked_pow, CheckedMul, One, Pow};

use super::NativeArithmetics;
use crate::array::PrimitiveArray;
use crate::compute::arity::{unary, unary_checked};

/// Raises an array of primitives to the power of exponent. Panics if one of
/// the values values overflows.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::powf_scalar;
/// use polars_arrow::array::Float32Array;
///
/// let a = Float32Array::from(&[Some(2f32), None]);
/// let actual = powf_scalar(&a, 2.0);
/// let expected = Float32Array::from(&[Some(4f32), None]);
/// assert_eq!(expected, actual);
/// ```
pub fn powf_scalar<T>(array: &PrimitiveArray<T>, exponent: T) -> PrimitiveArray<T>
where
    T: NativeArithmetics + Pow<T, Output = T>,
{
    unary(array, |x| x.pow(exponent), array.data_type().clone())
}

/// Checked operation of raising an array of primitives to the power of
/// exponent. If the result from the multiplications overflows, the validity
/// for that index is changed returned.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::basic::checked_powf_scalar;
/// use polars_arrow::array::Int8Array;
///
/// let a = Int8Array::from(&[Some(1i8), None, Some(7i8)]);
/// let actual = checked_powf_scalar(&a, 8usize);
/// let expected = Int8Array::from(&[Some(1i8), None, None]);
/// assert_eq!(expected, actual);
/// ```
pub fn checked_powf_scalar<T>(array: &PrimitiveArray<T>, exponent: usize) -> PrimitiveArray<T>
where
    T: NativeArithmetics + CheckedMul + One,
{
    let op = move |a: T| checked_pow(a, exponent);

    unary_checked(array, op, array.data_type().clone())
}
