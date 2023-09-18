//! Contains arithemtic functions for [`PrimitiveArray`]s.
//!
//! Each operation has four variants, like the rest of Rust's ecosystem:
//! * usual, that [`panic!`]s on overflow
//! * `checked_*` that turns overflowings to `None`
//! * `overflowing_*` returning a [`Bitmap`](crate::bitmap::Bitmap) with items that overflow.
//! * `saturating_*` that saturates the result.
mod add;
pub use add::*;
mod div;
pub use div::*;
mod mul;
pub use mul::*;
mod pow;
pub use pow::*;
mod rem;
pub use rem::*;
mod sub;
pub use sub::*;

use std::ops::Neg;

use num_traits::{CheckedNeg, WrappingNeg};

use crate::{array::PrimitiveArray, types::NativeType};

use super::super::arity::{unary, unary_checked};

/// Trait describing a [`NativeType`] whose semantics of arithmetic in Arrow equals
/// the semantics in Rust.
/// A counter example is `i128`, that in arrow represents a decimal while in rust represents
/// a signed integer.
pub trait NativeArithmetics: NativeType {}
impl NativeArithmetics for u8 {}
impl NativeArithmetics for u16 {}
impl NativeArithmetics for u32 {}
impl NativeArithmetics for u64 {}
impl NativeArithmetics for i8 {}
impl NativeArithmetics for i16 {}
impl NativeArithmetics for i32 {}
impl NativeArithmetics for i64 {}
impl NativeArithmetics for f32 {}
impl NativeArithmetics for f64 {}

/// Negates values from array.
///
/// # Examples
/// ```
/// use arrow2::compute::arithmetics::basic::negate;
/// use arrow2::array::PrimitiveArray;
///
/// let a = PrimitiveArray::from([None, Some(6), None, Some(7)]);
/// let result = negate(&a);
/// let expected = PrimitiveArray::from([None, Some(-6), None, Some(-7)]);
/// assert_eq!(result, expected)
/// ```
pub fn negate<T>(array: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeType + Neg<Output = T>,
{
    unary(array, |a| -a, array.data_type().clone())
}

/// Checked negates values from array.
///
/// # Examples
/// ```
/// use arrow2::compute::arithmetics::basic::checked_negate;
/// use arrow2::array::{Array, PrimitiveArray};
///
/// let a = PrimitiveArray::from([None, Some(6), Some(i8::MIN), Some(7)]);
/// let result = checked_negate(&a);
/// let expected = PrimitiveArray::from([None, Some(-6), None, Some(-7)]);
/// assert_eq!(result, expected);
/// assert!(!result.is_valid(2))
/// ```
pub fn checked_negate<T>(array: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeType + CheckedNeg,
{
    unary_checked(array, |a| a.checked_neg(), array.data_type().clone())
}

/// Wrapping negates values from array.
///
/// # Examples
/// ```
/// use arrow2::compute::arithmetics::basic::wrapping_negate;
/// use arrow2::array::{Array, PrimitiveArray};
///
/// let a = PrimitiveArray::from([None, Some(6), Some(i8::MIN), Some(7)]);
/// let result = wrapping_negate(&a);
/// let expected = PrimitiveArray::from([None, Some(-6), Some(i8::MIN), Some(-7)]);
/// assert_eq!(result, expected);
/// ```
pub fn wrapping_negate<T>(array: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeType + WrappingNeg,
{
    unary(array, |a| a.wrapping_neg(), array.data_type().clone())
}
