//! Defines kernels suitable to perform operations to primitive arrays.

use super::utils::{check_same_len, combine_validities_and};
use crate::array::PrimitiveArray;
use crate::datatypes::ArrowDataType;
use crate::types::NativeType;

/// Applies an unary and infallible function to a [`PrimitiveArray`].
///
/// This is the /// fastest way to perform an operation on a [`PrimitiveArray`] when the benefits
/// of a vectorized operation outweighs the cost of branching nulls and non-nulls.
///
/// # Implementation
/// This will apply the function for all values, including those on null slots.
/// This implies that the operation must be infallible for any value of the
/// corresponding type or this function may panic.
#[inline]
pub fn unary<I, F, O>(array: &PrimitiveArray<I>, op: F, dtype: ArrowDataType) -> PrimitiveArray<O>
where
    I: NativeType,
    O: NativeType,
    F: Fn(I) -> O,
{
    let values = array.values().iter().map(|v| op(*v)).collect::<Vec<_>>();

    PrimitiveArray::<O>::new(dtype, values.into(), array.validity().cloned())
}

/// Applies a binary operations to two primitive arrays.
///
/// This is the fastest way to perform an operation on two primitive array when the benefits of a
/// vectorized operation outweighs the cost of branching nulls and non-nulls.
///
/// # Errors
/// This function errors iff the arrays have a different length.
///
/// # Implementation
/// This will apply the function for all values, including those on null slots.
/// This implies that the operation must be infallible for any value of the
/// corresponding type.
/// The types of the arrays are not checked with this operation. The closure
/// "op" needs to handle the different types in the arrays. The datatype for the
/// resulting array has to be selected by the implementer of the function as
/// an argument for the function.
#[inline]
pub fn binary<T, D, F>(
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<D>,
    dtype: ArrowDataType,
    op: F,
) -> PrimitiveArray<T>
where
    T: NativeType,
    D: NativeType,
    F: Fn(T, D) -> T,
{
    check_same_len(lhs, rhs).unwrap();

    let validity = combine_validities_and(lhs.validity(), rhs.validity());

    let values = lhs
        .values()
        .iter()
        .zip(rhs.values().iter())
        .map(|(l, r)| op(*l, *r))
        .collect::<Vec<_>>()
        .into();

    PrimitiveArray::<T>::new(dtype, values, validity)
}
