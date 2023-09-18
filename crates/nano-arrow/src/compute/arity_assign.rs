//! Defines generics suitable to perform operations to [`PrimitiveArray`] in-place.

use super::utils::check_same_len;
use crate::{array::PrimitiveArray, types::NativeType};
use either::Either;

/// Applies an unary function to a [`PrimitiveArray`], optionally in-place.
///
/// # Implementation
/// This function tries to apply the function directly to the values of the array.
/// If that region is shared, this function creates a new region and writes to it.
///
/// # Panics
/// This function panics iff
/// * the arrays have a different length.
/// * the function itself panics.
#[inline]
pub fn unary<I, F>(array: &mut PrimitiveArray<I>, op: F)
where
    I: NativeType,
    F: Fn(I) -> I,
{
    if let Some(values) = array.get_mut_values() {
        // mutate in place
        values.iter_mut().for_each(|l| *l = op(*l));
    } else {
        // alloc and write to new region
        let values = array.values().iter().map(|l| op(*l)).collect::<Vec<_>>();
        array.set_values(values.into());
    }
}

/// Applies a binary function to two [`PrimitiveArray`]s, optionally in-place, returning
/// a new [`PrimitiveArray`].
///
/// # Implementation
/// This function tries to apply the function directly to the values of the array.
/// If that region is shared, this function creates a new region and writes to it.
/// # Panics
/// This function panics iff
/// * the arrays have a different length.
/// * the function itself panics.
#[inline]
pub fn binary<T, D, F>(lhs: &mut PrimitiveArray<T>, rhs: &PrimitiveArray<D>, op: F)
where
    T: NativeType,
    D: NativeType,
    F: Fn(T, D) -> T,
{
    check_same_len(lhs, rhs).unwrap();

    // both for the validity and for the values
    // we branch to check if we can mutate in place
    // if we can, great that is fastest.
    // if we cannot, we allocate a new buffer and assign values to that
    // new buffer, that is benchmarked to be ~2x faster than first memcpy and assign in place
    // for the validity bits it can be much faster as we might need to iterate all bits if the
    // bitmap has an offset.
    if let Some(rhs) = rhs.validity() {
        if lhs.validity().is_none() {
            lhs.set_validity(Some(rhs.clone()));
        } else {
            lhs.apply_validity(|bitmap| {
                match bitmap.into_mut() {
                    Either::Left(immutable) => {
                        // alloc new region
                        &immutable & rhs
                    }
                    Either::Right(mutable) => {
                        // mutate in place
                        (mutable & rhs).into()
                    }
                }
            });
        }
    };

    if let Some(values) = lhs.get_mut_values() {
        // mutate values in place
        values
            .iter_mut()
            .zip(rhs.values().iter())
            .for_each(|(l, r)| *l = op(*l, *r));
    } else {
        // alloc new region
        let values = lhs
            .values()
            .iter()
            .zip(rhs.values().iter())
            .map(|(l, r)| op(*l, *r))
            .collect::<Vec<_>>();
        lhs.set_values(values.into());
    }
}
