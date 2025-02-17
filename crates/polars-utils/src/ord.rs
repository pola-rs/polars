use std::cmp::Ordering;

use crate::float::IsFloat;

#[inline]
/// NaN will be smaller than every valid value
pub fn compare_fn_nan_min<T>(a: &T, b: &T) -> Ordering
where
    T: PartialOrd + IsFloat,
{
    // this branch should be optimized away for integers
    if T::is_float() {
        match (a.is_nan(), b.is_nan()) {
            // SAFETY: we checked nans
            (false, false) => unsafe { a.partial_cmp(b).unwrap_unchecked() },
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
        }
    } else {
        // SAFETY:
        // all integers are Ord
        unsafe { a.partial_cmp(b).unwrap_unchecked() }
    }
}

#[inline]
/// NaN will be larger than every valid value
pub fn compare_fn_nan_max<T>(a: &T, b: &T) -> Ordering
where
    T: PartialOrd + IsFloat,
{
    // this branch should be optimized away for integers
    if T::is_float() {
        match (a.is_nan(), b.is_nan()) {
            // SAFETY: we checked nans
            (false, false) => unsafe { a.partial_cmp(b).unwrap_unchecked() },
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
        }
    } else {
        // SAFETY:
        // all integers are Ord
        unsafe { a.partial_cmp(b).unwrap_unchecked() }
    }
}
