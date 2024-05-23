use num_traits::Float;

use self::search_sorted::{binary_search_ca, SearchSortedSide};
use crate::prelude::*;

impl<T> ChunkedArray<T>
where
    T: PolarsFloatType,
    T::Native: Float,
{
    fn float_arg_max_sorted_ascending(&self) -> usize {
        let ca = self;
        debug_assert!(ca.is_sorted_ascending_flag());

        let maybe_max_idx = ca.last_non_null().unwrap();
        let maybe_max = unsafe { ca.value_unchecked(maybe_max_idx) };
        if !maybe_max.is_nan() {
            return maybe_max_idx;
        }

        let search_val = std::iter::once(Some(T::Native::nan()));
        let idx = binary_search_ca(ca, search_val, SearchSortedSide::Left, false)[0] as usize;
        idx.saturating_sub(1)
    }

    fn float_arg_max_sorted_descending(&self) -> usize {
        let ca = self;
        debug_assert!(ca.is_sorted_descending_flag());

        let maybe_max_idx = ca.first_non_null().unwrap();

        let maybe_max = unsafe { ca.value_unchecked(maybe_max_idx) };
        if !maybe_max.is_nan() {
            return maybe_max_idx;
        }

        let search_val = std::iter::once(Some(T::Native::nan()));
        let idx = binary_search_ca(ca, search_val, SearchSortedSide::Right, true)[0] as usize;
        if idx == ca.len() {
            idx - 1
        } else {
            idx
        }
    }
}

/// # Safety
/// `ca` has a float dtype, has at least 1 non-null value and is sorted ascending
pub fn float_arg_max_sorted_ascending<T>(ca: &ChunkedArray<T>) -> usize
where
    T: PolarsNumericType,
{
    with_match_physical_float_polars_type!(ca.dtype(), |$T| {
        let ca: &ChunkedArray<$T> = unsafe {
            &*(ca as *const ChunkedArray<T> as *const ChunkedArray<$T>)
        };
        ca.float_arg_max_sorted_ascending()
    })
}

/// # Safety
/// `ca` has a float dtype, has at least 1 non-null value and is sorted descending
pub fn float_arg_max_sorted_descending<T>(ca: &ChunkedArray<T>) -> usize
where
    T: PolarsNumericType,
{
    with_match_physical_float_polars_type!(ca.dtype(), |$T| {
        let ca: &ChunkedArray<$T> = unsafe {
            &*(ca as *const ChunkedArray<T> as *const ChunkedArray<$T>)
        };
        ca.float_arg_max_sorted_descending()
    })
}
