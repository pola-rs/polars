use num_traits::Float;

use self::search_sorted::{SearchSortedSide, binary_search_ca};
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

        // The value before the left-most NaN is the largest non-NaN one, if there is any.
        let search_val = std::iter::once(Some(T::Native::nan()));
        let idx = binary_search_ca(ca, search_val, SearchSortedSide::Left, false)[0] as usize;
        let candidate = idx.saturating_sub(1);
        if candidate < ca.first_non_null().unwrap() {
            // Stepping back landed on a null, so every non-null value is NaN. Report a NaN,
            // as the NaN-ignoring reduction does for an all-NaN input.
            maybe_max_idx
        } else {
            candidate
        }
    }

    fn float_arg_max_sorted_descending(&self) -> usize {
        let ca = self;
        debug_assert!(ca.is_sorted_descending_flag());

        let maybe_max_idx = ca.first_non_null().unwrap();

        let maybe_max = unsafe { ca.value_unchecked(maybe_max_idx) };
        if !maybe_max.is_nan() {
            return maybe_max_idx;
        }

        // The value after the right-most NaN is the largest non-NaN one, if there is any.
        let search_val = std::iter::once(Some(T::Native::nan()));
        let idx = binary_search_ca(ca, search_val, SearchSortedSide::Right, true)[0] as usize;
        let candidate = if idx == ca.len() { idx - 1 } else { idx };
        if candidate > ca.last_non_null().unwrap() {
            // Stepping forward landed on a null, so every non-null value is NaN. Report a NaN,
            // as the NaN-ignoring reduction does for an all-NaN input.
            maybe_max_idx
        } else {
            candidate
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
