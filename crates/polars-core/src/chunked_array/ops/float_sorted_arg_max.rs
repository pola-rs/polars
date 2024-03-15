//! Implementations of the ChunkAgg trait.
use num_traits::Float;

use self::search_sorted::{
    binary_search_array, slice_sorted_non_null_and_offset, SearchSortedSide,
};
use crate::prelude::*;

impl<T> ChunkedArray<T>
where
    T: PolarsFloatType,
    T::Native: Float,
{
    fn float_arg_max_sorted_ascending(&self) -> usize {
        let ca = self;
        debug_assert!(ca.is_sorted_ascending_flag());
        let is_descending = false;
        let side = SearchSortedSide::Left;

        let maybe_max_idx = ca.last_non_null().unwrap();

        let maybe_max = unsafe { ca.value_unchecked(maybe_max_idx) };
        if !maybe_max.is_nan() {
            return maybe_max_idx;
        }

        let (offset, ca) = unsafe { slice_sorted_non_null_and_offset(ca) };
        let arr = unsafe { ca.downcast_get_unchecked(0) };
        let search_val = T::Native::nan();
        let idx = binary_search_array(side, arr, search_val, is_descending) as usize;

        let idx = idx.saturating_sub(1);

        offset + idx
    }

    fn float_arg_max_sorted_descending(&self) -> usize {
        let ca = self;
        debug_assert!(ca.is_sorted_descending_flag());
        let is_descending = true;
        let side = SearchSortedSide::Right;

        let maybe_max_idx = ca.first_non_null().unwrap();

        let maybe_max = unsafe { ca.value_unchecked(maybe_max_idx) };
        if !maybe_max.is_nan() {
            return maybe_max_idx;
        }

        let (offset, ca) = unsafe { slice_sorted_non_null_and_offset(ca) };
        let arr = unsafe { ca.downcast_get_unchecked(0) };
        let search_val = T::Native::nan();
        let idx = binary_search_array(side, arr, search_val, is_descending) as usize;

        let idx = if idx == arr.len() { idx - 1 } else { idx };

        offset + idx
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
