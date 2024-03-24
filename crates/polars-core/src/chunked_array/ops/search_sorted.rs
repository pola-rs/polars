use std::cmp::Ordering;
use std::fmt::Debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SearchSortedSide {
    #[default]
    Any,
    Left,
    Right,
}

/// Search the left or right index that still fulfills the requirements.
fn get_side_idx<'a, A>(side: SearchSortedSide, mid: IdxSize, arr: &'a A, len: usize) -> IdxSize
where
    A: StaticArray,
    A::ValueT<'a>: TotalOrd + Debug + Copy,
{
    let mut mid = mid;

    // approach the boundary from any side
    // this is O(n) we could make this binary search later
    match side {
        SearchSortedSide::Any => mid,
        SearchSortedSide::Left => {
            if mid as usize == len {
                mid -= 1;
            }

            let current = unsafe { arr.get_unchecked(mid as usize) };
            loop {
                if mid == 0 {
                    return mid;
                }
                mid -= 1;
                if current.tot_ne(unsafe { &arr.get_unchecked(mid as usize) }) {
                    return mid + 1;
                }
            }
        },
        SearchSortedSide::Right => {
            if mid as usize == len {
                return mid;
            }
            let current = unsafe { arr.get_unchecked(mid as usize) };
            let bound = (len - 1) as IdxSize;
            loop {
                if mid >= bound {
                    return mid + 1;
                }
                mid += 1;
                if current.tot_ne(unsafe { &arr.get_unchecked(mid as usize) }) {
                    return mid;
                }
            }
        },
    }
}

pub fn binary_search_array<'a, A>(
    side: SearchSortedSide,
    arr: &'a A,
    search_value: A::ValueT<'a>,
    descending: bool,
) -> IdxSize
where
    A: StaticArray,
    A::ValueT<'a>: TotalOrd + Debug + Copy,
{
    let mut size = arr.len() as IdxSize;
    let mut left = 0 as IdxSize;
    let mut right = size;
    while left < right {
        let mid = left + size / 2;

        // SAFETY: the call is made safe by the following invariants:
        // - `mid >= 0`
        // - `mid < size`: `mid` is limited by `[left; right)` bound.
        let cmp = match unsafe { arr.get_unchecked(mid as usize) } {
            None => Ordering::Less,
            Some(value) => {
                if descending {
                    search_value.tot_cmp(&value)
                } else {
                    value.tot_cmp(&search_value)
                }
            },
        };

        // The reason why we use if/else control flow rather than match
        // is because match reorders comparison operations, which is perf sensitive.
        // This is x86 asm for u8: https://rust.godbolt.org/z/8Y8Pra.
        if cmp == Ordering::Less {
            left = mid + 1;
        } else if cmp == Ordering::Greater {
            right = mid;
        } else {
            return get_side_idx(side, mid, arr, arr.len());
        }

        size = right - left;
    }

    left
}

/// Get a slice of the non-null values of a sorted array. The returned array
/// will have a single chunk.
/// # Safety
/// The array is sorted and has at least one non-null value.
pub unsafe fn slice_sorted_non_null_and_offset<T>(ca: &ChunkedArray<T>) -> (usize, ChunkedArray<T>)
where
    T: PolarsDataType,
{
    let offset = ca.first_non_null().unwrap();
    let length = 1 + ca.last_non_null().unwrap() - offset;
    let out = ca.slice(offset as i64, length);

    debug_assert!(out.null_count() != out.len());
    debug_assert!(out.null_count() == 0);

    (offset, out.rechunk())
}
