use std::cmp::Ordering;
use std::fmt::Debug;

use arrow::array::Array;
use arrow::legacy::prelude::*;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::total_ord::{TotalEq, TotalOrd};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SearchSortedSide {
    #[default]
    Any,
    Left,
    Right,
}

/// Search the left or right index that still fulfills the requirements.
fn finish_side<'a, A>(
    side: SearchSortedSide,
    out: &mut Vec<IdxSize>,
    mid: IdxSize,
    arr: &'a A,
    len: usize,
) where
    A: StaticArray,
    A::ValueT<'a>: TotalOrd + Debug + Copy,
{
    let mut mid = mid;

    // approach the boundary from any side
    // this is O(n) we could make this binary search later
    match side {
        SearchSortedSide::Any => {
            out.push(mid);
        },
        SearchSortedSide::Left => {
            if mid as usize == len {
                mid -= 1;
            }

            let current = unsafe { arr.get_unchecked(mid as usize) };
            loop {
                if mid == 0 {
                    out.push(mid);
                    break;
                }
                mid -= 1;
                if current.tot_ne(unsafe { &arr.get_unchecked(mid as usize) }) {
                    out.push(mid + 1);
                    break;
                }
            }
        },
        SearchSortedSide::Right => {
            if mid as usize == len {
                out.push(mid);
                return;
            }
            let current = unsafe { arr.get_unchecked(mid as usize) };
            let bound = (len - 1) as IdxSize;
            loop {
                if mid >= bound {
                    out.push(mid + 1);
                    break;
                }
                mid += 1;
                if current.tot_ne(unsafe { &arr.get_unchecked(mid as usize) }) {
                    out.push(mid);
                    break;
                }
            }
        },
    }
}

fn binary_search_array<'a, A>(
    side: SearchSortedSide,
    out: &mut Vec<IdxSize>,
    arr: &'a A,
    len: usize,
    search_value: A::ValueT<'a>,
    descending: bool,
) where
    A: StaticArray,
    A::ValueT<'a>: TotalOrd + Debug + Copy,
{
    let mut size = len as IdxSize;
    let mut left = 0 as IdxSize;
    let mut right = size;
    let current_len = out.len();
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
            finish_side(side, out, mid, arr, len);
            break;
        }

        size = right - left;
    }
    if out.len() == current_len {
        out.push(left);
    }
}

fn search_sorted_ca_array<T>(
    ca: &ChunkedArray<T>,
    search_values: &ChunkedArray<T>,
    side: SearchSortedSide,
    descending: bool,
) -> Vec<IdxSize>
where
    T: PolarsNumericType,
{
    let ca = ca.rechunk();
    let arr = ca.downcast_iter().next().unwrap();

    let mut out = Vec::with_capacity(search_values.len());

    for search_arr in search_values.downcast_iter() {
        if search_arr.null_count() == 0 {
            for search_value in search_arr.values_iter() {
                binary_search_array(side, &mut out, arr, ca.len(), *search_value, descending)
            }
        } else {
            for opt_v in search_arr.into_iter() {
                match opt_v {
                    None => out.push(0),
                    Some(search_value) => binary_search_array(
                        side,
                        &mut out,
                        arr,
                        ca.len(),
                        *search_value,
                        descending,
                    ),
                }
            }
        }
    }
    out
}

fn search_sorted_bin_array(
    ca: &BinaryChunked,
    search_values: &BinaryChunked,
    side: SearchSortedSide,
    descending: bool,
) -> Vec<IdxSize> {
    let ca = ca.rechunk();
    let arr = ca.downcast_iter().next().unwrap();

    let mut out = Vec::with_capacity(search_values.len());

    for search_arr in search_values.downcast_iter() {
        if search_arr.null_count() == 0 {
            for search_value in search_arr.values_iter() {
                binary_search_array(side, &mut out, arr, ca.len(), search_value, descending)
            }
        } else {
            for opt_v in search_arr.into_iter() {
                match opt_v {
                    None => out.push(0),
                    Some(search_value) => {
                        binary_search_array(side, &mut out, arr, ca.len(), search_value, descending)
                    },
                }
            }
        }
    }
    out
}

pub fn search_sorted(
    s: &Series,
    search_values: &Series,
    side: SearchSortedSide,
    descending: bool,
) -> PolarsResult<IdxCa> {
    let original_dtype = s.dtype();
    let s = s.to_physical_repr();
    let phys_dtype = s.dtype();

    match phys_dtype {
        DataType::String => {
            let ca = s.str().unwrap();
            let ca = ca.as_binary();
            let search_values = search_values.str()?;
            let search_values = search_values.as_binary();
            let idx = search_sorted_bin_array(&ca, &search_values, side, descending);

            Ok(IdxCa::new_vec(s.name(), idx))
        },
        DataType::Binary => {
            let ca = s.binary().unwrap();
            let search_values = search_values.binary().unwrap();
            let idx = search_sorted_bin_array(ca, search_values, side, descending);

            Ok(IdxCa::new_vec(s.name(), idx))
        },
        dt if dt.is_numeric() => {
            let search_values = search_values.to_physical_repr();

            let idx = with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                let search_values: &ChunkedArray<$T> = search_values.as_ref().as_ref().as_ref();

                search_sorted_ca_array(ca, search_values, side, descending)
            });
            Ok(IdxCa::new_vec(s.name(), idx))
        },
        _ => polars_bail!(opq = search_sorted, original_dtype),
    }
}
