//! The `list.get` kernels over the arrays of `polars-array`.

use std::ops::Range;

use polars_array::builder::new_full_null_like;
use polars_array::{PlArray, PlListArray, PlPrimitiveArray};
use polars_arrow::bitmap::bitmask::BitMask;
use polars_arrow::legacy::index::IndexToUsize;
use polars_utils::IdxSize;

use crate::gather::take_unchecked;

/// The position in the values array that `index` picks out of an element covering `range`, or
/// `None` if it falls outside it.
///
/// A negative index counts back from the end of the element, and an element that covers no values
/// at all has no position for any index to land on.
#[inline]
fn position_in(index: i64, range: Range<usize>) -> Option<usize> {
    index
        .negative_to_usize(range.len())
        .map(|position| range.start + position)
}

/// Returns the value at `index` within every element of `arr`.
pub fn sublist_get(arr: &PlListArray, index: i64) -> Box<dyn PlArray> {
    if arr.is_empty() {
        return arr.values().sliced(0, 0);
    }

    if arr.is_scalar() {
        // SAFETY: the array holds at least one element, so element 0 is in bounds.
        let position = unsafe {
            (!arr.is_null_unchecked(0))
                .then(|| position_in(index, arr.value_range_unchecked(0)))
                .flatten()
        };

        return match position {
            // SAFETY: the position lies within the range the element covers.
            Some(position) => unsafe { arr.values().new_from_index_unchecked(position, arr.len()) },
            None => new_full_null_like(arr.values(), arr.len()),
        };
    }

    if let Some(offsets) = arr.flat_offsets() {
        let offsets = offsets.as_slice();
        let indices = match arr
            .validity()
            .and_then(|validity| match validity.scalar_value() {
                Some(true) => None,
                Some(false) => Some(Err(())),
                None => Some(Ok(BitMask::from_bitmap(
                    validity.flat_bitmap().expect("a mask is flat or scalar"),
                ))),
            }) {
            Some(Err(())) => return new_full_null_like(arr.values(), arr.len()),
            // SAFETY: the mask holds one bit per element, so every index below is in bounds.
            Some(Ok(mask)) => {
                positions_in(offsets, index, |i| unsafe { mask.get_bit_unchecked(i) })
            },
            None => positions_in(offsets, index, |_| true),
        };

        // SAFETY: every index lands within the range the element it is read for covers.
        return unsafe { take_unchecked(arr.values(), &indices) };
    }

    let indices = (0..arr.len())
        .map(|i| {
            // SAFETY: `i` is an element of `arr`.
            unsafe {
                (!arr.is_null_unchecked(i))
                    .then(|| position_in(index, arr.value_range_unchecked(i)))
                    .flatten()
            }
            .map(|position| position as IdxSize)
        })
        .collect::<PlPrimitiveArray<IdxSize>>();

    // SAFETY: every index lands within the range the element it is read for covers.
    unsafe { take_unchecked(arr.values(), &indices) }
}

/// The position `index` lands on within each of the elements `offsets` holds the ends of.
#[inline]
fn positions_in(
    offsets: &[u64],
    index: i64,
    is_valid: impl Fn(usize) -> bool,
) -> PlPrimitiveArray<IdxSize> {
    let mut start = offsets[0] as usize;

    offsets[1..]
        .iter()
        .enumerate()
        .map(|(i, &end)| {
            let range = start..end as usize;
            start = range.end;

            is_valid(i)
                .then(|| position_in(index, range))
                .flatten()
                .map(|position| position as IdxSize)
        })
        .collect()
}

/// Whether `index` falls outside at least one of the non-null elements of `arr`.
pub fn index_is_oob(arr: &PlListArray, index: i64) -> bool {
    if arr.null_count() == arr.len() {
        return false;
    }

    if let Some(range) = arr.scalar_offsets() {
        return position_in(index, range).is_none();
    }

    let offsets = arr
        .flat_offsets()
        .expect("the elements cover ranges of their own")
        .as_slice();
    let validity = arr.validity();

    let mut start = offsets[0] as usize;
    for (i, &end) in offsets[1..].iter().enumerate() {
        let range = start..end as usize;
        start = range.end;

        // SAFETY: `i` is an element of `arr`, which the mask covers.
        let is_valid = validity.is_none_or(|validity| unsafe { validity.get_unchecked(i) });
        if is_valid && position_in(index, range).is_none() {
            return true;
        }
    }

    false
}

/// Wraps every element of `array` in a list of its own, turning `[1, 2, 3]` into `[[1], [2], [3]]`.
pub fn array_to_unit_list(array: Box<dyn PlArray>) -> PlListArray {
    let length = array.len();
    if length == 0 {
        return PlListArray::new_empty(array);
    }

    if array.is_scalar() {
        return PlListArray::new_scalar(array.sliced(0, 1), length);
    }

    unsafe { PlListArray::new_unchecked(array, (0..=length as u64).collect(), length, None) }
}
