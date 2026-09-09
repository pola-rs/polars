//! The `list.get` kernels over the arrays of `polars-array`.

use std::ops::Range;

use arrow::bitmap::bitmask::BitMask;
use arrow::legacy::index::IndexToUsize;
use polars_array::builder::new_full_null_like;
use polars_array::{PlArray, PlListArray, PlPrimitiveArray};
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

    // A chunk whose own buffers stand for a single list repeated says the same of every element:
    // the index lands at one position within that list, and the value there is the answer at every
    // element in turn, in `O(1)`.
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

    // The elements cover ranges of their own, which the offsets hold end to end: walking them and
    // the mask as the slices they are settles the representation once instead of at every element.
    if let Some(offsets) = arr.flat_offsets() {
        let offsets = offsets.as_slice();
        let indices = match arr.validity().and_then(|validity| {
            // A mask of a single bit says the same of every element: either none of them is null,
            // which is nothing to read, or all of them are.
            match validity.scalar_value() {
                Some(true) => None,
                Some(false) => Some(Err(())),
                None => Some(Ok(BitMask::from_bitmap(
                    validity.flat_bitmap().expect("a mask is flat or scalar"),
                ))),
            }
        }) {
            // Every element is null, so no index lands anywhere and no value is ever read.
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

    // The elements share one range but not one mask bit, so the index lands at the same position
    // for every one of them and it is only the mask that tells them apart.
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
///
/// An element `is_valid` answers `false` for is null, which no index lands within.
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
    // An array of nothing but nulls holds no list for the index to fall outside of, which covers
    // an empty one as well.
    if arr.null_count() == arr.len() {
        return false;
    }

    // Offsets that hold a single range say every element covers it, and at least one element is
    // not null: whether the index falls outside that one range answers for the whole array.
    if let Some(range) = arr.scalar_offsets() {
        return position_in(index, range).is_none();
    }

    // The offsets hold the range of every element, and the mask says which of them are null: both
    // are read out once, so the loop below touches nothing but the slices they are.
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

    // A chunk that repeats one element wraps into one list repeated: every element is the list of
    // that same value, so the offsets need hold nothing but the range it covers.
    if array.is_scalar() {
        return PlListArray::new_scalar(array.sliced(0, 1), length);
    }

    // Every element covers the one value at its own position, so the offsets count up by one.
    //
    // SAFETY: those offsets are one per element plus the end of the last, ascending, and they end
    // at the length of the values — which is what a pass over them would have to check.
    unsafe { PlListArray::new_unchecked(array, (0..=length as u64).collect(), length, None) }
}
