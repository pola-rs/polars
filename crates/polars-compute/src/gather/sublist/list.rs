//! The `list.get` kernels over the arrays of `polars-array`.

use std::ops::Range;

use arrow::legacy::index::IndexToUsize;
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
            None => arr.values().new_full_null(arr.len()),
        };
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

    (0..arr.len()).any(|i| {
        // SAFETY: `i` is an element of `arr`.
        unsafe {
            !arr.is_null_unchecked(i) && position_in(index, arr.value_range_unchecked(i)).is_none()
        }
    })
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
    PlListArray::new(array, (0..=length as u64).collect(), length, None)
}
