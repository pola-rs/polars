//! The `list.get` kernels over the arrays of `polars-array`.
//!
//! The elements of a list array cover ranges of one values array, so indexing them is a gather out
//! of those values: the index is turned into one position per element and read out of the values
//! in whatever representation they are in. Offsets that hold a single range say that every element
//! covers it, so the index is resolved once and the one value it lands on stands for the whole
//! result.

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
///
/// An element that is null, and one the index falls outside of, read as a null value.
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
            None => arr.values().full_null_like(arr.len()),
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

#[cfg(test)]
mod test {
    use polars_array::{PlBitmap, StaticArray};
    use polars_buffer::Buffer;

    use super::*;

    /// `[[1, 2, 3], [4, 5], [6]]`.
    fn get_array() -> PlListArray {
        PlListArray::from_offsets(
            PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5, 6]).into_boxed(),
            Buffer::from(vec![0u64, 3, 5, 6]),
        )
    }

    fn as_i32(array: &dyn PlArray) -> Vec<Option<i32>> {
        array
            .as_any()
            .downcast_ref::<PlPrimitiveArray<i32>>()
            .unwrap()
            .iter()
            .collect()
    }

    #[test]
    fn every_element_is_indexed() {
        let arr = get_array();

        assert_eq!(as_i32(&*sublist_get(&arr, 0)), [Some(1), Some(4), Some(6)]);
        assert_eq!(as_i32(&*sublist_get(&arr, -1)), [Some(3), Some(5), Some(6)]);
        // An index past the shortest element lands on nothing there.
        assert_eq!(as_i32(&*sublist_get(&arr, 1)), [Some(2), Some(5), None]);
        assert_eq!(as_i32(&*sublist_get(&arr, 3)), [None, None, None]);

        assert!(!index_is_oob(&arr, 0));
        assert!(index_is_oob(&arr, 1));
    }

    /// A null element holds no list to index, and an empty one holds no value to land on.
    #[test]
    fn null_and_empty_elements_read_as_null() {
        let arr = PlListArray::new(
            PlPrimitiveArray::from_vec(vec![1i32, 2, 3]).into_boxed(),
            Buffer::from(vec![0u64, 2, 2, 3]),
            3,
            Some([true, true, false].into_iter().collect()),
        );

        assert_eq!(as_i32(&*sublist_get(&arr, 0)), [Some(1), None, None]);
        // The elements the index falls outside of are the empty one and the null one, and a null
        // element is not one it can fall outside of.
        assert!(index_is_oob(&arr, 0));
    }

    /// A chunk whose offsets hold a single range is indexed in the one element every element of it
    /// stands for, and the value read out is repeated rather than written out.
    #[test]
    fn a_repeated_element_is_indexed_once() {
        let arr = PlListArray::new_scalar(
            PlPrimitiveArray::from_vec(vec![1i32, 2, 3]).into_boxed(),
            100,
        );
        assert!(arr.offsets_are_scalar());

        let out = sublist_get(&arr, 1);
        assert!(out.is_scalar());
        assert_eq!(
            out.as_any().downcast_ref::<PlPrimitiveArray<i32>>(),
            Some(&PlPrimitiveArray::new_scalar(2i32, 100)),
        );

        // The one range every element covers decides whether the index falls outside of it.
        assert!(!index_is_oob(&arr, 2));
        assert!(index_is_oob(&arr, 3));

        // An index that lands on nothing lands on nothing at every element.
        assert_eq!(sublist_get(&arr, 3).null_count(), 100);

        // A mask that marks every element null leaves no list to index at all.
        let all_null = arr
            .clone()
            .with_validity(Some(PlBitmap::new_scalar(false, 100)));
        assert_eq!(sublist_get(&all_null, 1).null_count(), 100);
        assert!(!index_is_oob(&all_null, 3));
    }

    /// The one range scalar offsets hold still stands for every element under a mask that holds a
    /// bit per element, and it is that mask alone that decides which of them are null.
    #[test]
    fn a_repeated_element_under_a_flat_mask() {
        let arr =
            PlListArray::new_scalar(PlPrimitiveArray::from_vec(vec![1i32, 2]).into_boxed(), 3)
                .with_validity(Some([true, false, true].into_iter().collect()));

        assert_eq!(as_i32(&*sublist_get(&arr, 0)), [Some(1), None, Some(1)]);
        assert!(!index_is_oob(&arr, 1));
        assert!(index_is_oob(&arr, 2));
    }

    /// Wrapping every element in a list of its own keeps a repeated element repeated: one list of
    /// that value stands for all of them.
    #[test]
    fn a_repeated_value_wraps_into_a_repeated_list() {
        let unit = array_to_unit_list(PlPrimitiveArray::new_scalar(7i32, 100).into_boxed());

        assert!(unit.is_scalar());
        assert_eq!(unit.len(), 100);
        assert_eq!(unit.values().len(), 1);
        assert_eq!(as_i32(&*unit.value(42)), [Some(7)]);
    }

    #[test]
    fn every_value_wraps_into_a_list_of_its_own() {
        let unit = array_to_unit_list(PlPrimitiveArray::from_iter([Some(1i32), None]).into_boxed());

        assert_eq!(unit.len(), 2);
        assert_eq!(as_i32(&*unit.value(0)), [Some(1)]);
        assert_eq!(as_i32(&*unit.value(1)), [None]);

        let empty = array_to_unit_list(PlPrimitiveArray::<i32>::new_empty().into_boxed());
        assert!(empty.is_empty());
    }
}
