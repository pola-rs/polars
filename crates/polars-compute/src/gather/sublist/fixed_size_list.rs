//! The `arr.get` kernels over the arrays of `polars-array`.
//!
//! Every element of a fixed size list array holds the same number of values, so an index lands at
//! the same position within all of them: whether it is in bounds is decided once for the whole
//! array rather than element by element. Values that hold the single element every element repeats
//! are indexed in that one element, and the value read out of it stands for the whole result in
//! turn.

use arrow::legacy::index::IndexToUsize;
use polars_array::bitmap::combine_validities_and;
use polars_array::{PlArray, PlBitmap, PlFixedSizeListArray, PlPrimitiveArray};
use polars_error::{PolarsResult, polars_bail};
use polars_utils::IdxSize;

use crate::gather::take_unchecked;

/// The position `index` picks out of a list of `width` values, or `None` if it falls outside it.
///
/// A negative index counts back from the end of the list.
#[inline]
fn position_in(index: i64, width: usize) -> Option<usize> {
    index.negative_to_usize(width)
}

/// Returns the value at `index` within every element of `arr`.
///
/// An index that falls outside the width of the elements reads as a null value if `null_on_oob`,
/// and is an error otherwise.
pub fn sub_fixed_size_list_get_literal(
    arr: &PlFixedSizeListArray,
    index: i64,
    null_on_oob: bool,
) -> PolarsResult<Box<dyn PlArray>> {
    if arr.is_empty() {
        return Ok(arr.values().sliced(0, 0));
    }

    // Every element is `width` values wide, so the index falls either within all of them or within
    // none: it is resolved once, and an out of bounds one is answered without a value being read.
    let Some(offset) = position_in(index, arr.width()) else {
        if !null_on_oob {
            polars_bail!(ComputeError: "get index is out of bounds");
        }
        return Ok(arr.values().full_null_like(arr.len()));
    };

    // Values that hold the single element every element of `arr` repeats are indexed in place: the
    // value at `offset` within that one element is the answer at every element in turn, in `O(1)`.
    if let Some(values) = arr.scalar_values() {
        // SAFETY: `offset` is within the width, which is how many values the one element holds.
        return Ok(unsafe { values.new_from_index_unchecked(offset, arr.len()) });
    }

    let indices = (0..arr.len())
        // SAFETY: `i` is an element of `arr`.
        .map(|i| (unsafe { arr.value_range_unchecked(i) }.start + offset) as IdxSize)
        .collect::<Vec<_>>();

    // SAFETY: every index lands within the element it is read for.
    Ok(unsafe { take_unchecked(arr.values(), &PlPrimitiveArray::from_vec(indices)) })
}

/// Returns the value at the index `index` holds for it within every element of `arr`.
///
/// An index that falls outside the width of the elements, and one that is null, read as a null
/// value if `null_on_oob`, and are an error otherwise.
///
/// # Panics
/// Panics unless `arr` and `index` hold the same number of elements.
pub fn sub_fixed_size_list_get(
    arr: &PlFixedSizeListArray,
    index: &PlPrimitiveArray<i64>,
    null_on_oob: bool,
) -> PolarsResult<Box<dyn PlArray>> {
    assert_eq!(
        arr.len(),
        index.len(),
        "`arr.get` reads one index per element of the array it indexes",
    );

    if arr.is_empty() {
        return Ok(arr.values().sliced(0, 0));
    }

    // Indices stored in the scalar representation are one index shared by every element, which
    // lands at the same position within all of them: it is resolved once, like a literal one.
    if let Some(value) = index.scalar_values() {
        let out = sub_fixed_size_list_get_literal(arr, value, null_on_oob)?;

        // An index that is null picks out no value at all, which is the null an out of bounds one
        // reads as in turn.
        let Some(validity) = index.validity() else {
            return Ok(out);
        };
        if !null_on_oob && validity.unset_bits() > 0 {
            polars_bail!(ComputeError: "get index is out of bounds");
        }

        let validity = combine_validities_and(out.validity(), Some(validity));
        return Ok(out.with_validity_broadcast(validity.map(PlBitmap::into_flat_or_scalar)));
    }

    let width = arr.width();
    let mut out_of_bounds = false;
    let indices = index
        .iter()
        .enumerate()
        .map(|(i, index)| {
            let position = index.and_then(|index| position_in(index, width));
            out_of_bounds |= position.is_none();

            position.map(|position| {
                // SAFETY: `i` is an element of `arr`, which holds as many as `index`.
                (unsafe { arr.value_range_unchecked(i) }.start + position) as IdxSize
            })
        })
        .collect::<PlPrimitiveArray<IdxSize>>();

    if !null_on_oob && out_of_bounds {
        polars_bail!(ComputeError: "get index is out of bounds");
    }

    // SAFETY: every index lands within the element it is read for.
    Ok(unsafe { take_unchecked(arr.values(), &indices) })
}

#[cfg(test)]
mod test {
    use polars_array::StaticArray;

    use super::*;

    /// `[[1, 2], [3, 4], [5, 6]]`.
    fn get_array() -> PlFixedSizeListArray {
        PlFixedSizeListArray::from_values(
            PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5, 6]).into_boxed(),
            2,
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

    fn get_literal(arr: &PlFixedSizeListArray, index: i64) -> Vec<Option<i32>> {
        as_i32(&*sub_fixed_size_list_get_literal(arr, index, true).unwrap())
    }

    #[test]
    fn every_element_is_indexed() {
        let arr = get_array();

        assert_eq!(get_literal(&arr, 0), [Some(1), Some(3), Some(5)]);
        assert_eq!(get_literal(&arr, 1), [Some(2), Some(4), Some(6)]);
        assert_eq!(get_literal(&arr, -1), [Some(2), Some(4), Some(6)]);
        assert_eq!(get_literal(&arr, -2), [Some(1), Some(3), Some(5)]);
    }

    /// The width bounds every element alike, so an index outside it is outside all of them: the
    /// answer is reached without a value being read.
    #[test]
    fn an_out_of_bounds_index_is_resolved_once() {
        let arr = get_array();

        assert_eq!(get_literal(&arr, 2), [None, None, None]);
        assert_eq!(get_literal(&arr, -3), [None, None, None]);
        assert!(sub_fixed_size_list_get_literal(&arr, 2, false).is_err());
    }

    /// Values that hold the one element every element repeats are indexed in that element, and the
    /// value read out of it is repeated rather than written out.
    #[test]
    fn a_repeated_element_is_indexed_once() {
        let element = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]).into_boxed();
        let scalar = PlFixedSizeListArray::new_scalar(element.clone(), 100);
        assert!(scalar.values_are_scalar());
        let flat = scalar.to_flat().into_owned();

        for index in [0, 1, 2, -1, -3] {
            let out = sub_fixed_size_list_get_literal(&scalar, index, true).unwrap();
            assert!(out.is_scalar(), "index {index} of {scalar:?}");
            assert_eq!(
                as_i32(&*out),
                get_literal(&flat, index),
                "index {index} of {scalar:?}",
            );
        }

        // An index the one element does not reach is one no element reaches.
        assert_eq!(
            sub_fixed_size_list_get_literal(&scalar, 3, true)
                .unwrap()
                .null_count(),
            100
        );
        assert_eq!(get_literal(&flat, 3), vec![None; 100]);
    }

    /// An index stored in the scalar representation is one index shared by every element, which
    /// lands at the same position within all of them.
    #[test]
    fn a_repeated_index_is_resolved_like_a_literal() {
        let arr = get_array();
        let scalar = PlPrimitiveArray::new_scalar(-1i64, 3);
        let flat = PlPrimitiveArray::from_vec(vec![-1i64; 3]);

        let out = sub_fixed_size_list_get(&arr, &scalar, true).unwrap();
        assert_eq!(as_i32(&*out), [Some(2), Some(4), Some(6)]);
        assert_eq!(
            as_i32(&*out),
            as_i32(&*sub_fixed_size_list_get(&arr, &flat, true).unwrap()),
        );
    }

    /// A null index picks out no value at all, and reads as an out of bounds one does in turn.
    #[test]
    fn a_null_index_reads_as_null() {
        let arr = get_array();
        let index = PlPrimitiveArray::from_iter([Some(0i64), None, Some(5)]);

        assert_eq!(
            as_i32(&*sub_fixed_size_list_get(&arr, &index, true).unwrap()),
            [Some(1), None, None],
        );
        assert!(sub_fixed_size_list_get(&arr, &index, false).is_err());

        // A repeated index that is null says the same of every element.
        let all_null = PlPrimitiveArray::<i64>::new_full_null(3);
        assert_eq!(
            sub_fixed_size_list_get(&arr, &all_null, true)
                .unwrap()
                .null_count(),
            3,
        );
        assert!(sub_fixed_size_list_get(&arr, &all_null, false).is_err());
    }

    /// Both representations of a repeated element are indexed alike by an index laid out one per
    /// element.
    #[test]
    fn a_repeated_element_under_a_flat_index() {
        let element = PlPrimitiveArray::from_vec(vec![1i32, 2]).into_boxed();
        let scalar = PlFixedSizeListArray::new_scalar(element, 3);
        let flat = scalar.to_flat().into_owned();
        let index = PlPrimitiveArray::from_iter([Some(1i64), None, Some(-2)]);

        let out = sub_fixed_size_list_get(&scalar, &index, true).unwrap();
        assert_eq!(as_i32(&*out), [Some(2), None, Some(1)]);
        assert_eq!(
            as_i32(&*out),
            as_i32(&*sub_fixed_size_list_get(&flat, &index, true).unwrap()),
        );
    }

    #[test]
    fn an_empty_array_is_indexed_into_nothing() {
        let arr =
            PlFixedSizeListArray::new_empty(PlPrimitiveArray::<i32>::new_empty().into_boxed(), 2);

        assert!(
            sub_fixed_size_list_get_literal(&arr, 0, false)
                .unwrap()
                .is_empty()
        );
        // The width is never reached, so an index outside it is not an error either.
        assert!(
            sub_fixed_size_list_get_literal(&arr, 7, false)
                .unwrap()
                .is_empty()
        );
    }
}
