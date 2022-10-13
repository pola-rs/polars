use arrow::array::ListArray;
use arrow::buffer::Buffer;

use crate::compute::take::take_unchecked;
use crate::prelude::*;
use crate::trusted_len::PushUnchecked;
use crate::utils::CustomIterTools;

/// Get the indices that would result in a get operation on the lists values.
/// for example, consider this list:
/// ```text
/// [[1, 2, 3],
///  [4, 5],
///  [6]]
///
///  This contains the following values array:
/// [1, 2, 3, 4, 5, 6]
///
/// get index 0
/// would lead to the following indexes:
///     [0, 3, 5].
/// if we use those in a take operation on the values array we get:
///     [1, 4, 6]
///
///
/// get index -1
/// would lead to the following indexes:
///     [2, 4, 5].
/// if we use those in a take operation on the values array we get:
///     [3, 5, 6]
///
/// ```
fn sublist_get_indexes(arr: &ListArray<i64>, index: i64) -> IdxArr {
    let offsets = arr.offsets().as_slice();
    let mut iter = offsets.iter();

    // the indices can be sliced, so we should not start at 0.
    let mut cum_offset = (*offsets.first().unwrap_or(&0)) as IdxSize;

    if let Some(mut previous) = iter.next().copied() {
        let a: IdxArr = iter
            .map(|&offset| {
                let len = offset - previous;
                previous = offset;
                // make sure that empty lists don't get accessed
                // and out of bounds return null
                if len == 0 {
                    return None;
                }
                if index >= len {
                    cum_offset += len as IdxSize;
                    return None;
                }

                let out = index
                    .negative_to_usize(len as usize)
                    .map(|idx| idx as IdxSize + cum_offset);
                cum_offset += len as IdxSize;
                out
            })
            .collect_trusted();

        a
    } else {
        IdxArr::from_slice([])
    }
}

pub fn sublist_get(arr: &ListArray<i64>, index: i64) -> ArrayRef {
    let take_by = sublist_get_indexes(arr, index);
    let values = arr.values();
    // Safety:
    // the indices we generate are in bounds
    unsafe { take_unchecked(&**values, &take_by) }
}

/// Convert a list `[1, 2, 3]` to a list type of `[[1], [2], [3]]`
pub fn array_to_unit_list(array: ArrayRef) -> ListArray<i64> {
    let len = array.len();
    let mut offsets = Vec::with_capacity(len + 1);
    // Safety: we allocated enough
    unsafe {
        offsets.push_unchecked(0i64);

        for _ in 0..len {
            offsets.push_unchecked(offsets.len() as i64)
        }
    };

    let offsets: Buffer<i64> = offsets.into();
    let dtype = ListArray::<i64>::default_datatype(array.data_type().clone());
    // Safety:
    // offsets are monotonically increasing
    unsafe { ListArray::<i64>::new_unchecked(dtype, offsets, array, None) }
}

#[cfg(test)]
mod test {
    use arrow::array::{Array, Int32Array, PrimitiveArray};
    use arrow::buffer::Buffer;
    use arrow::datatypes::DataType;

    use super::*;

    fn get_array() -> ListArray<i64> {
        let values = Int32Array::from_slice(&[1, 2, 3, 4, 5, 6]);
        let offsets = Buffer::from(vec![0i64, 3, 5, 6]);

        let dtype = ListArray::<i64>::default_datatype(DataType::Int32);
        ListArray::<i64>::from_data(dtype, offsets, Box::new(values), None)
    }

    #[test]
    fn test_sublist_get_indexes() {
        let arr = get_array();
        let out = sublist_get_indexes(&arr, 0);
        assert_eq!(out.values().as_slice(), &[0, 3, 5]);
        let out = sublist_get_indexes(&arr, -1);
        assert_eq!(out.values().as_slice(), &[2, 4, 5]);
        let out = sublist_get_indexes(&arr, 3);
        assert_eq!(out.null_count(), 3);

        let values = Int32Array::from_iter([
            Some(1),
            Some(1),
            Some(3),
            Some(4),
            Some(5),
            Some(6),
            Some(7),
            Some(8),
            Some(9),
            None,
            Some(11),
        ]);
        let offsets = Buffer::from(vec![0i64, 1, 2, 3, 6, 9, 11]);

        let dtype = ListArray::<i64>::default_datatype(DataType::Int32);
        let arr = ListArray::<i64>::from_data(dtype, offsets, Box::new(values), None);

        let out = sublist_get_indexes(&arr, 1);
        assert_eq!(
            out.into_iter()
                .map(|opt_v| opt_v.cloned())
                .collect::<Vec<_>>(),
            &[None, None, None, Some(4), Some(7), Some(10)]
        );
    }

    #[test]
    fn test_sublist_get() {
        let arr = get_array();

        let out = sublist_get(&arr, 0);
        let out = out.as_any().downcast_ref::<PrimitiveArray<i32>>().unwrap();

        assert_eq!(out.values().as_slice(), &[1, 4, 6]);
        let out = sublist_get(&arr, -1);
        let out = out.as_any().downcast_ref::<PrimitiveArray<i32>>().unwrap();
        assert_eq!(out.values().as_slice(), &[3, 5, 6]);
    }
}
