use crate::index::IndexToUsize;
use crate::kernels::take::take_unchecked;
use crate::trusted_len::PushUnchecked;
use crate::utils::CustomIterTools;
use arrow::array::{ArrayRef, ListArray, PrimitiveArray};
use arrow::buffer::{Buffer, MutableBuffer};

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
fn sublist_get_indexes(arr: &ListArray<i64>, index: i64) -> PrimitiveArray<u32> {
    let mut iter = arr.offsets().iter();

    let mut cum_offset = 0u32;

    if let Some(mut previous) = iter.next().copied() {
        let a: PrimitiveArray<u32> = iter
            .map(|&offset| {
                let len = offset - previous;
                previous = offset;

                let out = index
                    .to_usize(len as usize)
                    .map(|idx| idx as u32 + cum_offset);
                cum_offset += len as u32;
                out
            })
            .collect_trusted();

        a
    } else {
        PrimitiveArray::<u32>::from_slice(&[])
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

    let offsets: Buffer<i64> = MutableBuffer::from_vec(offsets).into();
    let dtype = ListArray::<i64>::default_datatype(array.data_type().clone());
    ListArray::<i64>::from_data(dtype, offsets, array, None)
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::buffer::Buffer;
    use arrow::datatypes::DataType;
    use std::sync::Arc;

    fn get_array() -> ListArray<i64> {
        let values = Int32Array::from_slice(&[1, 2, 3, 4, 5, 6]);
        let offsets = Buffer::from(&[0i64, 3, 5, 6]);

        let dtype = ListArray::<i64>::default_datatype(DataType::Int32);
        ListArray::<i64>::from_data(dtype, offsets, Arc::new(values), None)
    }

    #[test]
    fn test_sublist_get_indexes() {
        let arr = get_array();
        let out = sublist_get_indexes(&arr, 0);
        assert_eq!(out.values().as_slice(), &[0, 3, 5]);
        let out = sublist_get_indexes(&arr, -1);
        assert_eq!(out.values().as_slice(), &[2, 4, 5]);
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
