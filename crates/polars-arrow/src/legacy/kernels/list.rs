use crate::array::growable::make_growable;
use crate::array::{Array, ArrayRef, ListArray};
use crate::bitmap::BitmapBuilder;
use crate::compute::utils::combine_validities_and;
use crate::legacy::prelude::*;
use crate::legacy::trusted_len::TrustedLenPush;
use crate::offset::{Offsets, OffsetsBuffer};

pub fn sublist_get(arr: &ListArray<i64>, index: i64) -> ArrayRef {
    let values = arr.values();

    let mut growable = make_growable(&[values.as_ref()], values.validity().is_some(), arr.len());
    let mut result_validity = BitmapBuilder::with_capacity(arr.len());
    let opt_outer_validity = arr.validity();
    let index = usize::try_from(index).unwrap();

    for (outer_idx, x) in arr.offsets().windows(2).enumerate() {
        let [i, j] = x else { unreachable!() };
        let i = usize::try_from(*i).unwrap();
        let j = usize::try_from(*j).unwrap();

        let (offset, len) = (i, j - i);

        let idx_is_oob = index >= len;
        let outer_is_valid =
            opt_outer_validity.map_or(true, |x| unsafe { x.get_bit_unchecked(outer_idx) });

        unsafe {
            if idx_is_oob {
                growable.extend_validity(1);
            } else {
                growable.extend(0, offset + index, 1);
            }

            result_validity.push_unchecked(!idx_is_oob & outer_is_valid);
        }
    }

    let values = growable.as_box();

    values.with_validity(combine_validities_and(
        Some(&result_validity.freeze()),
        values.validity(),
    ))
}

/// Check if an index is out of bounds for at least one sublist.
pub fn index_is_oob(arr: &ListArray<i64>, index: i64) -> bool {
    if arr.null_count() == 0 {
        arr.offsets()
            .lengths()
            .any(|len| index.negative_to_usize(len).is_none())
    } else {
        arr.offsets()
            .lengths()
            .zip(arr.validity().unwrap())
            .any(|(len, valid)| {
                if valid {
                    index.negative_to_usize(len).is_none()
                } else {
                    // skip nulls
                    false
                }
            })
    }
}

/// Convert a list `[1, 2, 3]` to a list type of `[[1], [2], [3]]`
pub fn array_to_unit_list(array: ArrayRef) -> ListArray<i64> {
    let len = array.len();
    let mut offsets = Vec::with_capacity(len + 1);
    // SAFETY: we allocated enough
    unsafe {
        offsets.push_unchecked(0i64);

        for _ in 0..len {
            offsets.push_unchecked(offsets.len() as i64)
        }
    };

    // SAFETY:
    // offsets are monotonically increasing
    unsafe {
        let offsets: OffsetsBuffer<i64> = Offsets::new_unchecked(offsets).into();
        let dtype = ListArray::<i64>::default_datatype(array.dtype().clone());
        ListArray::<i64>::new(dtype, offsets, array, None)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::array::{Int32Array, PrimitiveArray};
    use crate::datatypes::ArrowDataType;

    fn get_array() -> ListArray<i64> {
        let values = Int32Array::from_slice([1, 2, 3, 4, 5, 6]);
        let offsets = OffsetsBuffer::try_from(vec![0i64, 3, 5, 6]).unwrap();

        let dtype = ListArray::<i64>::default_datatype(ArrowDataType::Int32);
        ListArray::<i64>::new(dtype, offsets, Box::new(values), None)
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
