use polars_arrow::array::builder::{ShareStrategy, make_builder};
use polars_arrow::array::{Array, ArrayRef, ListArray, PrimitiveArray, StaticArray};
use polars_arrow::bitmap::BitmapBuilder;
use polars_arrow::datatypes::IdxArr;
use polars_arrow::legacy::index::IndexToUsize;
use polars_arrow::legacy::trusted_len::TrustedLenPush;
use polars_arrow::legacy::utils::CustomIterTools;
use polars_arrow::offset::{Offsets, OffsetsBuffer};
use polars_error::{PolarsResult, polars_ensure};
use polars_utils::IdxSize;
use polars_utils::slice::slice_offsets;

use crate::gather::take_unchecked;

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
        if arr.null_count() == 0 {
            iter.map(|&offset| {
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
            .collect_trusted()
        } else {
            // we can ensure that validity is not none as we have null value.
            let validity = arr.validity().unwrap();
            iter.enumerate()
                .map(|(i, &offset)| {
                    let len = offset - previous;
                    previous = offset;
                    // make sure that empty and null lists don't get accessed and return null.
                    // SAFETY, we are within bounds
                    if len == 0 || !unsafe { validity.get_bit_unchecked(i) } {
                        cum_offset += len as IdxSize;
                        return None;
                    }

                    // make sure that out of bounds return null
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
                .collect_trusted()
        }
    } else {
        IdxArr::from_slice([])
    }
}

pub fn sublist_get(arr: &ListArray<i64>, index: i64) -> ArrayRef {
    let take_by = sublist_get_indexes(arr, index);
    let values = arr.values();
    // SAFETY:
    // the indices we generate are in bounds
    unsafe { take_unchecked(&**values, &take_by) }
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

/// Slice list `i` of `arr` with offset `offsets[i * offset_stride]` and length
/// `lengths[i * lengths_stride]`.
///
/// Also returns whether all output lists are valid and non-empty.
pub fn sublist_slice_var(
    arr: &ListArray<i64>,
    offsets: &[i64],
    offset_stride: usize,
    lengths: &[usize],
    lengths_stride: usize,
) -> (ListArray<i64>, bool) {
    let mut fast_explode = arr.null_count() == 0;
    let in_offsets = arr.offsets().as_slice();
    let mut out_offsets = Vec::with_capacity(arr.len() + 1);
    out_offsets.push(0i64);
    // Ranges in `values` to copy, with adjacent ranges merged.
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut unchanged = true;
    for i in 0..arr.len() {
        let list_start = in_offsets[i] as usize;
        let list_len = (in_offsets[i + 1] - in_offsets[i]) as usize;
        let (start, len) = if arr.is_null(i) {
            (list_start, 0)
        } else {
            let (start, len) = slice_offsets(
                offsets[i * offset_stride],
                lengths[i * lengths_stride],
                list_len,
            );
            fast_explode &= len > 0;
            (list_start + start, len)
        };

        unchanged &= start == list_start && len == list_len;
        out_offsets.push(out_offsets.last().unwrap() + len as i64);
        match ranges.last_mut() {
            Some((s, l)) if *s + *l == start => *l += len,
            _ if len > 0 => ranges.push((start, len)),
            _ => {},
        }
    }

    if unchanged {
        return (arr.clone(), fast_explode);
    }

    let values = arr.values().as_ref();
    let mut builder = make_builder(values.dtype());
    builder.reserve(*out_offsets.last().unwrap() as usize);
    for (start, len) in ranges {
        builder.subslice_extend(values, start, len, ShareStrategy::Always);
    }
    // SAFETY: the offsets are monotonically increasing by construction.
    let out_offsets = unsafe { Offsets::new_unchecked(out_offsets) };
    let out = ListArray::new(
        arr.dtype().clone(),
        out_offsets.into(),
        builder.freeze_reset(),
        arr.validity().cloned(),
    );
    (out, fast_explode)
}

/// Gather from list `i * list_stride` of `lists` the elements at the indices in
/// list `i * idx_stride` of `idxs`, for `i` in `0..len`. `idxs` must have `i64`
/// values. Negative indices count from the end. Out-of-bounds indices give null
/// if `null_on_oob`, and an error otherwise.
///
/// Also returns whether all output lists are valid and non-empty.
pub fn sublist_gather(
    lists: &ListArray<i64>,
    list_stride: usize,
    idxs: &ListArray<i64>,
    idx_stride: usize,
    len: usize,
    null_on_oob: bool,
) -> PolarsResult<(ListArray<i64>, bool)> {
    let list_offsets = lists.offsets().as_slice();
    let idx_offsets = idxs.offsets().as_slice();
    let idx_values: &PrimitiveArray<i64> = idxs.values().as_any().downcast_ref().unwrap();
    let mut out_offsets = Vec::with_capacity(len + 1);
    out_offsets.push(0i64);
    let mut validity = BitmapBuilder::with_capacity(len);
    // Indices into `lists.values()`.
    let mut take_idxs: Vec<IdxSize> = Vec::new();
    let mut take_validity = BitmapBuilder::new();
    let mut fast_explode = true;
    for row in 0..len {
        let l = row * list_stride;
        let i = row * idx_stride;
        let valid = lists.is_valid(l) && idxs.is_valid(i);
        validity.push(valid);
        let (idx_start, idx_end) = if valid {
            (idx_offsets[i] as usize, idx_offsets[i + 1] as usize)
        } else {
            (0, 0)
        };
        fast_explode &= idx_start < idx_end;

        let list_start = list_offsets[l];
        let list_len = list_offsets[l + 1] - list_start;
        for j in idx_start..idx_end {
            let pos = match idx_values.get(j) {
                None => None,
                Some(v) => {
                    let v = if v < 0 { v + list_len } else { v };
                    let pos = (0..list_len).contains(&v).then_some(v);
                    polars_ensure!(
                        pos.is_some() || null_on_oob,
                        OutOfBounds: "gather indices are out of bounds"
                    );
                    pos
                },
            };
            take_idxs.push(pos.map_or(0, |p| (list_start + p) as IdxSize));
            take_validity.push(pos.is_some());
        }
        out_offsets.push(take_idxs.len() as i64);
    }

    let take_idxs = IdxArr::from_vec(take_idxs).with_validity(take_validity.into_opt_validity());
    // SAFETY: the valid indices are in bounds of `lists.values()`.
    let values = unsafe { take_unchecked(lists.values().as_ref(), &take_idxs) };
    // SAFETY: the offsets are monotonically increasing by construction.
    let out_offsets = unsafe { Offsets::new_unchecked(out_offsets) };
    let out = ListArray::new(
        lists.dtype().clone(),
        out_offsets.into(),
        values,
        validity.into_opt_validity(),
    );
    Ok((out, fast_explode))
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
    use polars_arrow::array::{Int32Array, PrimitiveArray};
    use polars_arrow::datatypes::ArrowDataType;

    use super::*;

    fn get_array() -> ListArray<i64> {
        let values = Int32Array::from_slice([1, 2, 3, 4, 5, 6]);
        let offsets = OffsetsBuffer::try_from(vec![0i64, 3, 5, 6]).unwrap();

        let dtype = ListArray::<i64>::default_datatype(ArrowDataType::Int32);
        ListArray::<i64>::new(dtype, offsets, Box::new(values), None)
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
        let offsets = OffsetsBuffer::try_from(vec![0i64, 1, 2, 3, 6, 9, 11]).unwrap();

        let dtype = ListArray::<i64>::default_datatype(ArrowDataType::Int32);
        let arr = ListArray::<i64>::new(dtype, offsets, Box::new(values), None);

        let out = sublist_get_indexes(&arr, 1);
        assert_eq!(
            out.into_iter().collect::<Vec<_>>(),
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
