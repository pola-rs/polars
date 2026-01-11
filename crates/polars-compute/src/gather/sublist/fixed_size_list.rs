use arrow::array::{Array, ArrayRef, FixedSizeListArray, ListArray, PrimitiveArray};
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::datatypes::ArrowDataType;
use arrow::legacy::prelude::*;
use arrow::legacy::utils::CustomIterTools;
use arrow::offset::{Offsets, OffsetsBuffer};
use polars_error::{PolarsResult, polars_bail};
use polars_utils::IdxSize;

use crate::gather::take_unchecked;

/// Converts an index value to an array index, handling negative indexing.
trait ToArrayIndex: Copy {
    fn to_array_index(self, len: usize) -> Option<usize>;
}

macro_rules! impl_to_array_index {
    (signed: $($t:ty),*) => {
        $(
            impl ToArrayIndex for $t {
                #[inline]
                fn to_array_index(self, len: usize) -> Option<usize> {
                    (self as i64).negative_to_usize(len)
                }
            }
        )*
    };
    (unsigned: $($t:ty),*) => {
        $(
            impl ToArrayIndex for $t {
                #[inline]
                fn to_array_index(self, len: usize) -> Option<usize> {
                    let idx = self as usize;
                    (idx < len).then_some(idx)
                }
            }
        )*
    };
}

impl_to_array_index!(signed: i8, i16, i32, i64);
impl_to_array_index!(unsigned: u8, u16, u32, u64);

fn sub_fixed_size_list_get_indexes_literal(width: usize, len: usize, index: i64) -> IdxArr {
    (0..len)
        .map(|i| {
            if index >= width as i64 {
                return None;
            }
            index
                .negative_to_usize(width)
                .map(|idx| (idx + i * width) as IdxSize)
        })
        .collect_trusted()
}

fn sub_fixed_size_list_get_indexes(width: usize, index: &PrimitiveArray<i64>) -> IdxArr {
    index
        .iter()
        .enumerate()
        .map(|(i, idx)| {
            idx.and_then(|&idx| {
                if idx >= width as i64 {
                    return None;
                }
                idx.negative_to_usize(width)
                    .map(|idx| (idx + i * width) as IdxSize)
            })
        })
        .collect_trusted()
}

pub fn sub_fixed_size_list_get_literal(
    arr: &FixedSizeListArray,
    index: i64,
    null_on_oob: bool,
) -> PolarsResult<ArrayRef> {
    let take_by = sub_fixed_size_list_get_indexes_literal(arr.size(), arr.len(), index);
    if !null_on_oob && take_by.has_nulls() {
        polars_bail!(ComputeError: "get index is out of bounds");
    }
    // SAFETY: the indices we generate are in bounds
    unsafe { Ok(take_unchecked(&**arr.values(), &take_by)) }
}

pub fn sub_fixed_size_list_get(
    arr: &FixedSizeListArray,
    index: &PrimitiveArray<i64>,
    null_on_oob: bool,
) -> PolarsResult<ArrayRef> {
    let take_by = sub_fixed_size_list_get_indexes(arr.size(), index);
    if !null_on_oob && take_by.has_nulls() {
        polars_bail!(ComputeError: "get index is out of bounds");
    }
    // SAFETY: the indices we generate are in bounds
    unsafe { Ok(take_unchecked(&**arr.values(), &take_by)) }
}

/// # Safety
/// Caller must ensure indices are within bounds of the source array,
/// or that `null_on_oob` is true.
unsafe fn take_no_validity<T, O>(
    width: usize,
    arr_len: usize,
    indices_offsets: &[O],
    indices_values: &PrimitiveArray<T>,
    null_on_oob: bool,
) -> PolarsResult<(IdxArr, OffsetsBuffer<O>)>
where
    T: arrow::types::NativeType + ToArrayIndex,
    O: arrow::offset::Offset,
{
    let total_indices = indices_offsets.last().unwrap().to_usize();
    let mut flat_indices: Vec<Option<IdxSize>> = Vec::with_capacity(total_indices);
    let mut offsets: Vec<O> = Vec::with_capacity(arr_len + 1);
    offsets.push(O::zero());

    for row in 0..arr_len {
        let idx_start = indices_offsets[row].to_usize();
        let idx_end = indices_offsets[row + 1].to_usize();

        for idx_pos in idx_start..idx_end {
            let idx_valid = indices_values
                .validity()
                .as_ref()
                .is_none_or(|v| v.get_bit_unchecked(idx_pos));

            if !idx_valid {
                flat_indices.push(None);
                continue;
            }

            let idx_val = indices_values.value(idx_pos);
            match idx_val.to_array_index(width) {
                Some(local_idx) => {
                    flat_indices.push(Some((row * width + local_idx) as IdxSize));
                },
                None if null_on_oob => flat_indices.push(None),
                None => polars_bail!(ComputeError:
                    "gather index is out of bounds for array of width {}", width
                ),
            }
        }
        offsets.push(O::from_usize(flat_indices.len()).unwrap());
    }

    let flat_indices_arr: IdxArr = flat_indices.into_iter().collect_trusted();
    let offsets_buffer: OffsetsBuffer<O> = Offsets::new_unchecked(offsets).into();
    Ok((flat_indices_arr, offsets_buffer))
}

/// # Safety
/// Caller must ensure indices are within bounds of the source array,
/// or that `null_on_oob` is true.
unsafe fn take_values_validity<T, O>(
    width: usize,
    arr_validity: &Bitmap,
    indices_offsets: &[O],
    indices_values: &PrimitiveArray<T>,
    null_on_oob: bool,
) -> PolarsResult<(IdxArr, OffsetsBuffer<O>, BitmapBuilder)>
where
    T: arrow::types::NativeType + ToArrayIndex,
    O: arrow::offset::Offset,
{
    let arr_len = arr_validity.len();
    let total_indices = indices_offsets.last().unwrap().to_usize();
    let mut flat_indices: Vec<Option<IdxSize>> = Vec::with_capacity(total_indices);
    let mut offsets: Vec<O> = Vec::with_capacity(arr_len + 1);
    offsets.push(O::zero());
    let mut validity = BitmapBuilder::with_capacity(arr_len);

    for row in 0..arr_len {
        if !arr_validity.get_bit_unchecked(row) {
            validity.push(false);
            offsets.push(*offsets.last().unwrap());
            continue;
        }
        validity.push(true);

        let idx_start = indices_offsets[row].to_usize();
        let idx_end = indices_offsets[row + 1].to_usize();

        for idx_pos in idx_start..idx_end {
            let idx_valid = indices_values
                .validity()
                .as_ref()
                .is_none_or(|v| v.get_bit_unchecked(idx_pos));

            if !idx_valid {
                flat_indices.push(None);
                continue;
            }

            let idx_val = indices_values.value(idx_pos);
            match idx_val.to_array_index(width) {
                Some(local_idx) => {
                    flat_indices.push(Some((row * width + local_idx) as IdxSize));
                },
                None if null_on_oob => flat_indices.push(None),
                None => polars_bail!(ComputeError:
                    "gather index is out of bounds for array of width {}", width
                ),
            }
        }
        offsets.push(O::from_usize(flat_indices.len()).unwrap());
    }

    let flat_indices_arr: IdxArr = flat_indices.into_iter().collect_trusted();
    let offsets_buffer: OffsetsBuffer<O> = Offsets::new_unchecked(offsets).into();
    Ok((flat_indices_arr, offsets_buffer, validity))
}

/// # Safety
/// Caller must ensure indices are within bounds of the source array,
/// or that `null_on_oob` is true.
unsafe fn take_indices_validity<T, O>(
    width: usize,
    arr_len: usize,
    indices_validity: &Bitmap,
    indices_offsets: &[O],
    indices_values: &PrimitiveArray<T>,
    null_on_oob: bool,
) -> PolarsResult<(IdxArr, OffsetsBuffer<O>, BitmapBuilder)>
where
    T: arrow::types::NativeType + ToArrayIndex,
    O: arrow::offset::Offset,
{
    let total_indices = indices_offsets.last().unwrap().to_usize();
    let mut flat_indices: Vec<Option<IdxSize>> = Vec::with_capacity(total_indices);
    let mut offsets: Vec<O> = Vec::with_capacity(arr_len + 1);
    offsets.push(O::zero());
    let mut validity = BitmapBuilder::with_capacity(arr_len);

    for row in 0..arr_len {
        if !indices_validity.get_bit_unchecked(row) {
            validity.push(false);
            offsets.push(*offsets.last().unwrap());
            continue;
        }
        validity.push(true);

        let idx_start = indices_offsets[row].to_usize();
        let idx_end = indices_offsets[row + 1].to_usize();

        for idx_pos in idx_start..idx_end {
            // Handle inner indices validity if present
            let idx_valid = indices_values
                .validity()
                .as_ref()
                .is_none_or(|v| v.get_bit_unchecked(idx_pos));

            if !idx_valid {
                flat_indices.push(None);
                continue;
            }

            let idx_val = indices_values.value(idx_pos);
            match idx_val.to_array_index(width) {
                Some(local_idx) => {
                    flat_indices.push(Some((row * width + local_idx) as IdxSize));
                },
                None if null_on_oob => flat_indices.push(None),
                None => polars_bail!(ComputeError:
                    "gather index is out of bounds for array of width {}", width
                ),
            }
        }
        offsets.push(O::from_usize(flat_indices.len()).unwrap());
    }

    let flat_indices_arr: IdxArr = flat_indices.into_iter().collect_trusted();
    let offsets_buffer: OffsetsBuffer<O> = Offsets::new_unchecked(offsets).into();
    Ok((flat_indices_arr, offsets_buffer, validity))
}

/// # Safety
/// Caller must ensure indices are within bounds of the source array,
/// or that `null_on_oob` is true.
unsafe fn take_values_indices_validity<T, O>(
    width: usize,
    arr_validity: &Bitmap,
    indices_validity: &Bitmap,
    indices_offsets: &[O],
    indices_values: &PrimitiveArray<T>,
    null_on_oob: bool,
) -> PolarsResult<(IdxArr, OffsetsBuffer<O>, BitmapBuilder)>
where
    T: arrow::types::NativeType + ToArrayIndex,
    O: arrow::offset::Offset,
{
    let arr_len = arr_validity.len();
    let total_indices = indices_offsets.last().unwrap().to_usize();
    let mut flat_indices: Vec<Option<IdxSize>> = Vec::with_capacity(total_indices);
    let mut offsets: Vec<O> = Vec::with_capacity(arr_len + 1);
    offsets.push(O::zero());
    let mut validity = BitmapBuilder::with_capacity(arr_len);

    for row in 0..arr_len {
        let arr_valid = arr_validity.get_bit_unchecked(row);
        let idx_valid = indices_validity.get_bit_unchecked(row);

        if !arr_valid || !idx_valid {
            validity.push(false);
            offsets.push(*offsets.last().unwrap());
            continue;
        }
        validity.push(true);

        let idx_start = indices_offsets[row].to_usize();
        let idx_end = indices_offsets[row + 1].to_usize();

        for idx_pos in idx_start..idx_end {
            let inner_valid = indices_values
                .validity()
                .as_ref()
                .is_none_or(|v| v.get_bit_unchecked(idx_pos));

            if !inner_valid {
                flat_indices.push(None);
                continue;
            }

            let idx_val = indices_values.value(idx_pos);
            match idx_val.to_array_index(width) {
                Some(local_idx) => {
                    flat_indices.push(Some((row * width + local_idx) as IdxSize));
                },
                None if null_on_oob => flat_indices.push(None),
                None => polars_bail!(ComputeError:
                    "gather index is out of bounds for array of width {}", width
                ),
            }
        }
        offsets.push(O::from_usize(flat_indices.len()).unwrap());
    }

    let flat_indices_arr: IdxArr = flat_indices.into_iter().collect_trusted();
    let offsets_buffer: OffsetsBuffer<O> = Offsets::new_unchecked(offsets).into();
    Ok((flat_indices_arr, offsets_buffer, validity))
}

fn sub_fixed_size_list_gather_typed<T, O>(
    arr: &FixedSizeListArray,
    indices: &ListArray<O>,
    null_on_oob: bool,
) -> PolarsResult<ListArray<O>>
where
    T: arrow::types::NativeType + ToArrayIndex,
    O: arrow::offset::Offset,
{
    let width = arr.size();
    let values = arr.values();
    let indices_offsets = indices.offsets().as_slice();
    let indices_values = indices
        .values()
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .expect("indices inner array type mismatch");

    let arr_has_validity = arr.validity().is_some();
    let idx_has_validity = indices.validity().is_some();

    // SAFETY: we check bounds via to_array_index, null_on_oob, or error
    let (flat_indices_arr, offsets_buffer, validity) = unsafe {
        match (arr_has_validity, idx_has_validity) {
            (false, false) => {
                let (idx, off) = take_no_validity(
                    width,
                    arr.len(),
                    indices_offsets,
                    indices_values,
                    null_on_oob,
                )?;
                (idx, off, None)
            },
            (true, false) => {
                let (idx, off, val) = take_values_validity(
                    width,
                    arr.validity().unwrap(),
                    indices_offsets,
                    indices_values,
                    null_on_oob,
                )?;
                (idx, off, val.into_opt_validity())
            },
            (false, true) => {
                let (idx, off, val) = take_indices_validity(
                    width,
                    arr.len(),
                    indices.validity().unwrap(),
                    indices_offsets,
                    indices_values,
                    null_on_oob,
                )?;
                (idx, off, val.into_opt_validity())
            },
            (true, true) => {
                let (idx, off, val) = take_values_indices_validity(
                    width,
                    arr.validity().unwrap(),
                    indices.validity().unwrap(),
                    indices_offsets,
                    indices_values,
                    null_on_oob,
                )?;
                (idx, off, val.into_opt_validity())
            },
        }
    };

    // SAFETY: indices are validated above
    let gathered_values = unsafe { take_unchecked(&**values, &flat_indices_arr) };

    let dtype = ListArray::<O>::default_datatype(gathered_values.dtype().clone());
    Ok(ListArray::new(
        dtype,
        offsets_buffer,
        gathered_values,
        validity,
    ))
}

/// Gather multiple indices from each fixed-size list element.
pub fn sub_fixed_size_list_gather(
    arr: &FixedSizeListArray,
    indices: &ListArray<i64>,
    null_on_oob: bool,
) -> PolarsResult<ListArray<i64>> {
    use ArrowDataType::*;
    match indices.values().dtype() {
        Int8 => sub_fixed_size_list_gather_typed::<i8, i64>(arr, indices, null_on_oob),
        Int16 => sub_fixed_size_list_gather_typed::<i16, i64>(arr, indices, null_on_oob),
        Int32 => sub_fixed_size_list_gather_typed::<i32, i64>(arr, indices, null_on_oob),
        Int64 => sub_fixed_size_list_gather_typed::<i64, i64>(arr, indices, null_on_oob),
        UInt8 => sub_fixed_size_list_gather_typed::<u8, i64>(arr, indices, null_on_oob),
        UInt16 => sub_fixed_size_list_gather_typed::<u16, i64>(arr, indices, null_on_oob),
        UInt32 => sub_fixed_size_list_gather_typed::<u32, i64>(arr, indices, null_on_oob),
        UInt64 => sub_fixed_size_list_gather_typed::<u64, i64>(arr, indices, null_on_oob),
        dt => polars_bail!(ComputeError: "unsupported index dtype for arr.gather: {:?}", dt),
    }
}

#[cfg(test)]
mod test {
    use polars_utils::pl_str::PlSmallStr;

    use super::*;

    fn fixture_array() -> FixedSizeListArray {
        let values = PrimitiveArray::from_vec(vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9]);
        FixedSizeListArray::new(
            ArrowDataType::FixedSizeList(
                Box::new(arrow::datatypes::Field::new(
                    PlSmallStr::from_static("item"),
                    ArrowDataType::Int64,
                    true,
                )),
                3,
            ),
            3,
            Box::new(values),
            None,
        ) // [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    }

    fn make_indices<T: arrow::types::NativeType>(
        values: Vec<Option<T>>,
        offsets: Vec<i64>,
    ) -> ListArray<i64> {
        let arr = PrimitiveArray::<T>::from_iter(values);
        let offsets: OffsetsBuffer<i64> = unsafe { Offsets::new_unchecked(offsets).into() };
        ListArray::new(
            ListArray::<i64>::default_datatype(arr.dtype().clone()),
            offsets,
            Box::new(arr),
            None,
        )
    }

    /// Test gather within FixedSizeListArray elements: negative indices, inner null indices, and OOB handling.
    #[test]
    fn test_sub_fixed_size_list_gather() {
        let arr = fixture_array();

        // basic usage
        let indices = make_indices::<i64>(vec![Some(0), Some(2), Some(1)], vec![0, 1, 2, 3]); // [[0], [2], [1]]
        let result = sub_fixed_size_list_gather(&arr, &indices, false).unwrap();
        let out = result
            .values()
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();
        assert_eq!(out.values().as_slice(), &[1, 6, 8]);

        // negative index, inner null
        let indices = make_indices::<i64>(vec![Some(-1), None, Some(0)], vec![0, 1, 2, 3]); // [[-1], [None], [0]]
        let result = sub_fixed_size_list_gather(&arr, &indices, false).unwrap();
        let out = result
            .values()
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();
        assert_eq!(out.value(0), 3);
        assert!(!out.is_valid(1));
        assert_eq!(out.value(2), 7);

        // oob error
        let indices = make_indices::<i64>(vec![Some(10)], vec![0, 1, 1, 1]); // [[10], [], []]
        assert!(sub_fixed_size_list_gather(&arr, &indices, false).is_err());

        // oob null_on_oob=true
        let result = sub_fixed_size_list_gather(&arr, &indices, true).unwrap();
        let out = result
            .values()
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();
        assert!(!out.is_valid(0));
    }
}
