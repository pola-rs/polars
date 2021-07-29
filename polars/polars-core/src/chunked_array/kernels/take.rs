use crate::prelude::*;
use crate::utils::CustomIterTools;
use arrow::array::*;
use arrow::bitmap::MutableBitmap;
use arrow::buffer::{Buffer, MutableBuffer};
use std::convert::TryFrom;
use std::sync::Arc;

/// Take kernel for single chunk with nulls and arrow array as index that may have nulls.
pub(crate) unsafe fn take_primitive_unchecked<T: PolarsNumericType>(
    arr: &PrimitiveArray<T::Native>,
    indices: &UInt32Array,
) -> Arc<PrimitiveArray<T::Native>> {
    let array_values = arr.values();
    let index_values = indices.values();
    let validity_values = arr.validity().as_ref().expect("should have nulls");

    // first take the values, these are always needed
    let values: AlignedVec<T::Native> = index_values
        .iter()
        .map(|idx| *array_values.get_unchecked(*idx as usize))
        .collect_trusted();

    // the validity buffer we will fill with all valid. And we unset the ones that are null
    // in later checks
    // this is in the assumption that most values will be valid.
    // Maybe we could add another branch based on the null count
    let mut validity = MutableBitmap::with_capacity(indices.len());
    validity.extend_constant(indices.len(), true);

    let arr = if let Some(validity_indices) = indices.validity().as_ref() {
        index_values.iter().enumerate().for_each(|(i, idx)| {
            // i is iteration count
            // idx is the index that we take from the values array.
            let idx = *idx as usize;
            if !validity_indices.get_bit_unchecked(i) || !validity_values.get_bit_unchecked(idx) {
                validity.set(i, false);
            }
        });
        PrimitiveArray::from_data(
            T::get_dtype().to_arrow(),
            values.into(),
            Some(validity.into()),
        )
    } else {
        index_values.iter().enumerate().for_each(|(i, idx)| {
            let idx = *idx as usize;
            if !validity_values.get_bit_unchecked(idx) {
                validity.set(i, false);
            }
        });
        PrimitiveArray::from_data(
            T::get_dtype().to_arrow(),
            values.into(),
            Some(validity.into()),
        )
    };

    Arc::new(arr)
}

/// Take kernel for single chunk without nulls and arrow array as index.
pub(crate) unsafe fn take_no_null_primitive<T: PolarsPrimitiveType>(
    arr: &PrimitiveArray<T::Native>,
    indices: &UInt32Array,
) -> Arc<PrimitiveArray<T::Native>> {
    assert_eq!(arr.null_count(), 0);

    let array_values = arr.values().as_slice();
    let index_values = indices.values().as_slice();

    let iter = index_values
        .iter()
        .map(|idx| *array_values.get_unchecked(*idx as usize));

    let values = Buffer::from_trusted_len_iter(iter);
    let validity = indices.validity().clone();
    Arc::new(PrimitiveArray::from_data(
        T::get_dtype().to_arrow(),
        values,
        validity,
    ))
}

/// Take kernel for single chunk without nulls and an iterator as index.
///
/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
pub(crate) unsafe fn take_no_null_primitive_iter_unchecked<
    T: PolarsPrimitiveType,
    I: IntoIterator<Item = usize>,
>(
    arr: &PrimitiveArray<T::Native>,
    indices: I,
) -> Arc<PrimitiveArray<T::Native>> {
    assert_eq!(arr.null_count(), 0);

    let array_values = arr.values().as_slice();

    let iter = indices
        .into_iter()
        .map(|idx| *array_values.get_unchecked(idx));

    let values = Buffer::from_trusted_len_iter_unchecked(iter);
    Arc::new(PrimitiveArray::from_data(
        T::get_dtype().to_arrow(),
        values,
        None,
    ))
}

/// Take kernel for a single chunk with null values and an iterator as index.
///
/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
pub(crate) unsafe fn take_primitive_iter_unchecked<
    T: PolarsPrimitiveType,
    I: IntoIterator<Item = usize>,
>(
    arr: &PrimitiveArray<T::Native>,
    indices: I,
) -> Arc<PrimitiveArray<T::Native>> {
    let array_values = arr.values().as_slice();
    let validity = arr.validity().as_ref().expect("should have nulls");

    let iter = indices.into_iter().map(|idx| {
        if validity.get_bit_unchecked(idx) {
            Some(*array_values.get_unchecked(idx))
        } else {
            None
        }
    });

    let arr = PrimitiveArray::from_trusted_len_iter_unchecked(iter);
    Arc::new(arr)
}

/// Take kernel for a single chunk without nulls and an iterator that can produce None values.
/// This is used in join operations.
///
/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
pub(crate) unsafe fn take_no_null_primitive_opt_iter_unchecked<
    T: PolarsPrimitiveType,
    I: IntoIterator<Item = Option<usize>>,
>(
    arr: &PrimitiveArray<T::Native>,
    indices: I,
) -> Arc<PrimitiveArray<T::Native>> {
    let array_values = arr.values().as_slice();

    let iter = indices
        .into_iter()
        .map(|opt_idx| opt_idx.map(|idx| *array_values.get_unchecked(idx)));
    let arr = PrimitiveArray::from_trusted_len_iter_unchecked(iter).to(T::get_dtype().to_arrow());

    Arc::new(arr)
}

/// Take kernel for a single chunk and an iterator that can produce None values.
/// This is used in join operations.
///
/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
pub(crate) unsafe fn take_primitive_opt_iter_unchecked<
    T: PolarsPrimitiveType,
    I: IntoIterator<Item = Option<usize>>,
>(
    arr: &PrimitiveArray<T::Native>,
    indices: I,
) -> Arc<PrimitiveArray<T::Native>> {
    let array_values = arr.values().as_slice();
    let validity = arr.validity().as_ref().expect("should have nulls");

    let iter = indices.into_iter().map(|opt_idx| {
        opt_idx.and_then(|idx| {
            if validity.get_bit_unchecked(idx) {
                Some(*array_values.get_unchecked(idx))
            } else {
                None
            }
        })
    });
    let arr = PrimitiveArray::from_trusted_len_iter_unchecked(iter).to(T::get_dtype().to_arrow());

    Arc::new(arr)
}

/// Take kernel for multiple chunks. We directly return a ChunkedArray because that path chooses the fastest collection path.
pub(crate) fn take_primitive_iter_n_chunks<T: PolarsNumericType, I: IntoIterator<Item = usize>>(
    ca: &ChunkedArray<T>,
    indices: I,
) -> ChunkedArray<T> {
    let taker = ca.take_rand();
    indices.into_iter().map(|idx| taker.get(idx)).collect()
}

/// Take kernel for multiple chunks where an iterator can produce None values.
/// Used in join operations. We directly return a ChunkedArray because that path chooses the fastest collection path.
pub(crate) fn take_primitive_opt_iter_n_chunks<
    T: PolarsNumericType,
    I: IntoIterator<Item = Option<usize>>,
>(
    ca: &ChunkedArray<T>,
    indices: I,
) -> ChunkedArray<T> {
    let taker = ca.take_rand();
    indices
        .into_iter()
        .map(|opt_idx| opt_idx.and_then(|idx| taker.get(idx)))
        .collect()
}

/// Take kernel for single chunk without nulls and an iterator as index.
///
/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
pub(crate) unsafe fn take_no_null_bool_iter_unchecked<I: IntoIterator<Item = usize>>(
    arr: &BooleanArray,
    indices: I,
) -> Arc<BooleanArray> {
    debug_assert_eq!(arr.null_count(), 0);
    let iter = indices
        .into_iter()
        .map(|idx| Some(arr.values().get_bit_unchecked(idx)));

    Arc::new(BooleanArray::from_trusted_len_iter_unchecked(iter))
}

/// Take kernel for single chunk and an iterator as index.
/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
pub(crate) unsafe fn take_bool_iter_unchecked<I: IntoIterator<Item = usize>>(
    arr: &BooleanArray,
    indices: I,
) -> Arc<BooleanArray> {
    let validity = arr.validity().as_ref().expect("should have nulls");

    let iter = indices.into_iter().map(|idx| {
        if validity.get_bit_unchecked(idx) {
            Some(arr.value_unchecked(idx))
        } else {
            None
        }
    });

    Arc::new(BooleanArray::from_trusted_len_iter_unchecked(iter))
}

/// Take kernel for single chunk and an iterator as index.
/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
pub(crate) unsafe fn take_bool_opt_iter_unchecked<I: IntoIterator<Item = Option<usize>>>(
    arr: &BooleanArray,
    indices: I,
) -> Arc<BooleanArray> {
    let validity = arr.validity().as_ref().expect("should have nulls");
    let iter = indices.into_iter().map(|opt_idx| {
        opt_idx.and_then(|idx| {
            if validity.get_bit_unchecked(idx) {
                Some(arr.value_unchecked(idx))
            } else {
                None
            }
        })
    });

    Arc::new(BooleanArray::from_trusted_len_iter_unchecked(iter))
}

/// Take kernel for single chunk without null values and an iterator as index that may produce None values.
/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
pub(crate) unsafe fn take_no_null_bool_opt_iter_unchecked<I: IntoIterator<Item = Option<usize>>>(
    arr: &BooleanArray,
    indices: I,
) -> Arc<BooleanArray> {
    let iter = indices
        .into_iter()
        .map(|opt_idx| opt_idx.map(|idx| arr.value_unchecked(idx)));

    Arc::new(BooleanArray::from_trusted_len_iter_unchecked(iter))
}

/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
pub(crate) unsafe fn take_no_null_utf8_iter_unchecked<I: IntoIterator<Item = usize>>(
    arr: &LargeStringArray,
    indices: I,
) -> Arc<LargeStringArray> {
    let iter = indices
        .into_iter()
        .map(|idx| Some(arr.value_unchecked(idx)));

    Arc::new(LargeStringArray::from_trusted_len_iter_unchecked(iter))
}

/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
pub(crate) unsafe fn take_utf8_iter_unchecked<I: IntoIterator<Item = usize>>(
    arr: &LargeStringArray,
    indices: I,
) -> Arc<LargeStringArray> {
    let validity = arr.validity().as_ref().expect("should have nulls");
    let iter = indices.into_iter().map(|idx| {
        if validity.get_bit_unchecked(idx) {
            Some(arr.value_unchecked(idx))
        } else {
            None
        }
    });

    Arc::new(LargeStringArray::from_trusted_len_iter_unchecked(iter))
}

/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
pub(crate) unsafe fn take_no_null_utf8_opt_iter_unchecked<I: IntoIterator<Item = Option<usize>>>(
    arr: &LargeStringArray,
    indices: I,
) -> Arc<LargeStringArray> {
    let iter = indices
        .into_iter()
        .map(|opt_idx| opt_idx.map(|idx| arr.value_unchecked(idx)));

    Arc::new(LargeStringArray::from_trusted_len_iter_unchecked(iter))
}

/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
pub(crate) unsafe fn take_utf8_opt_iter_unchecked<I: IntoIterator<Item = Option<usize>>>(
    arr: &LargeStringArray,
    indices: I,
) -> Arc<LargeStringArray> {
    let validity = arr.validity().as_ref().expect("should have nulls");
    let iter = indices.into_iter().map(|opt_idx| {
        opt_idx.and_then(|idx| {
            if validity.get_bit_unchecked(idx) {
                Some(arr.value_unchecked(idx))
            } else {
                None
            }
        })
    });
    Arc::new(LargeStringArray::from_trusted_len_iter_unchecked(iter))
}

pub(crate) unsafe fn take_utf8_unchecked(
    arr: &LargeStringArray,
    indices: &UInt32Array,
) -> Arc<LargeStringArray> {
    let data_len = indices.len();

    let mut offset_buf = MutableBuffer::<i64>::from_len_zeroed(data_len + 1);
    let offset_typed = offset_buf.as_mut_slice();

    let mut length_so_far = 0;
    offset_typed[0] = length_so_far;

    let validity;

    // The required size is yet unknown
    // Allocate 2.0 times the expected size.
    // where expected size is the length of bytes multiplied by the factor (take_len / current_len)
    let mut values_capacity = if arr.len() > 0 {
        ((arr.len() as f32 * 2.0) as usize) / arr.len() * indices.len() as usize
    } else {
        0
    };

    // 16 bytes per string as default alloc
    let mut values_buf = AlignedVec::<u8>::with_capacity(values_capacity);

    // both 0 nulls
    if arr.null_count() == 0 && indices.null_count() == 0 {
        offset_typed
            .iter_mut()
            .skip(1)
            .enumerate()
            .for_each(|(idx, offset)| {
                let index = indices.value_unchecked(idx) as usize;
                let s = arr.value_unchecked(index);
                length_so_far += s.len() as i64;
                *offset = length_so_far;

                if length_so_far as usize >= values_capacity {
                    values_buf.reserve(values_capacity);
                    values_capacity *= 2;
                }

                values_buf.extend_from_slice(s.as_bytes())
            });
        validity = None;
    } else if arr.null_count() == 0 {
        offset_typed
            .iter_mut()
            .skip(1)
            .enumerate()
            .for_each(|(idx, offset)| {
                if indices.is_valid(idx) {
                    let index = indices.value_unchecked(idx) as usize;
                    let s = arr.value_unchecked(index);
                    length_so_far += s.len() as i64;

                    if length_so_far as usize >= values_capacity {
                        values_buf.reserve(values_capacity);
                        values_capacity *= 2;
                    }

                    values_buf.extend_from_slice(s.as_bytes())
                }
                *offset = length_so_far;
            });
        validity = indices.validity().clone();
    } else {
        let mut builder = MutableUtf8Array::with_capacities(data_len, length_so_far as usize);
        let validity_arr = arr.validity().as_ref().expect("should have nulls");

        if indices.null_count() == 0 {
            (0..data_len).for_each(|idx| {
                let index = indices.value_unchecked(idx) as usize;
                builder.push(if validity_arr.get_bit_unchecked(index) {
                    let s = arr.value_unchecked(index);
                    Some(s)
                } else {
                    None
                });
            });
        } else {
            let validity_indices = indices.validity().as_ref().expect("should have nulls");
            (0..data_len).for_each(|idx| {
                if validity_indices.get_bit_unchecked(idx) {
                    let index = indices.value_unchecked(idx) as usize;

                    if validity_arr.get_bit_unchecked(index) {
                        let s = arr.value_unchecked(index);
                        builder.push(Some(s));
                    } else {
                        builder.push_null();
                    }
                } else {
                    builder.push_null();
                }
            });
        }

        let array: Utf8Array<i64> = builder.into();
        return Arc::new(array);
    }

    // Safety: all "values" are &str, and thus valid utf8
    Arc::new(Utf8Array::<i64>::from_data_unchecked(
        offset_buf.into(),
        values_buf.into(),
        validity,
    ))
}

/// Forked and adapted from arrow-rs
/// This is faster because it does no bounds checks and allocates directly into aligned memory
///
/// Takes/filters a list array's inner data using the offsets of the list array.
///
/// Where a list array has indices `[0,2,5,10]`, taking indices of `[2,0]` returns
/// an array of the indices `[5..10, 0..2]` and offsets `[0,5,7]` (5 elements and 2
/// elements)
///
/// # Safety
/// No bounds checks
unsafe fn take_value_indices_from_list(
    list: &ListArray<i64>,
    indices: &UInt32Array,
) -> (UInt32Array, AlignedVec<i64>) {
    let offsets = list.offsets().as_slice();

    let mut new_offsets = AlignedVec::with_capacity(indices.len());
    // will likely have at least indices.len values
    let mut values = AlignedVec::with_capacity(indices.len());
    let mut current_offset = 0;
    // add first offset
    new_offsets.push(0);
    // compute the value indices, and set offsets accordingly

    let indices_values = indices.values();

    if indices.null_count() == 0 {
        for i in 0..indices.len() {
            let idx = *indices_values.get_unchecked(i) as usize;
            let start = *offsets.get_unchecked(idx);
            let end = *offsets.get_unchecked(idx + 1);
            current_offset += end - start;
            new_offsets.push(current_offset);

            let mut curr = start;

            // if start == end, this slot is empty
            while curr < end {
                values.push(curr as u32);
                curr += 1;
            }
        }
    } else {
        let validity = indices.validity().as_ref().expect("should have nulls");

        for i in 0..indices.len() {
            if validity.get_bit_unchecked(i) {
                let idx = *indices_values.get_unchecked(i) as usize;
                let start = *offsets.get_unchecked(idx);
                let end = *offsets.get_unchecked(idx + 1);
                current_offset += end - start;
                new_offsets.push(current_offset);

                let mut curr = start;

                // if start == end, this slot is empty
                while curr < end {
                    values.push(curr as u32);
                    curr += 1;
                }
            } else {
                new_offsets.push(current_offset);
            }
        }
    }

    (
        PrimitiveArray::from_data(ArrowDataType::UInt32, values.into(), None),
        new_offsets,
    )
}

/// Forked and adapted from arrow-rs
/// This is faster because it does no bounds checks and allocates directly into aligned memory
///
/// # Safety
/// No bounds checks
pub(crate) unsafe fn take_list_unchecked(
    values: &ListArray<i64>,
    indices: &UInt32Array,
) -> ListArray<i64> {
    // taking the whole list or a contiguous sublist
    let (list_indices, offsets) = take_value_indices_from_list(values, indices);

    // tmp series so that we can take primitives from it
    let s = Series::try_from(("", values.values().clone() as ArrayRef)).unwrap();
    let taken = s
        .take_unchecked(&UInt32Chunked::new_from_chunks(
            "",
            vec![Arc::new(list_indices) as ArrayRef],
        ))
        .unwrap();
    let taken = taken.chunks()[0].clone();

    let validity =
        if values.null_count() > 0 || indices.null_count() > 0 {
            // determine null buffer, which are a function of `values` and `indices`
            let mut validity = MutableBitmap::with_capacity(indices.len());
            validity.extend_constant(indices.len(), true);

            {
                offsets.as_slice().windows(2).enumerate().for_each(
                    |(i, window): (usize, &[i64])| {
                        if window[0] == window[1] {
                            // offsets are equal, slot is null
                            validity.set(i, false);
                        }
                    },
                );
            }
            Some(validity.into())
        } else {
            None
        };
    ListArray::from_data(values.data_type().clone(), offsets.into(), taken, validity)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_utf8_kernel() {
        let s = LargeStringArray::from(vec![Some("foo"), None, Some("bar")]);
        unsafe {
            let out = take_utf8_unchecked(&s, &UInt32Array::from_slice(&[1, 2]));
            assert!(out.is_null(0));
            assert!(out.is_valid(1));
            let out = take_utf8_unchecked(&s, &UInt32Array::from(vec![None, Some(2)]));
            assert!(out.is_null(0));
            assert!(out.is_valid(1));
            let out = take_utf8_unchecked(&s, &UInt32Array::from(vec![None, None]));
            assert!(out.is_null(0));
            assert!(out.is_null(1));
        }
    }
}
