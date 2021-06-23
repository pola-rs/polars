use crate::prelude::*;
use arrow::array::*;
use arrow::buffer::MutableBuffer;
use std::sync::Arc;

/// Take kernel for single chunk without nulls and arrow array as index.
pub(crate) unsafe fn take_no_null_primitive<T: PolarsPrimitiveType>(
    arr: &PrimitiveArray<T::Native>,
    indices: &UInt32Array,
) -> Arc<PrimitiveArray<T::Native>> {
    assert_eq!(arr.null_count(), 0);

    let data_len = indices.len();
    let array_values = arr.values();
    let index_values = indices.values();

    let mut values = MutableBuffer::<T::Native>::with_capacity(data_len);
    let iter = index_values
        .iter()
        .map(|idx| *array_values.get_unchecked(*idx as usize));
    values.extend(iter);

    let nulls = indices.validity().clone();

    Arc::new(to_primitive::<T>(values, nulls))
}

/// Take kernel for single chunk without nulls and an iterator as index.
pub(crate) unsafe fn take_no_null_primitive_iter_unchecked<
    T: PolarsPrimitiveType,
    I: IntoIterator<Item = usize>,
>(
    arr: &PrimitiveArray<T::Native>,
    indices: I,
) -> Arc<PrimitiveArray<T::Native>> {
    assert_eq!(arr.null_count(), 0);

    let array_values = arr.values();

    let av = indices
        .into_iter()
        .map(|idx| *array_values.get_unchecked(idx))
        .collect::<AlignedVec<_>>();

    let arr = to_primitive::<T>(av, None);
    Arc::new(arr)
}

/// Take kernel for single chunk without nulls and an iterator as index that does bound checks.
pub(crate) fn take_no_null_primitive_iter<T: PolarsPrimitiveType, I: IntoIterator<Item = usize>>(
    arr: &PrimitiveArray<T::Native>,
    indices: I,
) -> Arc<PrimitiveArray<T::Native>> {
    assert_eq!(arr.null_count(), 0);

    let array_values = arr.values();

    let av = indices
        .into_iter()
        .map(|idx| array_values[idx])
        .collect::<AlignedVec<_>>();
    let arr = to_primitive::<T>(av, None);

    Arc::new(arr)
}

/// Take kernel for a single chunk with null values and an iterator as index.
pub(crate) unsafe fn take_primitive_iter_unchecked<
    T: PolarsPrimitiveType,
    I: IntoIterator<Item = usize>,
>(
    arr: &PrimitiveArray<T::Native>,
    indices: I,
) -> Arc<PrimitiveArray<T::Native>> {
    let array_values = arr.values();

    let arr = indices
        .into_iter()
        .map(|idx| {
            if arr.is_valid(idx) {
                Some(*array_values.get_unchecked(idx))
            } else {
                None
            }
        })
        .collect::<PrimitiveArray<T::Native>>()
        .to(T::get_dtype().to_arrow());

    Arc::new(arr)
}

/// Take kernel for a single chunk with null values and an iterator as index that does bound checks.
pub(crate) fn take_primitive_iter<T: PolarsPrimitiveType, I: IntoIterator<Item = usize>>(
    arr: &PrimitiveArray<T::Native>,
    indices: I,
) -> Arc<PrimitiveArray<T::Native>> {
    let array_values = arr.values();

    let arr = indices
        .into_iter()
        .map(|idx| {
            if arr.is_valid(idx) {
                Some(array_values[idx])
            } else {
                None
            }
        })
        .collect::<PrimitiveArray<T::Native>>()
        .to(T::get_dtype().to_arrow());

    Arc::new(arr)
}

/// Take kernel for a single chunk without nulls and an iterator that can produce None values.
/// This is used in join operations.
pub(crate) unsafe fn take_no_null_primitive_opt_iter_unchecked<
    T: PolarsPrimitiveType,
    I: IntoIterator<Item = Option<usize>>,
>(
    arr: &PrimitiveArray<T::Native>,
    indices: I,
) -> Arc<PrimitiveArray<T::Native>> {
    let array_values = arr.values();

    let iter = indices
        .into_iter()
        .map(|opt_idx| opt_idx.map(|idx| *array_values.get_unchecked(idx)));
    let arr = PrimitiveArray::from_trusted_len_iter_unchecked(iter).to(T::get_dtype().to_arrow());

    Arc::new(arr)
}

/// Take kernel for a single chunk and an iterator that can produce None values.
/// This is used in join operations.
pub(crate) unsafe fn take_primitive_opt_iter_unchecked<
    T: PolarsPrimitiveType,
    I: IntoIterator<Item = Option<usize>>,
>(
    arr: &PrimitiveArray<T::Native>,
    indices: I,
) -> Arc<PrimitiveArray<T::Native>> {
    let array_values = arr.values();

    let iter = indices.into_iter().map(|opt_idx| {
        opt_idx.and_then(|idx| {
            if arr.is_valid(idx) {
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

/// Take kernel for single chunk without nulls and an iterator as index that does bound checks.
pub(crate) fn take_no_null_bool_iter<I: IntoIterator<Item = usize>>(
    arr: &BooleanArray,
    indices: I,
) -> Arc<BooleanArray> {
    debug_assert_eq!(arr.null_count(), 0);

    let iter = indices.into_iter().map(|idx| Some(arr.value(idx)));

    Arc::new(iter.collect())
}

/// Take kernel for single chunk without nulls and an iterator as index.
pub(crate) unsafe fn take_no_null_bool_iter_unchecked<I: IntoIterator<Item = usize>>(
    arr: &BooleanArray,
    indices: I,
) -> Arc<BooleanArray> {
    debug_assert_eq!(arr.null_count(), 0);
    let iter = indices
        .into_iter()
        .map(|idx| Some(arr.values().get_bit_unchecked(idx)));

    Arc::new(iter.collect())
}

/// Take kernel for single chunk and an iterator as index that does bound checks.
pub(crate) fn take_bool_iter<I: IntoIterator<Item = usize>>(
    arr: &BooleanArray,
    indices: I,
) -> Arc<BooleanArray> {
    let iter = indices.into_iter().map(|idx| {
        if arr.is_null(idx) {
            None
        } else {
            Some(arr.value(idx))
        }
    });

    Arc::new(iter.collect())
}

/// Take kernel for single chunk and an iterator as index.
pub(crate) unsafe fn take_bool_iter_unchecked<I: IntoIterator<Item = usize>>(
    arr: &BooleanArray,
    indices: I,
) -> Arc<BooleanArray> {
    let iter = indices.into_iter().map(|idx| {
        if arr.is_null(idx) {
            None
        } else {
            Some(arr.value_unchecked(idx))
        }
    });

    Arc::new(iter.collect())
}

/// Take kernel for single chunk and an iterator as index.
pub(crate) unsafe fn take_bool_opt_iter_unchecked<I: IntoIterator<Item = Option<usize>>>(
    arr: &BooleanArray,
    indices: I,
) -> Arc<BooleanArray> {
    let iter = indices.into_iter().map(|opt_idx| {
        opt_idx.and_then(|idx| {
            if arr.is_null(idx) {
                None
            } else {
                Some(arr.value_unchecked(idx))
            }
        })
    });

    Arc::new(iter.collect())
}

/// Take kernel for single chunk without null values and an iterator as index that may produce None values.
pub(crate) unsafe fn take_no_null_bool_opt_iter_unchecked<I: IntoIterator<Item = Option<usize>>>(
    arr: &BooleanArray,
    indices: I,
) -> Arc<BooleanArray> {
    let iter = indices
        .into_iter()
        .map(|opt_idx| opt_idx.map(|idx| arr.value_unchecked(idx)));

    Arc::new(iter.collect())
}

pub(crate) unsafe fn take_no_null_utf8_iter_unchecked<I: IntoIterator<Item = usize>>(
    arr: &LargeStringArray,
    indices: I,
) -> Arc<LargeStringArray> {
    let iter = indices
        .into_iter()
        .map(|idx| Some(arr.value_unchecked(idx)));

    Arc::new(iter.collect())
}

pub(crate) unsafe fn take_utf8_iter_unchecked<I: IntoIterator<Item = usize>>(
    arr: &LargeStringArray,
    indices: I,
) -> Arc<LargeStringArray> {
    let iter = indices.into_iter().map(|idx| {
        if arr.is_null(idx) {
            None
        } else {
            Some(arr.value_unchecked(idx))
        }
    });

    Arc::new(iter.collect())
}

pub(crate) unsafe fn take_no_null_utf8_opt_iter_unchecked<I: IntoIterator<Item = Option<usize>>>(
    arr: &LargeStringArray,
    indices: I,
) -> Arc<LargeStringArray> {
    let iter = indices
        .into_iter()
        .map(|opt_idx| opt_idx.map(|idx| arr.value_unchecked(idx)));

    Arc::new(iter.collect())
}

pub(crate) unsafe fn take_utf8_opt_iter_unchecked<I: IntoIterator<Item = Option<usize>>>(
    arr: &LargeStringArray,
    indices: I,
) -> Arc<LargeStringArray> {
    let iter = indices.into_iter().map(|opt_idx| {
        opt_idx.and_then(|idx| {
            if arr.is_null(idx) {
                None
            } else {
                Some(arr.value_unchecked(idx))
            }
        })
    });

    Arc::new(iter.collect())
}

pub(crate) fn take_no_null_utf8_iter<I: IntoIterator<Item = usize>>(
    arr: &LargeStringArray,
    indices: I,
) -> Arc<LargeStringArray> {
    let iter = indices.into_iter().map(|idx| Some(arr.value(idx)));

    Arc::new(iter.collect())
}

pub(crate) fn take_utf8_iter<I: IntoIterator<Item = usize>>(
    arr: &LargeStringArray,
    indices: I,
) -> Arc<LargeStringArray> {
    let iter = indices.into_iter().map(|idx| {
        if arr.is_null(idx) {
            None
        } else {
            Some(arr.value(idx))
        }
    });

    Arc::new(iter.collect())
}

pub(crate) unsafe fn take_utf8(
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

        if indices.null_count() == 0 {
            (0..data_len).for_each(|idx| {
                let index = indices.value_unchecked(idx) as usize;
                builder.push(if arr.is_valid(index) {
                    let s = arr.value_unchecked(index);
                    Some(s)
                } else {
                    None
                });
            });
        } else {
            (0..data_len).for_each(|idx| {
                if indices.is_valid(idx) {
                    let index = indices.value_unchecked(idx) as usize;

                    if arr.is_valid(index) {
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_utf8_kernel() {
        let s = LargeStringArray::from(vec![Some("foo"), None, Some("bar")]);
        unsafe {
            let out = take_utf8(&s, &UInt32Array::from_slice(&[1, 2]));
            assert!(out.is_null(0));
            assert!(out.is_valid(1));
            let out = take_utf8(&s, &UInt32Array::from(vec![None, Some(2)]));
            assert!(out.is_null(0));
            assert!(out.is_valid(1));
            let out = take_utf8(&s, &UInt32Array::from(vec![None, None]));
            assert!(out.is_null(0));
            assert!(out.is_null(1));
        }
    }
}
