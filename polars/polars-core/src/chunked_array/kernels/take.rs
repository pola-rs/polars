use crate::prelude::*;
use arrow::array::{
    Array, ArrayData, BooleanArray, LargeStringArray, LargeStringBuilder, PrimitiveArray,
    UInt32Array,
};
use arrow::buffer::MutableBuffer;
use std::mem;
use std::sync::Arc;

/// Take kernel for single chunk without nulls and arrow array as index.
pub(crate) unsafe fn take_no_null_primitive<T: PolarsNumericType>(
    arr: &PrimitiveArray<T>,
    indices: &UInt32Array,
) -> Arc<PrimitiveArray<T>> {
    assert_eq!(arr.null_count(), 0);

    let data_len = indices.len();
    let array_values = arr.values();
    let index_values = indices.values();

    let mut values = AlignedVec::<T::Native>::with_capacity_aligned(data_len);
    let iter = index_values
        .iter()
        .map(|idx| *array_values.get_unchecked(*idx as usize));
    values.extend(iter);

    let nulls = indices.data_ref().null_buffer().cloned();

    let arr = values.into_primitive_array::<T>(nulls);
    Arc::new(arr)
}

/// Take kernel for single chunk without nulls and an iterator as index.
pub(crate) unsafe fn take_no_null_primitive_iter_unchecked<
    T: PolarsNumericType,
    I: IntoIterator<Item = usize>,
>(
    arr: &PrimitiveArray<T>,
    indices: I,
) -> Arc<PrimitiveArray<T>> {
    assert_eq!(arr.null_count(), 0);

    let array_values = arr.values();

    let av = indices
        .into_iter()
        .map(|idx| *array_values.get_unchecked(idx))
        .collect::<AlignedVec<_>>();

    let arr = av.into_primitive_array::<T>(None);
    Arc::new(arr)
}

/// Take kernel for single chunk without nulls and an iterator as index that does bound checks.
pub(crate) fn take_no_null_primitive_iter<T: PolarsNumericType, I: IntoIterator<Item = usize>>(
    arr: &PrimitiveArray<T>,
    indices: I,
) -> Arc<PrimitiveArray<T>> {
    assert_eq!(arr.null_count(), 0);

    let array_values = arr.values();

    let av = indices
        .into_iter()
        .map(|idx| array_values[idx])
        .collect::<AlignedVec<_>>();
    let arr = av.into_primitive_array(None);

    Arc::new(arr)
}

/// Take kernel for a single chunk with null values and an iterator as index.
pub(crate) unsafe fn take_primitive_iter_unchecked<
    T: PolarsNumericType,
    I: IntoIterator<Item = usize>,
>(
    arr: &PrimitiveArray<T>,
    indices: I,
) -> Arc<PrimitiveArray<T>> {
    let array_values = arr.values();

    let iter = indices.into_iter().map(|idx| {
        if arr.is_valid(idx) {
            Some(*array_values.get_unchecked(idx))
        } else {
            None
        }
    });
    let arr = PrimitiveArray::from_trusted_len_iter(iter);

    Arc::new(arr)
}

/// Take kernel for a single chunk with null values and an iterator as index that does bound checks.
pub(crate) fn take_primitive_iter<T: PolarsNumericType, I: IntoIterator<Item = usize>>(
    arr: &PrimitiveArray<T>,
    indices: I,
) -> Arc<PrimitiveArray<T>> {
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
        .collect();

    Arc::new(arr)
}

/// Take kernel for a single chunk without nulls and an iterator that can produce None values.
/// This is used in join operations.
pub(crate) unsafe fn take_no_null_primitive_opt_iter_unchecked<
    T: PolarsNumericType,
    I: IntoIterator<Item = Option<usize>>,
>(
    arr: &PrimitiveArray<T>,
    indices: I,
) -> Arc<PrimitiveArray<T>> {
    let array_values = arr.values();

    let iter = indices
        .into_iter()
        .map(|opt_idx| opt_idx.map(|idx| *array_values.get_unchecked(idx)));
    let arr = PrimitiveArray::from_trusted_len_iter(iter);

    Arc::new(arr)
}

/// Take kernel for a single chunk and an iterator that can produce None values.
/// This is used in join operations.
pub(crate) unsafe fn take_primitive_opt_iter_unchecked<
    T: PolarsNumericType,
    I: IntoIterator<Item = Option<usize>>,
>(
    arr: &PrimitiveArray<T>,
    indices: I,
) -> Arc<PrimitiveArray<T>> {
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
    let arr = PrimitiveArray::from_trusted_len_iter(iter);

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
        .map(|idx| Some(arr.value_unchecked(idx)));

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

    let offset_len_in_bytes = (data_len + 1) * mem::size_of::<i64>();
    let mut offset_buf = MutableBuffer::new(offset_len_in_bytes);
    offset_buf.resize(offset_len_in_bytes, 0);
    let offset_typed = offset_buf.typed_data_mut();

    let mut length_so_far = 0;
    offset_typed[0] = length_so_far;

    let nulls;

    // The required size is yet unknown
    // Allocate 2.0 times the expected size.
    // where expected size is the length of bytes multiplied by the factor (take_len / current_len)
    let mut values_capacity = if arr.len() > 0 {
        ((arr.value_data().len() as f32 * 2.0) as usize) / arr.len() * indices.len() as usize
    } else {
        0
    };

    // 16 bytes per string as default alloc
    let mut values_buf = AlignedVec::<u8>::with_capacity_aligned(values_capacity);

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
        nulls = None;
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
        nulls = indices.data_ref().null_buffer().cloned();
    } else {
        let mut builder = LargeStringBuilder::with_capacity(data_len, length_so_far as usize);

        if indices.null_count() == 0 {
            (0..data_len).for_each(|idx| {
                let index = indices.value_unchecked(idx) as usize;
                if arr.is_valid(index) {
                    let s = arr.value_unchecked(index);
                    builder.append_value(s).unwrap();
                } else {
                    builder.append_null().unwrap();
                }
            });
        } else {
            (0..data_len).for_each(|idx| {
                if indices.is_valid(idx) {
                    let index = indices.value_unchecked(idx) as usize;

                    if arr.is_valid(index) {
                        let s = arr.value_unchecked(index);
                        builder.append_value(s).unwrap();
                    } else {
                        builder.append_null().unwrap();
                    }
                } else {
                    builder.append_null().unwrap();
                }
            });
        }

        return Arc::new(builder.finish());
    }
    values_buf.shrink_to_fit();

    let mut data = ArrayData::builder(ArrowDataType::LargeUtf8)
        .len(data_len)
        .add_buffer(offset_buf.into())
        .add_buffer(values_buf.into_arrow_buffer());
    if let Some(null_buffer) = nulls {
        data = data.null_bit_buffer(null_buffer);
    }
    Arc::new(LargeStringArray::from(data.build()))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_utf8_kernel() {
        let s = LargeStringArray::from(vec![Some("foo"), None, Some("bar")]);
        unsafe {
            let out = take_utf8(&s, &UInt32Array::from(vec![1, 2]));
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
