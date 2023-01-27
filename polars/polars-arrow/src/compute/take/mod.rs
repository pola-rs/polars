mod boolean;

use arrow::array::*;
use arrow::bitmap::MutableBitmap;
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, PhysicalType};
use arrow::offset::Offsets;
use arrow::types::NativeType;

use crate::bit_util::unset_bit_raw;
use crate::prelude::*;
use crate::trusted_len::{PushUnchecked, TrustedLen};
use crate::utils::{with_match_primitive_type, CustomIterTools};

/// # Safety
/// Does not do bounds checks
pub unsafe fn take_unchecked(arr: &dyn Array, idx: &IdxArr) -> ArrayRef {
    use PhysicalType::*;
    match arr.data_type().to_physical_type() {
        Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {
            let arr: &PrimitiveArray<$T> = arr.as_any().downcast_ref().unwrap();
            if arr.null_count() > 0 {
                take_primitive_unchecked::<$T>(arr, idx)
            } else {
                take_no_null_primitive_unchecked::<$T>(arr, idx)
            }
        }),
        LargeUtf8 => {
            let arr = arr.as_any().downcast_ref().unwrap();
            take_utf8_unchecked(arr, idx)
        }
        Boolean => {
            let arr = arr.as_any().downcast_ref().unwrap();
            Box::new(boolean::take_unchecked(arr, idx))
        }
        // TODO! implement proper unchecked version
        #[cfg(feature = "compute")]
        _ => {
            use arrow::compute::take::take;
            take(arr, idx).unwrap()
        }
        #[cfg(not(feature = "compute"))]
        _ => {
            panic!("activate compute feature")
        }
    }
}

/// Take kernel for single chunk with nulls and arrow array as index that may have nulls.
/// # Safety
/// caller must ensure indices are in bounds
pub unsafe fn take_primitive_unchecked<T: NativeType>(
    arr: &PrimitiveArray<T>,
    indices: &IdxArr,
) -> Box<PrimitiveArray<T>> {
    let array_values = arr.values().as_slice();
    let index_values = indices.values().as_slice();
    let validity_values = arr.validity().expect("should have nulls");

    // first take the values, these are always needed
    let values: Vec<T> = index_values
        .iter()
        .map(|idx| {
            debug_assert!((*idx as usize) < array_values.len());
            *array_values.get_unchecked(*idx as usize)
        })
        .collect_trusted();

    // the validity buffer we will fill with all valid. And we unset the ones that are null
    // in later checks
    // this is in the assumption that most values will be valid.
    // Maybe we could add another branch based on the null count
    let mut validity = MutableBitmap::with_capacity(indices.len());
    validity.extend_constant(indices.len(), true);
    let validity_ptr = validity.as_slice().as_ptr() as *mut u8;

    if let Some(validity_indices) = indices.validity().as_ref() {
        index_values.iter().enumerate().for_each(|(i, idx)| {
            // i is iteration count
            // idx is the index that we take from the values array.
            let idx = *idx as usize;
            if !validity_indices.get_bit_unchecked(i) || !validity_values.get_bit_unchecked(idx) {
                unset_bit_raw(validity_ptr, i);
            }
        });
    } else {
        index_values.iter().enumerate().for_each(|(i, idx)| {
            let idx = *idx as usize;
            if !validity_values.get_bit_unchecked(idx) {
                unset_bit_raw(validity_ptr, i);
            }
        });
    };
    let arr = PrimitiveArray::new(T::PRIMITIVE.into(), values.into(), Some(validity.into()));

    Box::new(arr)
}

/// Take kernel for single chunk without nulls and arrow array as index.
/// # Safety
/// caller must ensure indices are in bounds
pub unsafe fn take_no_null_primitive_unchecked<T: NativeType>(
    arr: &PrimitiveArray<T>,
    indices: &IdxArr,
) -> Box<PrimitiveArray<T>> {
    debug_assert!(arr.null_count() == 0);
    let array_values = arr.values().as_slice();
    let index_values = indices.values().as_slice();

    let iter = index_values.iter().map(|idx| {
        debug_assert!((*idx as usize) < array_values.len());
        *array_values.get_unchecked(*idx as usize)
    });

    let values: Buffer<_> = Vec::from_trusted_len_iter(iter).into();
    let validity = indices.validity().cloned();
    Box::new(PrimitiveArray::new(T::PRIMITIVE.into(), values, validity))
}

/// Take kernel for single chunk without nulls and an iterator as index.
///
/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
#[inline]
pub unsafe fn take_no_null_primitive_iter_unchecked<T: NativeType, I: TrustedLen<Item = usize>>(
    arr: &PrimitiveArray<T>,
    indices: I,
) -> Box<PrimitiveArray<T>> {
    debug_assert!(!arr.has_validity());
    let array_values = arr.values().as_slice();

    let iter = indices.into_iter().map(|idx| {
        debug_assert!((idx) < array_values.len());
        *array_values.get_unchecked(idx)
    });

    let values: Buffer<_> = Vec::from_trusted_len_iter(iter).into();
    Box::new(PrimitiveArray::new(T::PRIMITIVE.into(), values, None))
}

/// Take kernel for a single chunk with null values and an iterator as index.
///
/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
#[inline]
pub unsafe fn take_primitive_iter_unchecked<T: NativeType, I: IntoIterator<Item = usize>>(
    arr: &PrimitiveArray<T>,
    indices: I,
) -> Box<PrimitiveArray<T>> {
    let array_values = arr.values().as_slice();
    let validity = arr.validity().expect("should have nulls");

    let iter = indices.into_iter().map(|idx| {
        if validity.get_bit_unchecked(idx) {
            Some(*array_values.get_unchecked(idx))
        } else {
            None
        }
    });

    let arr = PrimitiveArray::from_trusted_len_iter_unchecked(iter);
    Box::new(arr)
}

/// Take kernel for a single chunk without nulls and an iterator that can produce None values.
/// This is used in join operations.
///
/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
#[inline]
pub unsafe fn take_no_null_primitive_opt_iter_unchecked<
    T: NativeType,
    I: IntoIterator<Item = Option<usize>>,
>(
    arr: &PrimitiveArray<T>,
    indices: I,
) -> Box<PrimitiveArray<T>> {
    let array_values = arr.values().as_slice();

    let iter = indices.into_iter().map(|opt_idx| {
        opt_idx.map(|idx| {
            debug_assert!(idx < array_values.len());
            *array_values.get_unchecked(idx)
        })
    });
    let arr = PrimitiveArray::from_trusted_len_iter_unchecked(iter).to(T::PRIMITIVE.into());

    Box::new(arr)
}

/// Take kernel for a single chunk and an iterator that can produce None values.
/// This is used in join operations.
///
/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
#[inline]
pub unsafe fn take_primitive_opt_iter_unchecked<
    T: NativeType,
    I: IntoIterator<Item = Option<usize>>,
>(
    arr: &PrimitiveArray<T>,
    indices: I,
) -> Box<PrimitiveArray<T>> {
    let array_values = arr.values().as_slice();
    let validity = arr.validity().expect("should have nulls");

    let iter = indices.into_iter().map(|opt_idx| {
        opt_idx.and_then(|idx| {
            if validity.get_bit_unchecked(idx) {
                debug_assert!(idx < array_values.len());
                Some(*array_values.get_unchecked(idx))
            } else {
                None
            }
        })
    });
    let arr = PrimitiveArray::from_trusted_len_iter_unchecked(iter).to(T::PRIMITIVE.into());

    Box::new(arr)
}

/// Take kernel for single chunk without nulls and an iterator as index.
///
/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
#[inline]
pub unsafe fn take_no_null_bool_iter_unchecked<I: IntoIterator<Item = usize>>(
    arr: &BooleanArray,
    indices: I,
) -> Box<BooleanArray> {
    debug_assert!(!arr.has_validity());
    let values = arr.values();

    let iter = indices.into_iter().map(|idx| {
        debug_assert!(idx < values.len());
        values.get_bit_unchecked(idx)
    });
    let mutable = MutableBitmap::from_trusted_len_iter_unchecked(iter);
    Box::new(BooleanArray::new(DataType::Boolean, mutable.into(), None))
}

/// Take kernel for single chunk and an iterator as index.
/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
#[inline]
pub unsafe fn take_bool_iter_unchecked<I: IntoIterator<Item = usize>>(
    arr: &BooleanArray,
    indices: I,
) -> Box<BooleanArray> {
    let validity = arr.validity().expect("should have nulls");

    let iter = indices.into_iter().map(|idx| {
        if validity.get_bit_unchecked(idx) {
            Some(arr.value_unchecked(idx))
        } else {
            None
        }
    });

    Box::new(BooleanArray::from_trusted_len_iter_unchecked(iter))
}

/// Take kernel for single chunk and an iterator as index.
/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
#[inline]
pub unsafe fn take_bool_opt_iter_unchecked<I: IntoIterator<Item = Option<usize>>>(
    arr: &BooleanArray,
    indices: I,
) -> Box<BooleanArray> {
    let validity = arr.validity().expect("should have nulls");
    let iter = indices.into_iter().map(|opt_idx| {
        opt_idx.and_then(|idx| {
            if validity.get_bit_unchecked(idx) {
                Some(arr.value_unchecked(idx))
            } else {
                None
            }
        })
    });

    Box::new(BooleanArray::from_trusted_len_iter_unchecked(iter))
}

/// Take kernel for single chunk without null values and an iterator as index that may produce None values.
/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
#[inline]
pub unsafe fn take_no_null_bool_opt_iter_unchecked<I: IntoIterator<Item = Option<usize>>>(
    arr: &BooleanArray,
    indices: I,
) -> Box<BooleanArray> {
    let iter = indices
        .into_iter()
        .map(|opt_idx| opt_idx.map(|idx| arr.value_unchecked(idx)));

    Box::new(BooleanArray::from_trusted_len_iter_unchecked(iter))
}

/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
#[inline]
pub unsafe fn take_no_null_utf8_iter_unchecked<I: IntoIterator<Item = usize>>(
    arr: &LargeStringArray,
    indices: I,
) -> Box<LargeStringArray> {
    let iter = indices.into_iter().map(|idx| {
        debug_assert!(idx < arr.len());
        arr.value_unchecked(idx)
    });
    Box::new(MutableUtf8Array::<i64>::from_trusted_len_values_iter_unchecked(iter).into())
}

/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
#[inline]
pub unsafe fn take_no_null_binary_iter_unchecked<I: IntoIterator<Item = usize>>(
    arr: &LargeBinaryArray,
    indices: I,
) -> Box<LargeBinaryArray> {
    let iter = indices.into_iter().map(|idx| {
        debug_assert!(idx < arr.len());
        arr.value_unchecked(idx)
    });
    Box::new(MutableBinaryArray::<i64>::from_trusted_len_values_iter_unchecked(iter).into())
}

/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
#[inline]
pub unsafe fn take_utf8_iter_unchecked<I: IntoIterator<Item = usize>>(
    arr: &LargeStringArray,
    indices: I,
) -> Box<LargeStringArray> {
    let validity = arr.validity().expect("should have nulls");
    let iter = indices.into_iter().map(|idx| {
        debug_assert!(idx < arr.len());
        if validity.get_bit_unchecked(idx) {
            Some(arr.value_unchecked(idx))
        } else {
            None
        }
    });

    Box::new(LargeStringArray::from_trusted_len_iter_unchecked(iter))
}

/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
#[inline]
pub unsafe fn take_binary_iter_unchecked<I: IntoIterator<Item = usize>>(
    arr: &LargeBinaryArray,
    indices: I,
) -> Box<LargeBinaryArray> {
    let validity = arr.validity().expect("should have nulls");
    let iter = indices.into_iter().map(|idx| {
        debug_assert!(idx < arr.len());
        if validity.get_bit_unchecked(idx) {
            Some(arr.value_unchecked(idx))
        } else {
            None
        }
    });

    Box::new(LargeBinaryArray::from_trusted_len_iter_unchecked(iter))
}

/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
#[inline]
pub unsafe fn take_no_null_utf8_opt_iter_unchecked<I: IntoIterator<Item = Option<usize>>>(
    arr: &LargeStringArray,
    indices: I,
) -> Box<LargeStringArray> {
    let iter = indices
        .into_iter()
        .map(|opt_idx| opt_idx.map(|idx| arr.value_unchecked(idx)));

    Box::new(LargeStringArray::from_trusted_len_iter_unchecked(iter))
}

/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
#[inline]
pub unsafe fn take_no_null_binary_opt_iter_unchecked<I: IntoIterator<Item = Option<usize>>>(
    arr: &LargeBinaryArray,
    indices: I,
) -> Box<LargeBinaryArray> {
    let iter = indices
        .into_iter()
        .map(|opt_idx| opt_idx.map(|idx| arr.value_unchecked(idx)));

    Box::new(LargeBinaryArray::from_trusted_len_iter_unchecked(iter))
}

/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
#[inline]
pub unsafe fn take_utf8_opt_iter_unchecked<I: IntoIterator<Item = Option<usize>>>(
    arr: &LargeStringArray,
    indices: I,
) -> Box<LargeStringArray> {
    let validity = arr.validity().expect("should have nulls");
    let iter = indices.into_iter().map(|opt_idx| {
        opt_idx.and_then(|idx| {
            if validity.get_bit_unchecked(idx) {
                Some(arr.value_unchecked(idx))
            } else {
                None
            }
        })
    });
    Box::new(LargeStringArray::from_trusted_len_iter_unchecked(iter))
}

/// # Safety
/// - no bounds checks
/// - iterator must be TrustedLen
#[inline]
pub unsafe fn take_binary_opt_iter_unchecked<I: IntoIterator<Item = Option<usize>>>(
    arr: &LargeBinaryArray,
    indices: I,
) -> Box<LargeBinaryArray> {
    let validity = arr.validity().expect("should have nulls");
    let iter = indices.into_iter().map(|opt_idx| {
        opt_idx.and_then(|idx| {
            if validity.get_bit_unchecked(idx) {
                Some(arr.value_unchecked(idx))
            } else {
                None
            }
        })
    });
    Box::new(LargeBinaryArray::from_trusted_len_iter_unchecked(iter))
}

/// # Safety
/// caller must ensure indices are in bounds
pub unsafe fn take_utf8_unchecked(
    arr: &LargeStringArray,
    indices: &IdxArr,
) -> Box<LargeStringArray> {
    let data_len = indices.len();

    let mut offset_buf = vec![0; data_len + 1];
    let offset_typed = offset_buf.as_mut_slice();

    let mut length_so_far = 0;
    offset_typed[0] = length_so_far;

    let validity;

    // The required size is yet unknown
    // Allocate 2.0 times the expected size.
    // where expected size is the length of bytes multiplied by the factor (take_len / current_len)
    let mut values_capacity = if arr.len() > 0 {
        ((arr.len() as f32 * 2.0) as usize) / arr.len() * indices.len()
    } else {
        0
    };

    // 16 bytes per string as default alloc
    let mut values_buf = Vec::<u8>::with_capacity(values_capacity);

    // both 0 nulls
    if !arr.has_validity() && !indices.has_validity() {
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
    } else if !arr.has_validity() {
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
        validity = indices.validity().cloned();
    } else {
        let mut builder = MutableUtf8Array::with_capacities(data_len, length_so_far as usize);
        let validity_arr = arr.validity().expect("should have nulls");

        if !indices.has_validity() {
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
            let validity_indices = indices.validity().expect("should have nulls");
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
        return Box::new(array);
    }

    // Safety: all "values" are &str, and thus valid utf8
    Box::new(Utf8Array::<i64>::from_data_unchecked_default(
        offset_buf.into(),
        values_buf.into(),
        validity,
    ))
}

/// # Safety
/// caller must ensure indices are in bounds
pub unsafe fn take_binary_unchecked(
    arr: &LargeBinaryArray,
    indices: &IdxArr,
) -> Box<LargeBinaryArray> {
    let data_len = indices.len();

    let mut offset_buf = vec![0; data_len + 1];
    let offset_typed = offset_buf.as_mut_slice();

    let mut length_so_far = 0;
    offset_typed[0] = length_so_far;

    let validity;

    // The required size is yet unknown
    // Allocate 2.0 times the expected size.
    // where expected size is the length of bytes multiplied by the factor (take_len / current_len)
    let mut values_capacity = if arr.len() > 0 {
        ((arr.len() as f32 * 2.0) as usize) / arr.len() * indices.len()
    } else {
        0
    };

    // 16 bytes per string as default alloc
    let mut values_buf = Vec::<u8>::with_capacity(values_capacity);

    // both 0 nulls
    if !arr.has_validity() && !indices.has_validity() {
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

                values_buf.extend_from_slice(s)
            });
        validity = None;
    } else if !arr.has_validity() {
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

                    values_buf.extend_from_slice(s)
                }
                *offset = length_so_far;
            });
        validity = indices.validity().cloned();
    } else {
        let mut builder = MutableBinaryArray::with_capacities(data_len, length_so_far as usize);
        let validity_arr = arr.validity().expect("should have nulls");

        if !indices.has_validity() {
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
            let validity_indices = indices.validity().expect("should have nulls");
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

        let array: BinaryArray<i64> = builder.into();
        return Box::new(array);
    }

    // Safety: all "values" are &str, and thus valid utf8
    Box::new(BinaryArray::<i64>::from_data_unchecked_default(
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
pub unsafe fn take_value_indices_from_list(
    list: &ListArray<i64>,
    indices: &IdxArr,
) -> (IdxArr, Offsets<i64>) {
    let offsets = list.offsets().as_slice();

    let mut new_offsets = Vec::with_capacity(indices.len());
    // will likely have at least indices.len values
    let mut values = Vec::with_capacity(indices.len());
    let mut current_offset = 0;
    // add first offset
    new_offsets.push(0);
    // compute the value indices, and set offsets accordingly

    let indices_values = indices.values();

    if !indices.has_validity() {
        for i in 0..indices.len() {
            let idx = *indices_values.get_unchecked(i) as usize;
            let start = *offsets.get_unchecked(idx);
            let end = *offsets.get_unchecked(idx + 1);
            current_offset += end - start;
            new_offsets.push(current_offset);

            let mut curr = start;

            // if start == end, this slot is empty
            while curr < end {
                values.push(curr as IdxSize);
                curr += 1;
            }
        }
    } else {
        let validity = indices.validity().expect("should have nulls");

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
                    values.push(curr as IdxSize);
                    curr += 1;
                }
            } else {
                new_offsets.push(current_offset);
            }
        }
    }

    // Safety:
    // offsets are monotonically increasing.
    unsafe {
        (
            IdxArr::from_data_default(values.into(), None),
            Offsets::new_unchecked(new_offsets),
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_utf8_kernel() {
        let s = LargeStringArray::from(vec![Some("foo"), None, Some("bar")]);
        unsafe {
            let out = take_utf8_unchecked(&s, &IdxArr::from_slice(&[1, 2]));
            assert!(out.is_null(0));
            assert!(out.is_valid(1));
            let out = take_utf8_unchecked(&s, &IdxArr::from(vec![None, Some(2)]));
            assert!(out.is_null(0));
            assert!(out.is_valid(1));
            let out = take_utf8_unchecked(&s, &IdxArr::from(vec![None, None]));
            assert!(out.is_null(0));
            assert!(out.is_null(1));
        }
    }
}
