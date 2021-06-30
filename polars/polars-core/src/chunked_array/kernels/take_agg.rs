//! kernels that combine take and aggregations.
use crate::prelude::*;
use arrow::array::{Array, PrimitiveArray};
use arrow::types::NativeType;

/// Take kernel for single chunk without nulls and an iterator as index.
pub(crate) unsafe fn take_agg_no_null_primitive_iter_unchecked<
    T: NativeType,
    I: IntoIterator<Item = usize>,
    F: Fn(T, T) -> T,
>(
    arr: &PrimitiveArray<T>,
    indices: I,
    f: F,
    init: T,
) -> T {
    debug_assert_eq!(arr.null_count(), 0);

    let array_values = arr.values().as_slice();

    indices
        .into_iter()
        .fold(init, |acc, idx| f(acc, *array_values.get_unchecked(idx)))
}

/// Take kernel for single chunk and an iterator as index.
pub(crate) unsafe fn take_agg_primitive_iter_unchecked<
    T: PolarsNumericType,
    I: IntoIterator<Item = usize>,
    F: Fn(T::Native, T::Native) -> T::Native,
>(
    arr: &PrimitiveArray<T::Native>,
    indices: I,
    f: F,
    init: T::Native,
) -> Option<T::Native> {
    if arr.null_count() == arr.len() {
        return None;
    }

    let array_values = arr.values().as_slice();
    let validity = arr
        .validity()
        .as_ref()
        .expect("null buffer should be there");

    let out = indices.into_iter().fold(init, |acc, idx| {
        if validity.get_bit_unchecked(idx) {
            f(acc, *array_values.get_unchecked(idx))
        } else {
            acc
        }
    });
    if out == init {
        None
    } else {
        Some(out)
    }
}

/// Take kernel for single chunk and an iterator as index.
pub(crate) unsafe fn take_agg_primitive_iter_unchecked_count_nulls<
    T: PolarsNumericType,
    I: IntoIterator<Item = usize>,
    F: Fn(T::Native, T::Native) -> T::Native,
>(
    arr: &PrimitiveArray<T::Native>,
    indices: I,
    f: F,
    init: T::Native,
) -> Option<(T::Native, u32)> {
    if arr.null_count() == arr.len() {
        return None;
    }

    let array_values = arr.values().as_slice();
    let validity = arr
        .validity()
        .as_ref()
        .expect("null buffer should be there");

    let mut null_count = 0;
    let out = indices.into_iter().fold(init, |acc, idx| {
        if validity.get_bit_unchecked(idx) {
            f(acc, *array_values.get_unchecked(idx))
        } else {
            null_count += 1;
            acc
        }
    });
    if out == init {
        None
    } else {
        Some((out, null_count))
    }
}
