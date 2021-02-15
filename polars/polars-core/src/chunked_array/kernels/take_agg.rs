//! kernels that combine take and aggregations.
use crate::prelude::*;
use arrow::array::{Array, PrimitiveArray};

/// Take kernel for single chunk without nulls and an iterator as index.
pub(crate) unsafe fn take_agg_no_null_primitive_iter_unchecked<
    T: PolarsNumericType,
    I: IntoIterator<Item = usize>,
    F: Fn(T::Native, T::Native) -> T::Native,
>(
    arr: &PrimitiveArray<T>,
    indices: I,
    f: F,
    init: T::Native,
) -> T::Native {
    debug_assert_eq!(arr.null_count(), 0);

    let array_values = arr.values();

    indices
        .into_iter()
        .fold(init, |acc, idx| f(acc, *array_values.get_unchecked(idx)))
}
