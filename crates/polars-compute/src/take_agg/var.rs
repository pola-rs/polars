#![allow(unsafe_op_in_unsafe_fn)]
//! Gather-and-reduce to a variance over a [`PlPrimitiveArray`].

use arrow::types::NativeType;
use num_traits::ToPrimitive;
use polars_array::PlPrimitiveArray;

use super::primitive::flat_validity;

/// Numerical stable online variance aggregation.
pub fn online_variance<I>(
    // iterator producing values
    iter: I,
    ddof: u8,
) -> Option<f64>
where
    I: IntoIterator<Item = f64>,
{
    let mut m2 = 0.0;
    let mut mean = 0.0;
    let mut count = 0u64;

    for value in iter {
        let new_count = count + 1;
        let delta_1 = value - mean;
        let new_mean = delta_1 / new_count as f64 + mean;
        let delta_2 = value - new_mean;
        let new_m2 = m2 + delta_1 * delta_2;

        count += 1;
        mean = new_mean;
        m2 = new_m2;
    }

    if count <= ddof as u64 {
        return None;
    }

    Some(m2 / (count as f64 - ddof as f64))
}

/// The variance of the values `indices` gather out of a chunk with no nulls in it.
///
/// # Safety
/// Every index must be in bounds of `arr`.
pub unsafe fn take_var_no_null_primitive_iter_unchecked<T, I>(
    arr: &PlPrimitiveArray<T>,
    indices: I,
    ddof: u8,
) -> Option<f64>
where
    T: NativeType + ToPrimitive,
    I: IntoIterator<Item = usize>,
{
    debug_assert!(arr.null_count() == 0);

    // Every index gathers the same value where the buffer holds a single slot, so the variance is
    // over that one value repeated — which is what a flat chunk of it would give as well. It is
    // not `0.0` in general: `ddof` still decides whether there are enough values at all.
    if let Some(value) = arr.scalar_values() {
        let value = unsafe { value.to_f64().unwrap_unchecked() };
        return online_variance(indices.into_iter().map(|_| value), ddof);
    }

    let values = arr.flat_values().unwrap();
    let iter = indices.into_iter().map(|idx| unsafe {
        let value = *values.get_unchecked(idx);
        value.to_f64().unwrap_unchecked()
    });
    online_variance(iter, ddof)
}

/// The variance of the non-null values `indices` gather out of a chunk.
///
/// # Safety
/// Every index must be in bounds of `arr`.
pub unsafe fn take_var_nulls_primitive_iter_unchecked<T, I>(
    arr: &PlPrimitiveArray<T>,
    indices: I,
    ddof: u8,
) -> Option<f64>
where
    T: NativeType + ToPrimitive,
    I: IntoIterator<Item = usize>,
{
    debug_assert!(arr.null_count() > 0);

    // Every element is null, so no index gathers a value and there is no variance.
    let validity = flat_validity(arr)?;

    if let Some(value) = arr.scalar_values() {
        let iter = indices.into_iter().filter_map(|idx| {
            unsafe { validity.get_bit_unchecked(idx) }.then(|| value.to_f64())?
        });
        return online_variance(iter, ddof);
    }

    let values = arr.flat_values().unwrap();
    let iter = indices.into_iter().flat_map(|idx| unsafe {
        if validity.get_bit_unchecked(idx) {
            let value = *values.get_unchecked(idx);
            value.to_f64()
        } else {
            None
        }
    });
    online_variance(iter, ddof)
}
