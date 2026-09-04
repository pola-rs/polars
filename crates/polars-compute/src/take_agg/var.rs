#![allow(unsafe_op_in_unsafe_fn)]
//! Gather-and-reduce to a variance over a [`PlPrimitiveArray`].

use arrow::types::NativeType;
use num_traits::ToPrimitive;
use polars_array::PlPrimitiveArray;

use super::primitive::{flat_validity, flat_values, repeated_value};

/// Numerical stable online variance aggregation.
///
/// See:
/// Welford, B. P. (1962). "Note on a method for calculating corrected sums of squares and products".
/// Technometrics. 4 (3): 419–420. doi:10.2307/1266577. JSTOR 1266577.
/// and:
/// Ling, Robert F. (1974). "Comparison of Several Algorithms for Computing Sample Means and Variances".
/// Journal of the American Statistical Association. 69 (348): 859–866. doi:10.2307/2286154. JSTOR 2286154.
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
    match repeated_value(arr) {
        Some(value) => {
            let value = unsafe { value.to_f64().unwrap_unchecked() };
            online_variance(indices.into_iter().map(|_| value), ddof)
        },
        None => {
            let values = flat_values(arr);
            let iter = indices.into_iter().map(|idx| unsafe {
                let value = *values.get_unchecked(idx);
                value.to_f64().unwrap_unchecked()
            });
            online_variance(iter, ddof)
        },
    }
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

    match repeated_value(arr) {
        Some(value) => {
            let iter = indices.into_iter().filter_map(|idx| {
                unsafe { validity.get_bit_unchecked(idx) }.then(|| value.to_f64())?
            });
            online_variance(iter, ddof)
        },
        None => {
            let values = flat_values(arr);
            let iter = indices.into_iter().flat_map(|idx| unsafe {
                if validity.get_bit_unchecked(idx) {
                    let value = *values.get_unchecked(idx);
                    value.to_f64()
                } else {
                    None
                }
            });
            online_variance(iter, ddof)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LENGTH: usize = 6;
    const INDICES: [usize; 4] = [0, 3, 1, 5];

    #[test]
    fn a_repeated_value_varies_the_same_either_way() {
        let scalar = PlPrimitiveArray::new_scalar(7.0f64, LENGTH);
        let flat = PlPrimitiveArray::from_vec(vec![7.0f64; LENGTH]);

        for ddof in [0, 1] {
            let scalar_var =
                unsafe { take_var_no_null_primitive_iter_unchecked(&scalar, INDICES, ddof) };
            let flat_var =
                unsafe { take_var_no_null_primitive_iter_unchecked(&flat, INDICES, ddof) };
            assert_eq!(scalar_var, flat_var);
            assert_eq!(scalar_var, Some(0.0));
        }

        // Fewer gathered values than `ddof` leaves no variance at all.
        assert_eq!(
            unsafe { take_var_no_null_primitive_iter_unchecked(&scalar, [0usize], 1) },
            None,
        );
    }

    #[test]
    fn a_repeated_null_varies_not_at_all() {
        let arr = PlPrimitiveArray::<f64>::new_full_null(LENGTH);
        assert_eq!(
            unsafe { take_var_nulls_primitive_iter_unchecked(&arr, INDICES, 0) },
            None,
        );
    }

    #[test]
    fn a_repeated_value_under_a_flat_mask() {
        let mask = [true, false, true, true, false, true];
        let scalar = PlPrimitiveArray::new_scalar(7.0f64, LENGTH)
            .with_validity(Some(mask.into_iter().collect()));
        let flat = PlPrimitiveArray::from_vec(vec![7.0f64; LENGTH])
            .with_validity(Some(mask.into_iter().collect()));

        assert_eq!(
            unsafe { take_var_nulls_primitive_iter_unchecked(&scalar, INDICES, 0) },
            unsafe { take_var_nulls_primitive_iter_unchecked(&flat, INDICES, 0) },
        );
    }
}
