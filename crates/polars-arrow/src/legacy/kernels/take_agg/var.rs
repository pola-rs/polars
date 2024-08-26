use super::*;

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

/// Take kernel for single chunk and an iterator as index.
/// # Safety
/// caller must ensure iterators indexes are in bounds
pub unsafe fn take_var_no_null_primitive_iter_unchecked<T, I>(
    arr: &PrimitiveArray<T>,
    indices: I,
    ddof: u8,
) -> Option<f64>
where
    T: NativeType + ToPrimitive,
    I: IntoIterator<Item = usize>,
{
    debug_assert!(arr.null_count() == 0);
    let array_values = arr.values().as_slice();
    let iter = unsafe {
        indices.into_iter().map(|idx| {
            let value = *array_values.get_unchecked(idx);
            value.to_f64().unwrap_unchecked()
        })
    };
    online_variance(iter, ddof)
}

/// Take kernel for single chunk and an iterator as index.
/// # Safety
/// caller must ensure iterators indexes are in bounds
pub unsafe fn take_var_nulls_primitive_iter_unchecked<T, I>(
    arr: &PrimitiveArray<T>,
    indices: I,
    ddof: u8,
) -> Option<f64>
where
    T: NativeType + ToPrimitive,
    I: IntoIterator<Item = usize>,
{
    debug_assert!(arr.null_count() > 0);
    let array_values = arr.values().as_slice();
    let validity = arr.validity().unwrap();

    let iter = unsafe {
        indices.into_iter().flat_map(|idx| {
            if validity.get_bit_unchecked(idx) {
                let value = *array_values.get_unchecked(idx);
                value.to_f64()
            } else {
                None
            }
        })
    };
    online_variance(iter, ddof)
}
