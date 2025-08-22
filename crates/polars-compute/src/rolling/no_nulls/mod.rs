mod mean;
mod min_max;
mod moment;
mod quantile;
mod sum;
use std::fmt::Debug;

use arrow::array::PrimitiveArray;
use arrow::datatypes::ArrowDataType;
use arrow::legacy::error::PolarsResult;
use arrow::legacy::utils::CustomIterTools;
use arrow::types::NativeType;
pub use mean::*;
pub use min_max::*;
pub use moment::*;
use num_traits::{Float, Num, NumCast};
pub use quantile::*;
pub use sum::*;

use super::*;

pub trait RollingAggWindowNoNulls<'a, T: NativeType> {
    fn new(
        slice: &'a [T],
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        window_size: Option<usize>,
    ) -> Self;

    /// Update and recompute the window
    ///
    /// # Safety
    /// `start` and `end` must be within the windows bounds
    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T>;
}

// Use an aggregation window that maintains the state
pub(super) fn rolling_apply_agg_window<'a, Agg, T, Fo>(
    values: &'a [T],
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    params: Option<RollingFnParams>,
) -> PolarsResult<ArrayRef>
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    Agg: RollingAggWindowNoNulls<'a, T>,
    T: Debug + NativeType + Num,
{
    let len = values.len();
    let (start, end) = det_offsets_fn(0, window_size, len);
    let mut agg_window = Agg::new(values, start, end, params, Some(window_size));
    if let Some(validity) = create_validity(min_periods, len, window_size, &det_offsets_fn) {
        if validity.iter().all(|x| !x) {
            return Ok(Box::new(PrimitiveArray::<T>::new_null(
                T::PRIMITIVE.into(),
                len,
            )));
        }
    }

    let out = (0..len).map(|idx| {
        let (start, end) = det_offsets_fn(idx, window_size, len);
        if end - start < min_periods {
            None
        } else {
            // SAFETY:
            // we are in bounds
            unsafe { agg_window.update(start, end) }
        }
    });
    let arr = PrimitiveArray::from_trusted_len_iter(out);
    Ok(Box::new(arr))
}

pub(super) fn rolling_apply_weights<T, Fo, Fa>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    aggregator: Fa,
    weights: &[T],
    centered: bool,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + num_traits::Zero + std::ops::Div<Output = T> + Copy,
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    Fa: Fn(&[T], &[T]) -> T,
{
    assert_eq!(weights.len(), window_size);
    let len = values.len();
    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            let vals = unsafe { values.get_unchecked(start..end) };
            let win_len = end - start;
            let weights_slice = if start == 0 && end == len && centered {
                // Edge case: There are fewer elements than the window size
                // in the array. We'll get the same array of values at each idx
                // so we need to use the idx to know how to slide the weights
                // along when centered.
                let slice_start = weights.len() - win_len - idx;
                let slice_end = slice_start + win_len;
                &weights[slice_start..slice_end]
            } else if start == 0 {
                // Truncated at the start: take the last win_len weights
                &weights[weights.len() - win_len..]
            } else if end == len {
                // Truncated at the end: take the first win_len weights
                &weights[..win_len]
            } else {
                // Full window
                weights
            };
            aggregator(vals, weights_slice)
        })
        .collect_trusted::<Vec<T>>();

    let validity = create_validity(min_periods, len, window_size, det_offsets_fn);
    Ok(Box::new(PrimitiveArray::new(
        ArrowDataType::from(T::PRIMITIVE),
        out.into(),
        validity.map(|b| b.into()),
    )))
}

fn compute_var_weights<T>(vals: &[T], weights: &[T]) -> T
where
    T: Float + std::ops::AddAssign,
{
    // Compute weighted mean and weighted sum of squares in a single pass
    let (wssq, wmean, total_weight) = vals.iter().zip(weights).fold(
        (T::zero(), T::zero(), T::zero()),
        |(wssq, wsum, wtot), (&v, &w)| (wssq + v * v * w, wsum + v * w, wtot + w),
    );
    if total_weight.is_zero() {
        panic!("Weighted variance is undefined if weights sum to 0");
    }
    let mean = wmean / total_weight;
    (wssq / total_weight) - (mean * mean)
}

pub(crate) fn compute_sum_weights<T>(values: &[T], weights: &[T]) -> T
where
    T: std::iter::Sum<T> + Copy + std::ops::Mul<Output = T>,
{
    values.iter().zip(weights).map(|(v, w)| *v * *w).sum()
}

/// Compute the weighted mean of values, given weights (not necessarily normalized).
/// Returns sum_i(values[i] * weights[i]) / sum_i(weights[i])
pub(crate) fn compute_mean_weights<T>(values: &[T], weights: &[T]) -> T
where
    T: std::iter::Sum<T>
        + Copy
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + num_traits::Zero,
{
    let (weighted_sum, total_weight) = values
        .iter()
        .zip(weights)
        .fold((T::zero(), T::zero()), |(wsum, wtot), (&v, &w)| {
            (wsum + v * w, wtot + w)
        });
    if total_weight.is_zero() {
        panic!("Weighted mean is undefined if weights sum to 0");
    }
    weighted_sum / total_weight
}

pub(super) fn coerce_weights<T: NumCast>(weights: &[f64]) -> Vec<T>
where
{
    weights
        .iter()
        .map(|v| NumCast::from(*v).unwrap())
        .collect::<Vec<_>>()
}
