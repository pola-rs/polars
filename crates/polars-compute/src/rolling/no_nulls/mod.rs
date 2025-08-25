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
            let weights_start = if centered {
                // When using centered weights, we need to find the right location
                // in the weights array specifically by aligning the center of the
                // window with idx, to handle cases where the window is smaller than
                // weights array.
                let center = (window_size / 2) as isize;
                let offset = center - (idx as isize - start as isize);
                offset.max(0) as usize
            } else if start == 0 {
                // When start is 0, we need to work backwards from the end of the
                // weights array to ensure we are lined up correctly (since the
                // start of the values array is implicitly cut off)
                weights.len() - win_len
            } else {
                0
            };
            let weights_slice = &weights[weights_start..weights_start + win_len];
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
