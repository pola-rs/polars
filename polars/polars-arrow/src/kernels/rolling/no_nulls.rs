use super::*;
use crate::utils::CustomIterTools;
use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::datatypes::DataType;
use arrow::types::NativeType;
use num::{Float, NumCast};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::sync::Arc;

pub use mean_no_nulls::rolling_mean;
pub use min_max_no_nulls::{rolling_max, rolling_min};
pub use quantile_no_nulls::{rolling_median, rolling_quantile};
pub use sum_no_nulls::rolling_sum;

pub(crate) trait RollingAggWindow<'a, T: NativeType> {
    fn new(slice: &'a [T], start: usize, end: usize) -> Self;

    unsafe fn update(&mut self, start: usize, end: usize) -> T;
}

// Use an aggregation window that maintains the state
pub(super) fn rolling_apply_agg_window<'a, Agg, T, Fo>(
    values: &'a [T],
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    Agg: RollingAggWindow<'a, T>,
    T: Debug + IsFloat + NativeType,
{
    let len = values.len();
    let (start, end) = det_offsets_fn(0, window_size, len);
    let mut agg_window = Agg::new(values, start, end);

    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            // safety:
            // we are in bounds
            unsafe { agg_window.update(start, end) }
        })
        .collect_trusted::<Vec<_>>();

    let validity = create_validity(min_periods, len as usize, window_size, det_offsets_fn);
    Arc::new(PrimitiveArray::from_data(
        T::PRIMITIVE.into(),
        out.into(),
        validity.map(|b| b.into()),
    ))
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum QuantileInterpolOptions {
    Nearest,
    Lower,
    Higher,
    Midpoint,
    Linear,
}

impl Default for QuantileInterpolOptions {
    fn default() -> Self {
        QuantileInterpolOptions::Nearest
    }
}

pub(super) fn rolling_apply_weights<T, Fo, Fa>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    aggregator: Fa,
    weights: &[T],
) -> ArrayRef
where
    T: NativeType,
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    Fa: Fn(&[T], &[T]) -> T,
{
    assert_eq!(weights.len(), window_size);
    let len = values.len();
    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            let vals = unsafe { values.get_unchecked(start..end) };

            aggregator(vals, weights)
        })
        .collect_trusted::<Vec<T>>();

    let validity = create_validity(min_periods, len as usize, window_size, det_offsets_fn);
    Arc::new(PrimitiveArray::from_data(
        DataType::from(T::PRIMITIVE),
        out.into(),
        validity.map(|b| b.into()),
    ))
}

pub(super) fn rolling_apply<T, K, Fo, Fa>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    aggregator: Fa,
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    Fa: Fn(&[T]) -> K,
    K: NativeType,
    T: Debug,
{
    let len = values.len();
    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            let vals = unsafe { values.get_unchecked(start..end) };
            aggregator(vals)
        })
        .collect_trusted::<Vec<K>>();

    let validity = create_validity(min_periods, len as usize, window_size, det_offsets_fn);
    Arc::new(PrimitiveArray::from_data(
        K::PRIMITIVE.into(),
        out.into(),
        validity.map(|b| b.into()),
    ))
}

pub(crate) fn compute_var<T>(vals: &[T]) -> T
where
    T: Float + std::ops::AddAssign + std::fmt::Debug,
{
    let mut count = T::zero();
    let mut sum = T::zero();
    let mut sum_of_squares = T::zero();

    for &val in vals {
        sum += val;
        sum_of_squares += val * val;
        count += T::one();
    }

    let mean = sum / count;
    // apply Bessel's correction
    ((sum_of_squares / count) - mean * mean) / (count - T::one()) * count
}

fn compute_var_weights<T>(vals: &[T], weights: &[T]) -> T
where
    T: Float + std::ops::AddAssign,
{
    let weighted_iter = vals.iter().zip(weights).map(|(x, y)| *x * *y);

    let mut count = T::zero();
    let mut sum = T::zero();
    let mut sum_of_squares = T::zero();

    for val in weighted_iter {
        sum += val;
        sum_of_squares += val * val;
        count += T::one();
    }

    let mean = sum / count;
    // apply Bessel's correction
    ((sum_of_squares / count) - mean * mean) / (count - T::one()) * count
}

pub(crate) fn compute_mean<T>(values: &[T]) -> T
where
    T: Float + std::iter::Sum<T>,
{
    values.iter().copied().sum::<T>() / T::from(values.len()).unwrap()
}

pub(crate) fn compute_mean_weights<T>(values: &[T], weights: &[T]) -> T
where
    T: Float + std::iter::Sum<T>,
{
    values.iter().zip(weights).map(|(v, w)| *v * *w).sum::<T>() / T::from(values.len()).unwrap()
}

pub(crate) fn compute_sum<T>(values: &[T]) -> T
where
    T: std::iter::Sum<T> + Copy,
{
    values.iter().copied().sum()
}

pub(crate) fn compute_sum_weights<T>(values: &[T], weights: &[T]) -> T
where
    T: std::iter::Sum<T> + Copy + std::ops::Mul<Output = T>,
{
    values.iter().zip(weights).map(|(v, w)| *v * *w).sum()
}

pub(super) fn coerce_weights<T: NumCast>(weights: &[f64]) -> Vec<T>
where
{
    weights
        .iter()
        .map(|v| NumCast::from(*v).unwrap())
        .collect::<Vec<_>>()
}

pub fn rolling_var<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + Float + std::ops::AddAssign,
{
    match (center, weights) {
        (true, None) => rolling_apply(
            values,
            window_size,
            min_periods,
            det_offsets_center,
            compute_var,
        ),
        (false, None) => rolling_apply(values, window_size, min_periods, det_offsets, compute_var),
        (true, Some(weights)) => {
            let weights = coerce_weights(weights);
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_var_weights,
                &weights,
            )
        }
        (false, Some(weights)) => {
            let weights = coerce_weights(weights);
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets,
                compute_var_weights,
                &weights,
            )
        }
    }
}
