use super::*;
use crate::data_types::IsFloat;
use crate::utils::CustomIterTools;
use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::datatypes::DataType;
use arrow::types::NativeType;
use num::{Bounded, Float, NumCast};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::ops::Mul;
use std::sync::Arc;

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

fn rolling_apply_weights<T, Fo, Fa>(
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

fn rolling_apply<T, K, Fo, Fa>(
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

pub(crate) fn compute_min<T>(values: &[T]) -> T
where
    T: NativeType + PartialOrd + IsFloat + Bounded,
{
    let mut min = T::max_value();

    for &v in values {
        if T::is_float() && v.is_nan() {
            return v;
        }
        if v < min {
            min = v
        }
    }
    min
}

pub(crate) fn compute_min_weights<T>(values: &[T], weights: &[T]) -> T
where
    T: NativeType + PartialOrd + std::ops::Mul<Output = T>,
{
    values
        .iter()
        .zip(weights)
        .map(|(v, w)| *v * *w)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

pub(crate) fn compute_max<T>(values: &[T]) -> T
where
    T: NativeType + PartialOrd + IsFloat + Bounded,
{
    let mut max = T::min_value();

    for &v in values {
        if T::is_float() && v.is_nan() {
            return v;
        }
        if v > max {
            max = v
        }
    }
    max
}

pub(crate) fn compute_max_weights<T>(values: &[T], weights: &[T]) -> T
where
    T: NativeType + PartialOrd + IsFloat + Bounded + Mul<Output = T>,
{
    let mut max = T::min_value();
    for v in values.iter().zip(weights).map(|(v, w)| *v * *w) {
        if T::is_float() && v.is_nan() {
            return v;
        }
        if v > max {
            max = v
        }
    }

    max
}

fn coerce_weights<T: NumCast>(weights: &[f64]) -> Vec<T>
where
{
    weights
        .iter()
        .map(|v| NumCast::from(*v).unwrap())
        .collect::<Vec<_>>()
}

pub fn rolling_mean<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + Float + std::iter::Sum<T>,
{
    match (center, weights) {
        (true, None) => rolling_apply(
            values,
            window_size,
            min_periods,
            det_offsets_center,
            compute_mean,
        ),
        (false, None) => rolling_apply(values, window_size, min_periods, det_offsets, compute_mean),
        (true, Some(weights)) => {
            let weights = coerce_weights(weights);
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_mean_weights,
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
                compute_mean_weights,
                &weights,
            )
        }
    }
}

pub fn rolling_min<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + PartialOrd + NumCast + Mul<Output = T> + Bounded + IsFloat,
{
    match (center, weights) {
        (true, None) => rolling_apply(
            values,
            window_size,
            min_periods,
            det_offsets_center,
            compute_min,
        ),
        (false, None) => rolling_apply(values, window_size, min_periods, det_offsets, compute_min),
        (true, Some(weights)) => {
            assert!(
                T::is_float(),
                "implementation error, should only be reachable by float types"
            );
            let weights = weights
                .iter()
                .map(|v| NumCast::from(*v).unwrap())
                .collect::<Vec<_>>();
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_min_weights,
                &weights,
            )
        }
        (false, Some(weights)) => {
            assert!(
                T::is_float(),
                "implementation error, should only be reachable by float types"
            );
            let weights = weights
                .iter()
                .map(|v| NumCast::from(*v).unwrap())
                .collect::<Vec<_>>();
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets,
                compute_min_weights,
                &weights,
            )
        }
    }
}

pub fn rolling_max<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + PartialOrd + IsFloat + Bounded + NumCast + Mul<Output = T>,
{
    match (center, weights) {
        (true, None) => rolling_apply(
            values,
            window_size,
            min_periods,
            det_offsets_center,
            compute_max,
        ),
        (false, None) => rolling_apply(values, window_size, min_periods, det_offsets, compute_max),
        (true, Some(weights)) => {
            assert!(
                T::is_float(),
                "implementation error, should only be reachable by float types"
            );
            let weights = weights
                .iter()
                .map(|v| NumCast::from(*v).unwrap())
                .collect::<Vec<_>>();
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_max_weights,
                &weights,
            )
        }
        (false, Some(weights)) => {
            assert!(
                T::is_float(),
                "implementation error, should only be reachable by float types"
            );
            let weights = weights
                .iter()
                .map(|v| NumCast::from(*v).unwrap())
                .collect::<Vec<_>>();
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets,
                compute_max_weights,
                &weights,
            )
        }
    }
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

pub fn rolling_sum<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum + NumCast + Mul<Output = T>,
{
    match (center, weights) {
        (true, None) => rolling_apply(
            values,
            window_size,
            min_periods,
            det_offsets_center,
            compute_sum,
        ),
        (false, None) => rolling_apply(values, window_size, min_periods, det_offsets, compute_sum),
        (true, Some(weights)) => {
            let weights = coerce_weights(weights);
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_sum_weights,
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
                compute_sum_weights,
                &weights,
            )
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rolling_sum() {
        let values = &[1.0, 2.0, 3.0, 4.0];

        let out = rolling_sum(values, 2, 2, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(3.0), Some(5.0), Some(7.0)]);

        let out = rolling_sum(values, 2, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(3.0), Some(5.0), Some(7.0)]);

        let out = rolling_sum(values, 4, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(3.0), Some(6.0), Some(10.0)]);

        let out = rolling_sum(values, 4, 1, true, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(3.0), Some(6.0), Some(10.0), Some(9.0)]);

        let out = rolling_sum(values, 4, 4, true, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, Some(10.0), None]);
    }
}
