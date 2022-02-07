use super::*;
use crate::utils::CustomIterTools;
use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::datatypes::DataType;
use arrow::types::NativeType;
use num::{Float, NumCast, ToPrimitive, Zero};
use std::any::Any;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Debug)]
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

fn rolling_apply_weights<Fo, Fa>(
    values: &[f64],
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    aggregator: Fa,
    weights: &[f64],
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    Fa: Fn(&[f64], &[f64]) -> f64,
{
    assert_eq!(weights.len(), window_size);
    let len = values.len();
    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            let vals = unsafe { values.get_unchecked(start..end) };

            aggregator(vals, weights)
        })
        .collect_trusted::<Vec<f64>>();

    let validity = create_validity(min_periods, len as usize, window_size, det_offsets_fn);
    Arc::new(PrimitiveArray::from_data(
        DataType::Float64,
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

fn rolling_apply_quantile<T, Fo, Fa>(
    values: &[T],
    quantile: f64,
    interpolation: QuantileInterpolOptions,
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    aggregator: Fa,
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    Fa: Fn(&[T], f64, QuantileInterpolOptions) -> T,
    T: Debug + NativeType,
{
    let len = values.len();
    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            let vals = unsafe { values.get_unchecked(start..end) };
            aggregator(vals, quantile, interpolation)
        })
        .collect_trusted::<Vec<T>>();

    let validity = create_validity(min_periods, len as usize, window_size, det_offsets_fn);
    Arc::new(PrimitiveArray::from_data(
        T::PRIMITIVE.into(),
        out.into(),
        validity.map(|b| b.into()),
    ))
}

#[allow(clippy::too_many_arguments)]
fn rolling_apply_convolve_quantile<T, Fo, Fa>(
    values: &[T],
    quantile: f64,
    interpolation: QuantileInterpolOptions,
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    aggregator: Fa,
    weights: &[f64],
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    Fa: Fn(&[T], f64, QuantileInterpolOptions) -> T,
    T: Debug + NativeType + Mul<Output = T> + NumCast + ToPrimitive + Zero,
{
    assert_eq!(weights.len(), window_size);
    let mut buf = vec![T::zero(); window_size];
    let len = values.len();
    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            let vals = unsafe { values.get_unchecked(start..end) };
            buf.iter_mut()
                .zip(vals.iter().zip(weights))
                .for_each(|(b, (v, w))| *b = *v * NumCast::from(*w).unwrap());

            aggregator(&buf, quantile, interpolation)
        })
        .collect_trusted::<Vec<T>>();

    let validity = create_validity(min_periods, len as usize, window_size, det_offsets_fn);
    Arc::new(PrimitiveArray::from_data(
        T::PRIMITIVE.into(),
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

pub fn rolling_quantile<T>(
    values: &[T],
    quantile: f64,
    interpolation: QuantileInterpolOptions,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType
        + std::iter::Sum<T>
        + std::cmp::PartialOrd
        + num::ToPrimitive
        + NumCast
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + Zero,
{
    match (center, weights) {
        (true, None) => rolling_apply_quantile(
            values,
            quantile,
            interpolation,
            window_size,
            min_periods,
            det_offsets_center,
            compute_quantile,
        ),
        (false, None) => rolling_apply_quantile(
            values,
            quantile,
            interpolation,
            window_size,
            min_periods,
            det_offsets,
            compute_quantile,
        ),
        (true, Some(weights)) => rolling_apply_convolve_quantile(
            values,
            quantile,
            interpolation,
            window_size,
            min_periods,
            det_offsets_center,
            compute_quantile,
            weights,
        ),
        (false, Some(weights)) => rolling_apply_convolve_quantile(
            values,
            quantile,
            interpolation,
            window_size,
            min_periods,
            det_offsets,
            compute_quantile,
            weights,
        ),
    }
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

pub(crate) fn compute_quantile<T>(
    values: &[T],
    quantile: f64,
    interpolation: QuantileInterpolOptions,
) -> T
where
    T: std::iter::Sum<T>
        + Copy
        + std::cmp::PartialOrd
        + num::ToPrimitive
        + NumCast
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Mul<Output = T>,
{
    if !(0.0..=1.0).contains(&quantile) {
        panic!("quantile should be between 0.0 and 1.0");
    }

    let mut vals: Vec<T> = values
        .iter()
        .copied()
        .map(|x| NumCast::from(x).unwrap())
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let length = vals.len();

    let mut idx = match interpolation {
        QuantileInterpolOptions::Nearest => ((length as f64) * quantile) as usize,
        QuantileInterpolOptions::Lower
        | QuantileInterpolOptions::Midpoint
        | QuantileInterpolOptions::Linear => ((length as f64 - 1.0) * quantile).floor() as usize,
        QuantileInterpolOptions::Higher => ((length as f64 - 1.0) * quantile).ceil() as usize,
    };

    idx = std::cmp::min(idx, length - 1);

    match interpolation {
        QuantileInterpolOptions::Midpoint => {
            let top_idx = ((length as f64 - 1.0) * quantile).ceil() as usize;
            if top_idx == idx {
                vals[idx]
            } else {
                (vals[idx] + vals[idx + 1]) / T::from::<f64>(2.0f64).unwrap()
            }
        }
        QuantileInterpolOptions::Linear => {
            let float_idx = (length as f64 - 1.0) * quantile;
            let top_idx = f64::ceil(float_idx) as usize;

            if top_idx == idx {
                vals[idx]
            } else {
                let proportion = T::from(float_idx - idx as f64).unwrap();
                proportion * (vals[top_idx] - vals[idx]) + vals[idx]
            }
        }
        _ => vals[idx],
    }
}

pub(crate) fn compute_min<T>(values: &[T]) -> T
where
    T: NativeType + PartialOrd,
{
    values
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
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
    T: NativeType + PartialOrd,
{
    values
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

pub(crate) fn compute_max_weights<T>(values: &[T], weights: &[T]) -> T
where
    T: NativeType + PartialOrd + std::ops::Mul<Output = T>,
{
    values
        .iter()
        .zip(weights)
        .map(|(v, w)| *v * *w)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

fn as_floats<T>(values: &[T]) -> &[f64]
where
    T: Any,
{
    let values_any = &values[0] as &dyn Any;
    // couldn't use downcast_ref because the slice is unsized
    if values_any.is::<f64>() {
        unsafe { std::mem::transmute::<&[T], &[f64]>(values) }
    } else {
        panic!()
    }
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
            let values = as_floats(values);
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_mean_weights,
                weights,
            )
        }
        (false, Some(weights)) => {
            let values = as_floats(values);
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets,
                compute_mean_weights,
                weights,
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
    T: NativeType + PartialOrd,
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
            let values = as_floats(values);
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_min_weights,
                weights,
            )
        }
        (false, Some(weights)) => {
            let values = as_floats(values);
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets,
                compute_min_weights,
                weights,
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
    T: NativeType + PartialOrd,
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
            let values = as_floats(values);
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_max_weights,
                weights,
            )
        }
        (false, Some(weights)) => {
            let values = as_floats(values);
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets,
                compute_max_weights,
                weights,
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
            let values = as_floats(values);
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_var_weights,
                weights,
            )
        }
        (false, Some(weights)) => {
            let values = as_floats(values);
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets,
                compute_var_weights,
                weights,
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
    T: NativeType + std::iter::Sum + Debug,
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
            let values = as_floats(values);
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_sum_weights,
                weights,
            )
        }
        (false, Some(weights)) => {
            let values = as_floats(values);
            rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets,
                compute_sum_weights,
                weights,
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

    #[test]
    fn test_rolling_median() {
        let values = &[1.0, 2.0, 3.0, 4.0];

        let out = rolling_quantile(
            values,
            0.5,
            QuantileInterpolOptions::Linear,
            2,
            2,
            false,
            None,
        );
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(1.5), Some(2.5), Some(3.5)]);

        let out = rolling_quantile(
            values,
            0.5,
            QuantileInterpolOptions::Linear,
            2,
            1,
            false,
            None,
        );
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.5), Some(2.5), Some(3.5)]);

        let out = rolling_quantile(
            values,
            0.5,
            QuantileInterpolOptions::Linear,
            4,
            1,
            false,
            None,
        );
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.5), Some(2.0), Some(2.5)]);

        let out = rolling_quantile(
            values,
            0.5,
            QuantileInterpolOptions::Linear,
            4,
            1,
            true,
            None,
        );
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.5), Some(2.0), Some(2.5), Some(3.0)]);

        let out = rolling_quantile(
            values,
            0.5,
            QuantileInterpolOptions::Linear,
            4,
            4,
            true,
            None,
        );
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, Some(2.5), None]);
    }

    #[test]
    fn test_rolling_quantile_limits() {
        let values = &[1.0, 2.0, 3.0, 4.0];

        let interpol_options = vec![
            QuantileInterpolOptions::Lower,
            QuantileInterpolOptions::Higher,
            QuantileInterpolOptions::Nearest,
            QuantileInterpolOptions::Midpoint,
            QuantileInterpolOptions::Linear,
        ];

        for interpol in interpol_options {
            let out1 = rolling_min(values, 2, 2, false, None);
            let out1 = out1.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out1 = out1.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            let out2 = rolling_quantile(values, 0.0, interpol, 2, 2, false, None);
            let out2 = out2.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out2 = out2.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            assert_eq!(out1, out2);

            let out1 = rolling_max(values, 2, 2, false, None);
            let out1 = out1.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out1 = out1.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            let out2 = rolling_quantile(values, 1.0, interpol, 2, 2, false, None);
            let out2 = out2.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out2 = out2.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            assert_eq!(out1, out2);
        }
    }
}
