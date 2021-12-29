use super::*;
use crate::utils::CustomIterTools;
use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::datatypes::DataType;
use arrow::types::NativeType;
use num::{Float, NumCast};
use std::any::Any;
use std::fmt::Debug;
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

fn rolling_apply_convolve<Fo, Fa>(
    values: &[f64],
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    aggregator: Fa,
    weights: &[f64],
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    Fa: Fn(&[f64]) -> f64,
{
    assert_eq!(weights.len(), window_size);
    let mut buf = vec![0.0; window_size];
    let len = values.len();
    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            let vals = unsafe { values.get_unchecked(start..end) };
            buf.iter_mut()
                .zip(vals.iter().zip(weights))
                .for_each(|(b, (v, w))| *b = *v * *w);

            aggregator(&buf)
        })
        .collect_trusted::<Vec<f64>>();

    let validity = create_validity(min_periods, len as usize, window_size, det_offsets_fn);
    Arc::new(PrimitiveArray::from_data(
        DataType::Float64,
        out.into(),
        validity.map(|b| b.into()),
    ))
}

#[allow(clippy::too_many_arguments)]
fn rolling_apply_pairs<T, K, Fo, Fa>(
    values: &[T],
    argument: f64,
    interpolation: QuantileInterpolOptions,
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    aggregator: Fa,
    weights: &[f64],
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    Fa: Fn(&[(T, f64)], f64, QuantileInterpolOptions) -> K,
    K: NativeType,
    T: Debug + Copy,
{
    assert_eq!(weights.len(), window_size);
    let mut buf: Vec<(T, f64)> = Vec::with_capacity(window_size);
    let len = values.len();
    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            let vals = unsafe { values.get_unchecked(start..end) };
            buf = vals
                .iter()
                .zip(weights.iter())
                .map(|(&x, &y)| (x, y))
                .collect();

            aggregator(&buf, argument, interpolation)
        })
        .collect_trusted::<MutableBuffer<K>>();

    let validity = create_validity(min_periods, len as usize, window_size, det_offsets_fn);
    Arc::new(PrimitiveArray::from_data(
        DataType::Float64,
        out.into(),
        validity.map(|b| b.into()),
    ))
}

fn rolling_apply_pairs<T, K, Fo, Fa>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    aggregator: Fa,
    weights: &[f64],
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    Fa: Fn(&[(T, f64)]) -> K,
    K: NativeType,
    T: Debug + Copy,
{
    assert_eq!(weights.len(), window_size);
    let mut buf: Vec<(T, f64)> = Vec::with_capacity(window_size);
    let len = values.len();
    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            let vals = unsafe { values.get_unchecked(start..end) };
            buf = vals
                .iter()
                .zip(weights.iter())
                .map(|(&x, &y)| (x, y))
                .collect();

            aggregator(&buf)
        })
        .collect_trusted::<MutableBuffer<K>>();

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
    Fa: Fn(&[T], f64, QuantileInterpolOptions) -> f64,
    T: Debug,
{
    let len = values.len();
    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            let vals = unsafe { values.get_unchecked(start..end) };
            aggregator(vals, quantile, interpolation)
        })
        .collect_trusted::<MutableBuffer<f64>>();

    let validity = create_validity(min_periods, len as usize, window_size, det_offsets_fn);
    Arc::new(PrimitiveArray::from_data(
        DataType::Float64,
        out.into(),
        validity.map(|b| b.into()),
    ))
}

pub(crate) fn compute_var<T>(vals: &[T]) -> T
where
    T: Float + std::iter::Sum,
{
    let len = T::from(vals.len()).unwrap();
    let mean = vals.iter().copied().sum::<T>() / len;

    let mut sum = T::zero();
    for &val in vals {
        let v = val - mean;
        sum = sum + v * v
    }
    sum / (len - T::one())
}

pub(crate) fn compute_mean<T>(values: &[T]) -> T
where
    T: Float + std::iter::Sum<T>,
{
    values.iter().copied().sum::<T>() / T::from(values.len()).unwrap()
}

pub(crate) fn compute_quantile<T>(
    values: &[T],
    quantile: f64,
    interpolation: QuantileInterpolOptions,
) -> f64
where
    T: std::iter::Sum<T> + Copy + std::cmp::PartialOrd + num::ToPrimitive,
{
    if !(0.0..=1.0).contains(&quantile) {
        panic!("quantile should be between 0.0 and 1.0");
    }

    let mut vals: Vec<f64> = values
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
        | QuantileInterpolOptions::Linear => ((length as f64 - 1.0) * quantile) as usize,
        QuantileInterpolOptions::Higher => ((length as f64 - 1.0) * quantile).ceil() as usize,
    };

    idx = std::cmp::min(idx, length);

    match interpolation {
        QuantileInterpolOptions::Midpoint => {
            let top_idx = ((length as f64 - 1.0) * quantile).ceil() as usize;
            if top_idx == idx {
                vals[idx]
            } else {
                (vals[idx] + vals[idx + 1]) / 2.0
            }
        }
        QuantileInterpolOptions::Linear => {
            let float_idx = (length as f64 - 1.0) * quantile;
            let top_idx = f64::ceil(float_idx) as usize;

            if top_idx == idx {
                vals[idx]
            } else {
                let proportion = float_idx - idx as f64;
                proportion * (vals[top_idx] - vals[idx]) + vals[idx]
            }
        }
        _ => vals[idx],
    }
}

pub(crate) fn compute_weighted_quantile<T>(
    values: &[(T, f64)],
    quantile: f64,
    interpolation: QuantileInterpolOptions,
) -> f64
where
    T: std::iter::Sum<T> + Copy + std::cmp::PartialOrd + num::ToPrimitive,
{
    if !(0.0..=1.0).contains(&quantile) {
        panic!("quantile value must be between 0.0 and 1.0");
    }

    let mut vals: Vec<(f64, f64)> = values
        .iter()
        .copied()
        .map(|(v, w)| (NumCast::from(v).unwrap(), w))
        .collect();
    vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let length = vals.len();
    let cumulative_weights: Vec<f64> = vals
        .iter()
        .scan(0.0, |acc, x| {
            *acc += x.1;
            Some(*acc)
        })
        .collect();

    let weight_quantile = cumulative_weights[length - 1] * quantile;

    let mut idx = match interpolation {
        QuantileInterpolOptions::Nearest => {
            let upper_idx = cumulative_weights
                .iter()
                .position(|x| x >= &weight_quantile)
                .unwrap();

            let lower_idx = cumulative_weights
                .iter()
                .rposition(|x| x <= &weight_quantile)
                .unwrap();

            let upper_dist = cumulative_weights[upper_idx] - quantile;
            let lower_dist = quantile - cumulative_weights[lower_idx];

            if upper_dist < lower_dist {
                upper_idx
            } else {
                lower_idx
            }
        }
        QuantileInterpolOptions::Lower
        | QuantileInterpolOptions::Midpoint
        | QuantileInterpolOptions::Linear => cumulative_weights
            .iter()
            .rposition(|x| x <= &weight_quantile)
            .unwrap(),
        QuantileInterpolOptions::Higher => cumulative_weights
            .iter()
            .position(|x| x >= &weight_quantile)
            .unwrap(),
    };

    idx = std::cmp::min(idx, length);

    match interpolation {
        QuantileInterpolOptions::Midpoint => {
            let top_idx = cumulative_weights
                .iter()
                .position(|x| x >= &weight_quantile)
                .unwrap();

            if top_idx == idx {
                vals[idx].0
            } else {
                (vals[idx].0 + vals[top_idx].0) / 2.0
            }
        }
        QuantileInterpolOptions::Linear => {
            let top_idx = cumulative_weights
                .iter()
                .position(|x| x >= &weight_quantile)
                .unwrap();

            let upper_lower_gap = cumulative_weights[top_idx] - cumulative_weights[idx];

            if (top_idx == idx) | (upper_lower_gap == 0.0) {
                vals[idx].0
            } else {
                let upper_dist = cumulative_weights[top_idx] - quantile;
                let proportion = upper_dist / upper_lower_gap;
                proportion * (vals[top_idx].0 - vals[idx].0) + vals[idx].0
            }
        }
        _ => vals[idx].0,
    }
}

pub(crate) fn compute_sum<T>(values: &[T]) -> T
where
    T: std::iter::Sum<T> + Copy,
{
    values.iter().copied().sum()
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
            rolling_apply_convolve(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_mean,
                weights,
            )
        }
        (false, Some(weights)) => {
            let values = as_floats(values);
            rolling_apply_convolve(
                values,
                window_size,
                min_periods,
                det_offsets,
                compute_mean,
                weights,
            )
        }
    }
}

pub fn rolling_median<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum<T> + std::cmp::PartialOrd + num::ToPrimitive,
{
    match (center, weights) {
        (true, None) => rolling_apply_quantile(
            values,
            0.5,
            QuantileInterpolOptions::Linear,
            window_size,
            min_periods,
            det_offsets_center,
            compute_quantile,
        ),
        (false, None) => rolling_apply_quantile(
            values,
            0.5,
            QuantileInterpolOptions::Linear,
            window_size,
            min_periods,
            det_offsets,
            compute_quantile,
        ),
        (true, Some(weights)) => {
            let values = as_floats(values);
            rolling_apply_pairs(
                values,
                0.5,
                QuantileInterpolOptions::Linear,
                window_size,
                min_periods,
                det_offsets_center,
                compute_weighted_quantile,
                weights,
            )
        }
        (false, Some(weights)) => {
            let values = as_floats(values);
            rolling_apply_pairs(
                values,
                0.5,
                QuantileInterpolOptions::Linear,
                window_size,
                min_periods,
                det_offsets,
                compute_weighted_quantile,
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
            rolling_apply_convolve(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_min,
                weights,
            )
        }
        (false, Some(weights)) => {
            let values = as_floats(values);
            rolling_apply_convolve(
                values,
                window_size,
                min_periods,
                det_offsets,
                compute_min,
                weights,
            )
        }
    }
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
    T: NativeType + std::iter::Sum<T> + std::cmp::PartialOrd + num::ToPrimitive,
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
        (true, Some(weights)) => {
            let values = as_floats(values);
            rolling_apply_pairs(
                values,
                quantile,
                interpolation,
                window_size,
                min_periods,
                det_offsets_center,
                compute_weighted_quantile,
                weights,
            )
        }
        (false, Some(weights)) => {
            let values = as_floats(values);
            rolling_apply_pairs(
                values,
                quantile,
                interpolation,
                window_size,
                min_periods,
                det_offsets,
                compute_weighted_quantile,
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
            rolling_apply_convolve(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_max,
                weights,
            )
        }
        (false, Some(weights)) => {
            let values = as_floats(values);
            rolling_apply_convolve(
                values,
                window_size,
                min_periods,
                det_offsets,
                compute_max,
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
    T: NativeType + Float + std::iter::Sum,
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
            rolling_apply_convolve(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_var,
                weights,
            )
        }
        (false, Some(weights)) => {
            let values = as_floats(values);
            rolling_apply_convolve(
                values,
                window_size,
                min_periods,
                det_offsets,
                compute_var,
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
            rolling_apply_convolve(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                compute_sum,
                weights,
            )
        }
        (false, Some(weights)) => {
            let values = as_floats(values);
            rolling_apply_convolve(
                values,
                window_size,
                min_periods,
                det_offsets,
                compute_sum,
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
}
