use super::*;
use crate::utils::CustomIterTools;
use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::buffer::MutableBuffer;
use arrow::datatypes::DataType;
use arrow::types::NativeType;
use num::Float;
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

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
        .collect_trusted::<MutableBuffer<f64>>();

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
        .collect_trusted::<MutableBuffer<K>>();

    let validity = create_validity(min_periods, len as usize, window_size, det_offsets_fn);
    Arc::new(PrimitiveArray::from_data(
        K::DATA_TYPE,
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
