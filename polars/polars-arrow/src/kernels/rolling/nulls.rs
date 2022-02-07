use super::*;
use crate::prelude::QuantileInterpolOptions;
use crate::utils::CustomIterTools;
use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::bitmap::utils::{count_zeros, get_bit_unchecked};
use arrow::types::NativeType;
use num::{Float, NumCast, One, Zero};
use std::ops::AddAssign;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

fn rolling_apply<T, K, Fo, Fa>(
    values: &[T],
    bitmap: &Bitmap,
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    aggregator: Fa,
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End) + Copy,
    // &[T] -> values of array
    // &[u8] -> validity bytes
    // usize -> offset in validity bytes array
    // usize -> min_periods
    Fa: Fn(&[T], &[u8], usize, usize) -> Option<K>,
    K: NativeType + Default,
{
    let len = values.len();
    let (validity_bytes, offset, _) = bitmap.as_slice();

    let mut validity = match create_validity(min_periods, len as usize, window_size, det_offsets_fn)
    {
        Some(v) => v,
        None => {
            let mut validity = MutableBitmap::with_capacity(len);
            validity.extend_constant(len, true);
            validity
        }
    };

    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            let vals = unsafe { values.get_unchecked(start..end) };
            match aggregator(vals, validity_bytes, offset + start, min_periods) {
                Some(val) => val,
                None => {
                    validity.set(idx, false);
                    K::default()
                }
            }
        })
        .collect_trusted::<Vec<K>>();

    Arc::new(PrimitiveArray::from_data(
        K::PRIMITIVE.into(),
        out.into(),
        Some(validity.into()),
    ))
}

#[allow(clippy::too_many_arguments)]
fn rolling_apply_quantile<T, Fo, Fa>(
    values: &[T],
    bitmap: &Bitmap,
    quantile: f64,
    interpolation: QuantileInterpolOptions,
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    aggregator: Fa,
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End) + Copy,
    // &[T] -> values of array
    // &[u8] -> validity bytes
    // QuantileInterpolOptions -> Interpolation option
    // usize -> offset in validity bytes array
    // usize -> min_periods
    Fa: Fn(&[T], &[u8], f64, QuantileInterpolOptions, usize, usize) -> Option<T>,
    T: Default + NativeType,
{
    let len = values.len();
    let (validity_bytes, offset, _) = bitmap.as_slice();

    let mut validity = match create_validity(min_periods, len as usize, window_size, det_offsets_fn)
    {
        Some(v) => v,
        None => {
            let mut validity = MutableBitmap::with_capacity(len);
            validity.extend_constant(len, true);
            validity
        }
    };

    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            let vals = unsafe { values.get_unchecked(start..end) };
            match aggregator(
                vals,
                validity_bytes,
                quantile,
                interpolation,
                offset + start,
                min_periods,
            ) {
                Some(val) => val,
                None => {
                    validity.set(idx, false);
                    T::default()
                }
            }
        })
        .collect_trusted::<Vec<T>>();

    Arc::new(PrimitiveArray::from_data(
        T::PRIMITIVE.into(),
        out.into(),
        Some(validity.into()),
    ))
}

fn compute_sum<T>(
    values: &[T],
    validity_bytes: &[u8],
    offset: usize,
    min_periods: usize,
) -> Option<T>
where
    T: NativeType + std::iter::Sum<T> + Zero + AddAssign,
{
    let null_count = count_zeros(validity_bytes, offset, values.len());
    if null_count == 0 {
        Some(no_nulls::compute_sum(values))
    } else if (values.len() - null_count) < min_periods {
        None
    } else {
        let mut out = Zero::zero();
        for (i, val) in values.iter().enumerate() {
            // Safety:
            // in bounds
            if unsafe { get_bit_unchecked(validity_bytes, offset + i) } {
                out += *val;
            }
        }
        Some(out)
    }
}

fn compute_mean<T>(
    values: &[T],
    validity_bytes: &[u8],
    offset: usize,
    min_periods: usize,
) -> Option<T>
where
    T: NativeType + std::iter::Sum<T> + Zero + AddAssign + Float,
{
    let null_count = count_zeros(validity_bytes, offset, values.len());
    if null_count == 0 {
        Some(no_nulls::compute_mean(values))
    } else if (values.len() - null_count) < min_periods {
        None
    } else {
        let mut out = T::zero();
        let mut count = T::zero();
        for (i, val) in values.iter().enumerate() {
            // Safety:
            // in bounds
            if unsafe { get_bit_unchecked(validity_bytes, offset + i) } {
                out += *val;
                count += One::one()
            }
        }
        Some(out / count)
    }
}

pub(crate) fn compute_var<T>(
    values: &[T],
    validity_bytes: &[u8],
    offset: usize,
    min_periods: usize,
) -> Option<T>
where
    T: NativeType + std::iter::Sum<T> + Zero + AddAssign + Float,
{
    let null_count = count_zeros(validity_bytes, offset, values.len());
    if null_count == 0 {
        Some(no_nulls::compute_var(values))
    } else if (values.len() - null_count) < min_periods {
        None
    } else {
        match compute_mean(values, validity_bytes, offset, min_periods) {
            None => None,
            Some(mean) => {
                let mut sum = T::zero();
                let mut count = T::zero();
                for (i, val) in values.iter().enumerate() {
                    // Safety:
                    // in bounds
                    if unsafe { get_bit_unchecked(validity_bytes, offset + i) } {
                        let v = *val - mean;
                        sum += v * v;
                        count += One::one()
                    }
                }
                Some(sum / (count - T::one()))
            }
        }
    }
}

fn compute_quantile<T>(
    values: &[T],
    validity_bytes: &[u8],
    quantile: f64,
    interpolation: QuantileInterpolOptions,
    offset: usize,
    min_periods: usize,
) -> Option<T>
where
    T: NativeType
        + std::iter::Sum<T>
        + Zero
        + AddAssign
        + std::cmp::PartialOrd
        + num::ToPrimitive
        + NumCast
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Mul<Output = T>,
{
    if !(0.0..=1.0).contains(&quantile) {
        panic!("quantile should be between 0.0 and 1.0");
    }

    let null_count = count_zeros(validity_bytes, offset, values.len());
    if null_count == 0 {
        return Some(no_nulls::compute_quantile(values, quantile, interpolation));
    } else if (values.len() - null_count) < min_periods {
        return None;
    }

    let mut vals: Vec<T> = Vec::new();
    for (i, val) in values.iter().enumerate() {
        // Safety:
        // in bounds
        if unsafe { get_bit_unchecked(validity_bytes, offset + i) } {
            vals.push(NumCast::from(*val).unwrap());
        }
    }
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
            Some((vals[idx] + vals[top_idx]) / T::from::<f64>(2.0f64).unwrap())
        }
        QuantileInterpolOptions::Linear => {
            let float_idx = (length as f64 - 1.0) * quantile;
            let top_idx = f64::ceil(float_idx) as usize;

            if top_idx == idx {
                Some(vals[idx])
            } else {
                let proportion = T::from(float_idx - idx as f64).unwrap();
                Some(proportion * (vals[top_idx] - vals[idx]) + vals[idx])
            }
        }
        _ => Some(vals[idx]),
    }
}

fn compute_min<T>(
    values: &[T],
    validity_bytes: &[u8],
    offset: usize,
    min_periods: usize,
) -> Option<T>
where
    T: NativeType + PartialOrd,
{
    let null_count = count_zeros(validity_bytes, offset, values.len());
    if null_count == 0 {
        Some(no_nulls::compute_min(values))
    } else if (values.len() - null_count) < min_periods {
        None
    } else {
        let mut out = None;
        for (i, val) in values.iter().enumerate() {
            // Safety:
            // in bounds
            if unsafe { get_bit_unchecked(validity_bytes, offset + i) } {
                match out {
                    None => {
                        out = Some(*val);
                    }
                    Some(a) => {
                        if *val < a {
                            out = Some(*val)
                        }
                    }
                }
            }
        }
        out
    }
}

fn compute_max<T>(
    values: &[T],
    validity_bytes: &[u8],
    offset: usize,
    min_periods: usize,
) -> Option<T>
where
    T: NativeType + PartialOrd,
{
    let null_count = count_zeros(validity_bytes, offset, values.len());
    if null_count == 0 {
        Some(no_nulls::compute_max(values))
    } else if (values.len() - null_count) < min_periods {
        None
    } else {
        let mut out = None;
        for (i, val) in values.iter().enumerate() {
            // Safety:
            // in bounds
            if unsafe { get_bit_unchecked(validity_bytes, offset + i) } {
                match out {
                    None => {
                        out = Some(*val);
                    }
                    Some(a) => {
                        if *val > a {
                            out = Some(*val)
                        }
                    }
                }
            }
        }
        out
    }
}

pub fn rolling_var<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum<T> + Zero + AddAssign + Float,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
            compute_var,
        )
    } else {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            compute_var,
        )
    }
}

pub fn rolling_sum<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum + Zero + AddAssign + Copy,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
            compute_sum,
        )
    } else {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            compute_sum,
        )
    }
}

pub fn rolling_quantile<T>(
    arr: &PrimitiveArray<T>,
    quantile: f64,
    interpolation: QuantileInterpolOptions,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType
        + std::iter::Sum
        + Zero
        + AddAssign
        + Copy
        + std::cmp::PartialOrd
        + num::ToPrimitive
        + NumCast
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Mul<Output = T>,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply_quantile(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            quantile,
            interpolation,
            window_size,
            min_periods,
            det_offsets_center,
            compute_quantile,
        )
    } else {
        rolling_apply_quantile(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            quantile,
            interpolation,
            window_size,
            min_periods,
            det_offsets,
            compute_quantile,
        )
    }
}

pub fn rolling_mean<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum + Zero + AddAssign + Copy + Float,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
            compute_mean,
        )
    } else {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            compute_mean,
        )
    }
}

pub fn rolling_min<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum + Zero + AddAssign + Copy + PartialOrd,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
            compute_min,
        )
    } else {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            compute_min,
        )
    }
}

pub fn rolling_max<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum + Zero + AddAssign + Copy + PartialOrd,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
            compute_max,
        )
    } else {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            compute_max,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::buffer::Buffer;
    use arrow::datatypes::DataType;

    #[test]
    fn test_rolling_sum_nulls() {
        let buf = Buffer::from(vec![1.0, 2.0, 3.0, 4.0]);
        let arr = &PrimitiveArray::from_data(
            DataType::Float64,
            buf,
            Some(Bitmap::from(&[true, false, true, true])),
        );

        let out = rolling_sum(arr, 2, 2, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, Some(7.0)]);

        let out = rolling_sum(arr, 2, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(3.0), Some(7.0)]);

        let out = rolling_sum(arr, 4, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(4.0), Some(8.0)]);

        let out = rolling_sum(arr, 4, 1, true, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(4.0), Some(8.0), Some(7.0)]);

        let out = rolling_sum(arr, 4, 4, true, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, None]);
    }

    #[test]
    fn test_rolling_median_nulls() {
        let buf = Buffer::from(vec![1.0, 2.0, 3.0, 4.0]);
        let arr = &PrimitiveArray::from_data(
            DataType::Float64,
            buf,
            Some(Bitmap::from(&[true, false, true, true])),
        );

        let out = rolling_quantile(arr, 0.5, QuantileInterpolOptions::Linear, 2, 2, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, Some(3.5)]);

        let out = rolling_quantile(arr, 0.5, QuantileInterpolOptions::Linear, 2, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(3.0), Some(3.5)]);

        let out = rolling_quantile(arr, 0.5, QuantileInterpolOptions::Linear, 4, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(2.0), Some(3.0)]);

        let out = rolling_quantile(arr, 0.5, QuantileInterpolOptions::Linear, 4, 1, true, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(2.0), Some(3.0), Some(3.5)]);

        let out = rolling_quantile(arr, 0.5, QuantileInterpolOptions::Linear, 4, 4, true, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, None]);
    }

    #[test]
    fn test_rolling_max_no_nulls() {
        let buf = Buffer::from(vec![1.0, 2.0, 3.0, 4.0]);
        let arr = &PrimitiveArray::from_data(
            DataType::Float64,
            buf,
            Some(Bitmap::from(&[true, true, true, true])),
        );
        let out = rolling_max(arr, 4, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);

        let out = rolling_max(arr, 2, 2, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(2.0), Some(3.0), Some(4.0)]);

        let out = rolling_max(arr, 4, 4, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, Some(4.0)])
    }

    #[test]
    fn test_rolling_quantile_nulls_limits() {
        // compare quantiles to corresponding min/max/median values
        let buf = Buffer::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let values = &PrimitiveArray::from_data(
            DataType::Float64,
            buf,
            Some(Bitmap::from(&[true, false, false, true, true])),
        );

        let interpol_options = vec![
            QuantileInterpolOptions::Lower,
            QuantileInterpolOptions::Higher,
            QuantileInterpolOptions::Nearest,
            QuantileInterpolOptions::Midpoint,
            QuantileInterpolOptions::Linear,
        ];

        for interpol in interpol_options {
            let out1 = rolling_min(values, 2, 1, false, None);
            let out1 = out1.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out1 = out1.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            let out2 = rolling_quantile(values, 0.0, interpol, 2, 1, false, None);
            let out2 = out2.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out2 = out2.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            assert_eq!(out1, out2);

            let out1 = rolling_max(values, 2, 1, false, None);
            let out1 = out1.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out1 = out1.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            let out2 = rolling_quantile(values, 1.0, interpol, 2, 1, false, None);
            let out2 = out2.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out2 = out2.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            assert_eq!(out1, out2);
        }
    }
}
