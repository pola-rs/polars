use std::fmt::Debug;

use super::*;
use crate::index::IdxSize;
use crate::trusted_len::TrustedLen;

// used by agg_quantile
pub fn rolling_quantile_by_iter<T, O>(
    values: &[T],
    quantile: f64,
    interpolation: QuantileInterpolOptions,
    offsets: O,
) -> ArrayRef
where
    O: Iterator<Item = (IdxSize, IdxSize)> + TrustedLen,
    T: std::iter::Sum<T>
        + NativeType
        + Copy
        + std::cmp::PartialOrd
        + num::ToPrimitive
        + NumCast
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + IsFloat,
{
    if values.is_empty() {
        let out: Vec<T> = vec![];
        return Box::new(PrimitiveArray::new(T::PRIMITIVE.into(), out.into(), None));
    }

    let mut sorted_window = SortedBuf::new(values, 0, 1);

    let out = offsets
        .map(|(start, len)| {
            let end = start + len;

            // safety:
            // we are in bounds
            if start == end {
                None
            } else {
                let window = unsafe { sorted_window.update(start as usize, end as usize) };
                Some(compute_quantile2(window, quantile, interpolation))
            }
        })
        .collect::<PrimitiveArray<T>>();

    Box::new(out)
}

pub(crate) fn compute_quantile2<T>(
    vals: &[T],
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
        + Mul<Output = T>
        + IsFloat,
{
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
                // safety
                // we are in bounds
                unsafe { *vals.get_unchecked(idx) }
            } else {
                // safety
                // we are in bounds
                let (mid, mid_plus_1) =
                    unsafe { (*vals.get_unchecked(idx), *vals.get_unchecked(idx + 1)) };

                (mid + mid_plus_1) / T::from::<f64>(2.0f64).unwrap()
            }
        }
        QuantileInterpolOptions::Linear => {
            let float_idx = (length as f64 - 1.0) * quantile;
            let top_idx = f64::ceil(float_idx) as usize;

            if top_idx == idx {
                // safety
                // we are in bounds
                unsafe { *vals.get_unchecked(idx) }
            } else {
                let proportion = T::from(float_idx - idx as f64).unwrap();
                proportion * (vals[top_idx] - vals[idx]) + vals[idx]
            }
        }
        _ => {
            // safety
            // we are in bounds
            unsafe { *vals.get_unchecked(idx) }
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
    T: NativeType
        + std::iter::Sum<T>
        + std::cmp::PartialOrd
        + num::ToPrimitive
        + NumCast
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + Zero
        + IsFloat,
{
    rolling_quantile(
        values,
        0.5,
        QuantileInterpolOptions::Linear,
        window_size,
        min_periods,
        center,
        weights,
    )
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
        + Zero
        + IsFloat,
{
    match (center, weights) {
        (true, None) => rolling_apply_quantile(
            values,
            quantile,
            interpolation,
            window_size,
            min_periods,
            det_offsets_center,
            compute_quantile2,
        ),
        (false, None) => rolling_apply_quantile(
            values,
            quantile,
            interpolation,
            window_size,
            min_periods,
            det_offsets,
            compute_quantile2,
        ),
        (true, Some(weights)) => rolling_apply_convolve_quantile(
            values,
            quantile,
            interpolation,
            window_size,
            min_periods,
            det_offsets_center,
            compute_quantile2,
            weights,
        ),
        (false, Some(weights)) => rolling_apply_convolve_quantile(
            values,
            quantile,
            interpolation,
            window_size,
            min_periods,
            det_offsets,
            compute_quantile2,
            weights,
        ),
    }
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
    T: Debug + NativeType + IsFloat + PartialOrd,
{
    let len = values.len();
    let (start, end) = det_offsets_fn(0, window_size, len);
    let mut sorted_window = SortedBuf::new(values, start, end);

    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);

            // Safety:
            // we are in bounds
            let window = unsafe { sorted_window.update(start, end) };
            aggregator(window, quantile, interpolation)
        })
        .collect_trusted::<Vec<T>>();

    let validity = create_validity(min_periods, len, window_size, det_offsets_fn);
    Box::new(PrimitiveArray::new(
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
    T: Debug + NativeType + Mul<Output = T> + NumCast + ToPrimitive + Zero + IsFloat + PartialOrd,
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

            sort_buf(&mut buf);
            aggregator(&buf, quantile, interpolation)
        })
        .collect_trusted::<Vec<T>>();

    let validity = create_validity(min_periods, len, window_size, det_offsets_fn);
    Box::new(PrimitiveArray::new(
        T::PRIMITIVE.into(),
        out.into(),
        validity.map(|b| b.into()),
    ))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::kernels::rolling::no_nulls::{rolling_max, rolling_min};

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
        let values = &[1.0f64, 2.0, 3.0, 4.0];

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
