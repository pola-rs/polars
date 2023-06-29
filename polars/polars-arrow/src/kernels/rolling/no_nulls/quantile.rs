use std::fmt::Debug;

use num_traits::ToPrimitive;
use polars_error::polars_ensure;

use super::QuantileInterpolOptions::*;
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
        + PartialOrd
        + ToPrimitive
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
        + PartialOrd
        + ToPrimitive
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
    _params: DynArgs,
) -> PolarsResult<ArrayRef>
where
    T: NativeType
        + std::iter::Sum<T>
        + PartialOrd
        + ToPrimitive
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
) -> PolarsResult<ArrayRef>
where
    T: NativeType
        + std::iter::Sum<T>
        + PartialOrd
        + ToPrimitive
        + NumCast
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + Zero
        + IsFloat,
{
    let offset_fn = match center {
        true => det_offsets_center,
        false => det_offsets,
    };
    match weights {
        None => Ok(rolling_apply_quantile(
            values,
            quantile,
            interpolation,
            window_size,
            min_periods,
            offset_fn,
            compute_quantile2,
        )),
        Some(weights) => {
            let wsum = weights.iter().sum();
            polars_ensure!(
                wsum != 0.0,
                ComputeError: "Weighted quantile is undefined if weights sum to 0"
            );
            Ok(rolling_apply_weighted_quantile(
                values,
                quantile,
                interpolation,
                window_size,
                min_periods,
                offset_fn,
                weights,
                wsum,
            ))
        }
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

#[inline]
fn compute_wq<T>(buf: &[(T, f64)], p: f64, wsum: f64, interp: QuantileInterpolOptions) -> T
where
    T: Debug
        + NativeType
        + Mul<Output = T>
        + Sub<Output = T>
        + NumCast
        + ToPrimitive
        + Zero
        + IsFloat
        + PartialOrd,
{
    // There are a few ways to compute a weighted quantile but no "canonical" way.
    // This is mostly taken from the Julia implementation which was readable and reasonable
    // https://juliastats.org/StatsBase.jl/stable/scalarstats/#Quantile-and-Related-Functions-1
    let (mut s, mut s_old, mut vk, mut v_old) = (0.0, 0.0, T::zero(), T::zero());

    // Once the cumulative weight crosses h, we've found our ind{ex/ices}. The definition may look
    // odd but it's the equivalent of taking h = p * (n - 1) + 1 if your data is indexed from 1.
    let h: f64 = p * (wsum - buf[0].1) + buf[0].1;
    for &(v, w) in buf.iter().filter(|(_, w)| *w != 0.0) {
        vk = v; // We need the "next" value if we break.
        if s > h {
            break;
        }
        (s_old, v_old) = (s, v);
        s += w;
    }
    match (h == s_old, interp) {
        (true, _) => v_old, // If we hit the break exactly interpolation shouldn't matter
        (_, Lower) => v_old,
        (_, Higher) => vk,
        (_, Nearest) => {
            if s - h > h - s_old {
                v_old
            } else {
                vk
            }
        }
        (_, Midpoint) => (vk + v_old) * NumCast::from(0.5).unwrap(),
        // This is seemingly the canonical way to do it.
        (_, Linear) => {
            v_old + <T as NumCast>::from((h - s_old) / (s - s_old)).unwrap() * (vk - v_old)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn rolling_apply_weighted_quantile<T, Fo>(
    values: &[T],
    p: f64,
    interpolation: QuantileInterpolOptions,
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    weights: &[f64],
    wsum: f64,
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
    T: Debug
        + NativeType
        + Mul<Output = T>
        + Sub<Output = T>
        + NumCast
        + ToPrimitive
        + Zero
        + IsFloat
        + PartialOrd,
{
    assert_eq!(weights.len(), window_size);
    let mut buf = vec![(T::zero(), 0.0); window_size];
    let len = values.len();
    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            let vals = unsafe { values.get_unchecked(start..end) };

            // Sorting is not ideal, see https://github.com/tobiasschoch/wquantile for something faster
            buf.iter_mut()
                .zip(vals.iter().zip(weights))
                .for_each(|(b, (v, w))| *b = (*v, *w));
            buf.sort_unstable_by(|&a, &b| compare_fn_nan_max(&a.0, &b.0));
            compute_wq(&buf, p, wsum, interpolation)
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
        )
        .unwrap();
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
        )
        .unwrap();
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
        )
        .unwrap();
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
        )
        .unwrap();
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
        )
        .unwrap();
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
            let out1 = rolling_min(values, 2, 2, false, None, None).unwrap();
            let out1 = out1.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out1 = out1.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            let out2 = rolling_quantile(values, 0.0, interpol, 2, 2, false, None).unwrap();
            let out2 = out2.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out2 = out2.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            assert_eq!(out1, out2);

            let out1 = rolling_max(values, 2, 2, false, None, None).unwrap();
            let out1 = out1.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out1 = out1.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            let out2 = rolling_quantile(values, 1.0, interpol, 2, 2, false, None).unwrap();
            let out2 = out2.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out2 = out2.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            assert_eq!(out1, out2);
        }
    }
}
