use std::fmt::Debug;

use num_traits::ToPrimitive;
use polars_error::polars_ensure;

use super::QuantileInterpolOptions::*;
use super::*;

pub struct QuantileWindow<'a, T: NativeType + IsFloat + PartialOrd> {
    sorted: SortedBuf<'a, T>,
    prob: f64,
    interpol: QuantileInterpolOptions,
}

impl<
        'a,
        T: NativeType
            + IsFloat
            + Float
            + std::iter::Sum
            + AddAssign
            + SubAssign
            + Div<Output = T>
            + NumCast
            + One
            + Zero
            + PartialOrd
            + Sub<Output = T>,
    > RollingAggWindowNoNulls<'a, T> for QuantileWindow<'a, T>
{
    fn new(slice: &'a [T], start: usize, end: usize, params: DynArgs) -> Self {
        let params = params.unwrap();
        let params = params.downcast_ref::<RollingQuantileParams>().unwrap();
        Self {
            sorted: SortedBuf::new(slice, start, end),
            prob: params.prob,
            interpol: params.interpol,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> T {
        let vals = self.sorted.update(start, end);
        let length = vals.len();

        let mut idx = match self.interpol {
            QuantileInterpolOptions::Nearest => ((length as f64) * self.prob) as usize,
            QuantileInterpolOptions::Lower
            | QuantileInterpolOptions::Midpoint
            | QuantileInterpolOptions::Linear => {
                ((length as f64 - 1.0) * self.prob).floor() as usize
            },
            QuantileInterpolOptions::Higher => ((length as f64 - 1.0) * self.prob).ceil() as usize,
        };

        idx = std::cmp::min(idx, length - 1);

        match self.interpol {
            QuantileInterpolOptions::Midpoint => {
                let top_idx = ((length as f64 - 1.0) * self.prob).ceil() as usize;
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
            },
            QuantileInterpolOptions::Linear => {
                let float_idx = (length as f64 - 1.0) * self.prob;
                let top_idx = f64::ceil(float_idx) as usize;

                if top_idx == idx {
                    // safety
                    // we are in bounds
                    unsafe { *vals.get_unchecked(idx) }
                } else {
                    let proportion = T::from(float_idx - idx as f64).unwrap();
                    proportion * (vals[top_idx] - vals[idx]) + vals[idx]
                }
            },
            _ => {
                // safety
                // we are in bounds
                unsafe { *vals.get_unchecked(idx) }
            },
        }
    }
}

pub fn rolling_quantile<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    params: DynArgs,
) -> PolarsResult<ArrayRef>
where
    T: NativeType
        + IsFloat
        + Float
        + std::iter::Sum
        + AddAssign
        + SubAssign
        + Div<Output = T>
        + NumCast
        + One
        + Zero
        + PartialOrd
        + Sub<Output = T>,
{
    let offset_fn = match center {
        true => det_offsets_center,
        false => det_offsets,
    };
    match weights {
        None => rolling_apply_agg_window::<QuantileWindow<_>, _, _>(
            values,
            window_size,
            min_periods,
            offset_fn,
            params,
        ),
        Some(weights) => {
            let wsum = weights.iter().sum();
            polars_ensure!(
                wsum != 0.0,
                ComputeError: "Weighted quantile is undefined if weights sum to 0"
            );
            let params = params.unwrap();
            let params = params.downcast_ref::<RollingQuantileParams>().unwrap();
            Ok(rolling_apply_weighted_quantile(
                values,
                params.prob,
                params.interpol,
                window_size,
                min_periods,
                offset_fn,
                weights,
                wsum,
            ))
        },
    }
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
    for &(v, w) in buf.iter() {
        if s > h {
            break;
        }
        (s_old, v_old, vk) = (s, vk, v);
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
        },
        (_, Midpoint) => (vk + v_old) * NumCast::from(0.5).unwrap(),
        // This is seemingly the canonical way to do it.
        (_, Linear) => {
            v_old + <T as NumCast>::from((h - s_old) / (s - s_old)).unwrap() * (vk - v_old)
        },
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
    // Keep nonzero weights and their indices to know which values we need each iteration.
    let nz_idx_wts: Vec<_> = weights.iter().enumerate().filter(|x| x.1 != &0.0).collect();
    let mut buf = vec![(T::zero(), 0.0); nz_idx_wts.len()];
    let len = values.len();
    let out = (0..len)
        .map(|idx| {
            // Don't need end. Window size is constant and we computed offsets from start above.
            let (start, _) = det_offsets_fn(idx, window_size, len);

            // Sorting is not ideal, see https://github.com/tobiasschoch/wquantile for something faster
            unsafe {
                buf.iter_mut()
                    .zip(nz_idx_wts.iter())
                    .for_each(|(b, (i, w))| *b = (*values.get_unchecked(i + start), **w));
            }
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
    use crate::legacy::kernels::rolling::no_nulls::{rolling_max, rolling_min};

    #[test]
    fn test_rolling_median() {
        let values = &[1.0, 2.0, 3.0, 4.0];
        let med_pars = Some(Arc::new(RollingQuantileParams {
            prob: 0.5,
            interpol: Linear,
        }) as Arc<dyn Any + Send + Sync>);
        let out = rolling_quantile(values, 2, 2, false, None, med_pars.clone()).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(1.5), Some(2.5), Some(3.5)]);

        let out = rolling_quantile(values, 2, 1, false, None, med_pars.clone()).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.5), Some(2.5), Some(3.5)]);

        let out = rolling_quantile(values, 4, 1, false, None, med_pars.clone()).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.5), Some(2.0), Some(2.5)]);

        let out = rolling_quantile(values, 4, 1, true, None, med_pars.clone()).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.5), Some(2.0), Some(2.5), Some(3.0)]);

        let out = rolling_quantile(values, 4, 4, true, None, med_pars.clone()).unwrap();
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
            let min_pars = Some(Arc::new(RollingQuantileParams {
                prob: 0.0,
                interpol,
            }) as Arc<dyn Any + Send + Sync>);
            let out1 = rolling_min(values, 2, 2, false, None, None).unwrap();
            let out1 = out1.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out1 = out1.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            let out2 = rolling_quantile(values, 2, 2, false, None, min_pars).unwrap();
            let out2 = out2.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out2 = out2.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            assert_eq!(out1, out2);

            let max_pars = Some(Arc::new(RollingQuantileParams {
                prob: 1.0,
                interpol,
            }) as Arc<dyn Any + Send + Sync>);
            let out1 = rolling_max(values, 2, 2, false, None, None).unwrap();
            let out1 = out1.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out1 = out1.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            let out2 = rolling_quantile(values, 2, 2, false, None, max_pars).unwrap();
            let out2 = out2.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out2 = out2.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            assert_eq!(out1, out2);
        }
    }
}
