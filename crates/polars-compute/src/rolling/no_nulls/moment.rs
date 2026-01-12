#![allow(unsafe_op_in_unsafe_fn)]
use num_traits::{FromPrimitive, ToPrimitive};
use polars_error::polars_ensure;

pub use super::super::moment::*;
use super::*;

pub fn rolling_var<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    params: Option<RollingFnParams>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + Float + IsFloat + ToPrimitive + FromPrimitive + AddAssign,
{
    let offset_fn = match center {
        true => det_offsets_center,
        false => det_offsets,
    };
    match weights {
        None => rolling_apply_agg_window::<MomentWindow<_, VarianceMoment>, _, _, _>(
            values,
            window_size,
            min_periods,
            offset_fn,
            params,
        ),
        Some(weights) => {
            // Validate and standardize the weights like we do for the mean. This definition is fine
            // because frequency weights and unbiasing don't make sense for rolling operations.
            let mut wts = no_nulls::coerce_weights(weights);
            let wsum = wts.iter().fold(T::zero(), |acc, x| acc + *x);
            polars_ensure!(
                wsum != T::zero(),
                ComputeError: "Weighted variance is undefined if weights sum to 0"
            );
            wts.iter_mut().for_each(|w| *w = *w / wsum);
            super::rolling_apply_weights(
                values,
                window_size,
                min_periods,
                offset_fn,
                compute_var_weights,
                &wts,
                center,
            )
        },
    }
}

pub fn rolling_skew<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    params: Option<RollingFnParams>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + Float + IsFloat + ToPrimitive + FromPrimitive + AddAssign,
{
    let offset_fn = match center {
        true => det_offsets_center,
        false => det_offsets,
    };
    rolling_apply_agg_window::<MomentWindow<_, SkewMoment>, _, _, _>(
        values,
        window_size,
        min_periods,
        offset_fn,
        params,
    )
}

pub fn rolling_kurtosis<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    params: Option<RollingFnParams>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + Float + IsFloat + ToPrimitive + FromPrimitive + AddAssign,
{
    let offset_fn = match center {
        true => det_offsets_center,
        false => det_offsets,
    };
    rolling_apply_agg_window::<MomentWindow<_, KurtosisMoment>, _, _, _>(
        values,
        window_size,
        min_periods,
        offset_fn,
        params,
    )
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rolling_var() {
        let values = &[1.0f64, 5.0, 3.0, 4.0];

        let out = rolling_var(values, 2, 2, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(8.0), Some(2.0), Some(0.5)]);

        let testpars = Some(RollingFnParams::Var(RollingVarParams { ddof: 0 }));
        let out = rolling_var(values, 2, 2, false, None, testpars).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(4.0), Some(1.0), Some(0.25)]);

        let out = rolling_var(values, 2, 1, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        // we cannot compare nans, so we compare the string values
        assert_eq!(
            format!("{:?}", out.as_slice()),
            format!("{:?}", &[None, Some(8.0), Some(2.0), Some(0.5)])
        );
        // test nan handling.
        let values = &[-10.0, 2.0, 3.0, f64::nan(), 5.0, 6.0, 7.0];
        let out = rolling_var(values, 3, 3, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        // we cannot compare nans, so we compare the string values
        assert_eq!(
            format!("{:?}", out.as_slice()),
            format!(
                "{:?}",
                &[
                    None,
                    None,
                    Some(52.33333333333333),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(0.9999999999999911)
                ]
            )
        );
    }
}
