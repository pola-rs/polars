#![allow(unsafe_op_in_unsafe_fn)]
use super::super::sum::SumWindow;
use super::*;

pub fn rolling_sum<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    _params: Option<RollingFnParams>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType
        + std::iter::Sum
        + NumCast
        + Mul<Output = T>
        + AddAssign
        + SubAssign
        + IsFloat
        + Num
        + PartialOrd,
{
    match (center, weights) {
        (true, None) => rolling_apply_agg_window::<SumWindow<T, T>, _, _, _>(
            values,
            window_size,
            min_periods,
            det_offsets_center,
            None,
        ),
        (false, None) => rolling_apply_agg_window::<SumWindow<T, T>, _, _, _>(
            values,
            window_size,
            min_periods,
            det_offsets,
            None,
        ),
        (true, Some(weights)) => {
            let weights = no_nulls::coerce_weights(weights);
            no_nulls::rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets_center,
                no_nulls::compute_sum_weights,
                &weights,
                center,
            )
        },
        (false, Some(weights)) => {
            let weights = no_nulls::coerce_weights(weights);
            no_nulls::rolling_apply_weights(
                values,
                window_size,
                min_periods,
                det_offsets,
                no_nulls::compute_sum_weights,
                &weights,
                center,
            )
        },
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_rolling_sum() {
        let values = &[1.0f64, 2.0, 3.0, 4.0];

        let out = rolling_sum(values, 2, 2, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(3.0), Some(5.0), Some(7.0)]);

        let out = rolling_sum(values, 2, 1, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(3.0), Some(5.0), Some(7.0)]);

        let out = rolling_sum(values, 4, 1, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(3.0), Some(6.0), Some(10.0)]);

        let out = rolling_sum(values, 4, 1, true, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(3.0), Some(6.0), Some(10.0), Some(9.0)]);

        let out = rolling_sum(values, 4, 4, true, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, Some(10.0), None]);

        // test nan handling.
        let values = &[1.0, 2.0, 3.0, f64::nan(), 5.0, 6.0, 7.0];
        let out = rolling_sum(values, 3, 3, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();

        assert_eq!(
            format!("{:?}", out.as_slice()),
            format!(
                "{:?}",
                &[
                    None,
                    None,
                    Some(6.0),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(18.0)
                ]
            )
        );
    }
}
