#![allow(unsafe_op_in_unsafe_fn)]

use super::super::mean::MeanWindow;
use super::*;

pub fn rolling_mean<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    _params: Option<RollingFnParams>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + Float + std::iter::Sum<T> + SubAssign + AddAssign + IsFloat,
{
    let offset_fn = match center {
        true => det_offsets_center,
        false => det_offsets,
    };
    match weights {
        None => rolling_apply_agg_window::<MeanWindow<_>, _, _, _>(
            values,
            window_size,
            min_periods,
            offset_fn,
            None,
        ),
        Some(weights) => {
            let wts = no_nulls::coerce_weights(weights);
            no_nulls::rolling_apply_weights(
                values,
                window_size,
                min_periods,
                offset_fn,
                no_nulls::compute_mean_weights,
                &wts,
                center,
            )
        },
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rolling_mean_window_size_zero() {
        let values = &[1.0f64, 2.0, 3.0, 4.0];

        // window_size=0: mean of empty = None
        let out = rolling_mean(values, 0, 0, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, None]);

        // center=true should behave the same
        let out = rolling_mean(values, 0, 0, true, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, None]);
    }
}
