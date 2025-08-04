use polars_utils::min_max::{MaxPropagateNan, MinMaxPolicy, MinPropagateNan};

use super::super::min_max::MinMaxWindow;
use super::*;

pub type MinWindow<'a, T> = MinMaxWindow<'a, T, MinPropagateNan>;
pub type MaxWindow<'a, T> = MinMaxWindow<'a, T, MaxPropagateNan>;

fn weighted_min_max<T, P>(values: &[T], weights: &[T]) -> T
where
    T: NativeType + std::ops::Mul<Output = T>,
    P: MinMaxPolicy,
{
    values
        .iter()
        .zip(weights)
        .map(|(v, w)| *v * *w)
        .reduce(P::best)
        .unwrap()
}

macro_rules! rolling_minmax_func {
    ($rolling_m:ident, $policy:ident) => {
        pub fn $rolling_m<T>(
            values: &[T],
            window_size: usize,
            min_periods: usize,
            center: bool,
            weights: Option<&[f64]>,
            _params: Option<RollingFnParams>,
        ) -> PolarsResult<ArrayRef>
        where
            T: NativeType + PartialOrd + IsFloat + Bounded + NumCast + Mul<Output = T> + Num,
        {
            let offset_fn = match center {
                true => det_offsets_center,
                false => det_offsets,
            };
            match weights {
                None => rolling_apply_agg_window::<MinMaxWindow<T, $policy>, _, _>(
                    values,
                    window_size,
                    min_periods,
                    offset_fn,
                    None,
                ),
                Some(weights) => {
                    assert!(
                        T::is_float(),
                        "implementation error, should only be reachable by float types"
                    );
                    let weights = weights
                        .iter()
                        .map(|v| NumCast::from(*v).unwrap())
                        .collect::<Vec<_>>();
                    no_nulls::rolling_apply_weights(
                        values,
                        window_size,
                        min_periods,
                        offset_fn,
                        weighted_min_max::<T, $policy>,
                        &weights,
                    )
                },
            }
        }
    };
}

rolling_minmax_func!(rolling_min, MinPropagateNan);
rolling_minmax_func!(rolling_max, MaxPropagateNan);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rolling_min_max() {
        let values = &[1.0f64, 5.0, 3.0, 4.0];

        let out = rolling_min(values, 2, 2, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(1.0), Some(3.0), Some(3.0)]);
        let out = rolling_max(values, 2, 2, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(5.0), Some(5.0), Some(4.0)]);

        let out = rolling_min(values, 2, 1, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(3.0), Some(3.0)]);
        let out = rolling_max(values, 2, 1, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(5.0), Some(5.0), Some(4.0)]);

        let out = rolling_max(values, 3, 1, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(5.0), Some(5.0), Some(5.0)]);

        // test nan handling.
        let values = &[1.0, 2.0, 3.0, f64::nan(), 5.0, 6.0, 7.0];
        let out = rolling_min(values, 3, 3, false, None, None).unwrap();
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
                    Some(1.0),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(5.0)
                ]
            )
        );

        let out = rolling_max(values, 3, 3, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(
            format!("{:?}", out.as_slice()),
            format!(
                "{:?}",
                &[
                    None,
                    None,
                    Some(3.0),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(f64::nan()),
                    Some(7.0)
                ]
            )
        );
    }
}
