use std::ops::{AddAssign, MulAssign};

use arrow::array::PrimitiveArray;
use arrow::types::NativeType;
use num::Float;

use crate::trusted_len::TrustedLen;
use crate::utils::CustomIterTools;

pub fn ewm_mean<I, T>(
    xs: I,
    alpha: T,
    adjust: bool,
    min_periods: usize,
    ignore_nulls: bool,
) -> PrimitiveArray<T>
where
    I: IntoIterator<Item = Option<T>>,
    I::IntoIter: TrustedLen,
    T: Float + NativeType + AddAssign + MulAssign,
{
    let new_wt = if adjust { T::one() } else { alpha };
    let old_wt_factor = T::one() - alpha;
    let mut old_wt = T::one();
    let mut weighted_avg = None;
    let mut non_null_cnt = 0usize;

    xs.into_iter()
        .enumerate()
        .map(|(i, opt_x)| {
            if opt_x.is_some() {
                non_null_cnt += 1;
            }
            match (i, weighted_avg) {
                (0, _) | (_, None) => weighted_avg = opt_x,
                (_, Some(w_avg)) => {
                    if opt_x.is_some() || !ignore_nulls {
                        old_wt *= old_wt_factor;
                        if let Some(x) = opt_x {
                            if w_avg != x {
                                weighted_avg =
                                    Some((old_wt * w_avg + new_wt * x) / (old_wt + new_wt));
                            }
                            old_wt = if adjust { old_wt + new_wt } else { T::one() };
                        }
                    }
                }
            }
            match non_null_cnt < min_periods {
                true => None,
                false => weighted_avg,
            }
        })
        .collect_trusted()
}

#[cfg(test)]
mod test {
    use super::super::assert_allclose;
    use super::*;
    const ALPHA: f64 = 0.5;
    const EPS: f64 = 1e-15;

    #[test]
    fn test_ewm_mean_without_null() {
        let xs: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), Some(3.0)];
        for adjust in [false, true] {
            for ignore_nulls in [false, true] {
                for min_periods in [0, 1] {
                    let result = ewm_mean(xs.clone(), ALPHA, adjust, min_periods, ignore_nulls);
                    let expected = match adjust {
                        false => PrimitiveArray::from([Some(1.0f64), Some(1.5f64), Some(2.25f64)]),
                        true => PrimitiveArray::from([
                            Some(1.00000000000000000000),
                            Some(1.66666666666666674068),
                            Some(2.42857142857142838110),
                        ]),
                    };
                    assert_allclose!(result, expected, 1e-15);
                }
                let result = ewm_mean(xs.clone(), ALPHA, adjust, 2, ignore_nulls);
                let expected = match adjust {
                    false => PrimitiveArray::from([None, Some(1.5f64), Some(2.25f64)]),
                    true => PrimitiveArray::from([
                        None,
                        Some(1.66666666666666674068),
                        Some(2.42857142857142838110),
                    ]),
                };
                assert_allclose!(result, expected, EPS);
            }
        }
    }

    #[test]
    fn test_ewm_mean_with_null() {
        let xs1 = vec![
            None,
            None,
            Some(5.0f64),
            Some(7.0f64),
            None,
            Some(2.0f64),
            Some(1.0f64),
            Some(4.0f64),
        ];
        assert_allclose!(
            ewm_mean(xs1.clone(), 0.5, true, 0, true),
            PrimitiveArray::from([
                None,
                None,
                Some(5.00000000000000000000),
                Some(6.33333333333333303727),
                Some(6.33333333333333303727),
                Some(3.85714285714285720630),
                Some(2.33333333333333348136),
                Some(3.19354838709677402164),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_mean(xs1.clone(), 0.5, true, 0, false),
            PrimitiveArray::from([
                None,
                None,
                Some(5.00000000000000000000),
                Some(6.33333333333333303727),
                Some(6.33333333333333303727),
                Some(3.18181818181818165669),
                Some(1.88888888888888883955),
                Some(3.03389830508474567239),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_mean(xs1.clone(), 0.5, false, 0, true),
            PrimitiveArray::from([
                None,
                None,
                Some(5.00000000000000000000),
                Some(6.00000000000000000000),
                Some(6.00000000000000000000),
                Some(4.00000000000000000000),
                Some(2.50000000000000000000),
                Some(3.25000000000000000000),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_mean(xs1.clone(), 0.5, false, 0, false),
            PrimitiveArray::from([
                None,
                None,
                Some(5.00000000000000000000),
                Some(6.00000000000000000000),
                Some(6.00000000000000000000),
                Some(3.33333333333333348136),
                Some(2.16666666666666696273),
                Some(3.08333333333333348136),
            ]),
            EPS
        );
    }
}
