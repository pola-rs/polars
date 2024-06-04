use std::ops::{AddAssign, MulAssign};

use num_traits::Float;

use crate::array::PrimitiveArray;
use crate::legacy::utils::CustomIterTools;
use crate::trusted_len::TrustedLen;
use crate::types::NativeType;

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
                },
            }
            match (non_null_cnt < min_periods, opt_x.is_some()) {
                (_, false) => None,
                (true, true) => None,
                (false, true) => weighted_avg,
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
                            Some(1.0),
                            Some(1.666_666_666_666_666_7),
                            Some(2.428_571_428_571_428_4),
                        ]),
                    };
                    assert_allclose!(result, expected, 1e-15);
                }
                let result = ewm_mean(xs.clone(), ALPHA, adjust, 2, ignore_nulls);
                let expected = match adjust {
                    false => PrimitiveArray::from([None, Some(1.5f64), Some(2.25f64)]),
                    true => PrimitiveArray::from([
                        None,
                        Some(1.666_666_666_666_666_7),
                        Some(2.428_571_428_571_428_4),
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
                Some(5.0),
                Some(6.333_333_333_333_333),
                None,
                Some(3.857_142_857_142_857),
                Some(2.333_333_333_333_333_5),
                Some(3.193_548_387_096_774),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_mean(xs1.clone(), 0.5, true, 0, false),
            PrimitiveArray::from([
                None,
                None,
                Some(5.0),
                Some(6.333_333_333_333_333),
                None,
                Some(3.181_818_181_818_181_7),
                Some(1.888_888_888_888_888_8),
                Some(3.033_898_305_084_745_7),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_mean(xs1.clone(), 0.5, false, 0, true),
            PrimitiveArray::from([
                None,
                None,
                Some(5.0),
                Some(6.0),
                None,
                Some(4.0),
                Some(2.5),
                Some(3.25),
            ]),
            EPS
        );
        assert_allclose!(
            ewm_mean(xs1, 0.5, false, 0, false),
            PrimitiveArray::from([
                None,
                None,
                Some(5.0),
                Some(6.0),
                None,
                Some(3.333_333_333_333_333_5),
                Some(2.166_666_666_666_667),
                Some(3.083_333_333_333_333_5),
            ]),
            EPS
        );
    }
}
