use std::ops::AddAssign;

use arrow::array::PrimitiveArray;
use arrow::types::NativeType;
use num::Float;

use crate::trusted_len::TrustedLen;
use crate::utils::CustomIterTools;

pub fn ewm_mean<I, T>(xs: I, alpha: T, adjust: bool, min_periods: usize) -> PrimitiveArray<T>
where
    I: IntoIterator<Item = Option<T>>,
    I::IntoIter: TrustedLen,
    T: Float + NativeType + AddAssign,
{
    if alpha.is_one() {
        return ewm_mean_alpha_equals_one(xs, min_periods);
    }

    let one_sub_alpha = T::one() - alpha;

    let mut opt_mean = None;
    let mut non_null_cnt = 0usize;

    let wgt = alpha;
    let mut wgt_sum = if adjust { T::zero() } else { T::one() };

    xs.into_iter()
        .map(|opt_x| {
            if let Some(x) = opt_x {
                non_null_cnt += 1;

                let prev_mean = opt_mean.unwrap_or(x);

                wgt_sum = one_sub_alpha * wgt_sum + wgt;

                let curr_mean = prev_mean + (x - prev_mean) * wgt / wgt_sum;

                opt_mean = Some(curr_mean);
            }
            match non_null_cnt < min_periods {
                true => None,
                false => opt_mean,
            }
        })
        .collect_trusted()
}

/// To prevent numerical instability (and as a slight optimization), we
/// special-case ``alpha=1``.
fn ewm_mean_alpha_equals_one<I, T>(xs: I, min_periods: usize) -> PrimitiveArray<T>
where
    I: IntoIterator<Item = Option<T>>,
    I::IntoIter: TrustedLen,
    T: Float + NativeType + AddAssign,
{
    let mut non_null_count = 0usize;
    xs.into_iter()
        .map(|opt_x| {
            if opt_x.is_some() {
                non_null_count += 1;
            }
            match non_null_count < min_periods {
                true => None,
                false => opt_x,
            }
        })
        .collect_trusted()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ewm_mean_without_null() {
        let xs = vec![Some(1.0f32), Some(2.0f32), Some(3.0f32)];

        for adjust in [false, true] {
            let result = ewm_mean(xs.clone().into_iter(), 0.5, adjust, 0);

            let expected = match adjust {
                false => PrimitiveArray::from([Some(1.0f32), Some(1.5f32), Some(2.25f32)]),
                true => PrimitiveArray::from([
                    Some(1.0f32),
                    Some(1.6666667f32), // <-- pandas: 1.66666667
                    Some(2.42857143),
                ]),
            };
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_ewm_mean_with_null() {
        let xs = vec![Some(1.0f32), None, Some(1.0f32), Some(1.0f32)].into_iter();
        let result = ewm_mean(xs, 0.5, false, 2);
        let expected = PrimitiveArray::from([None, None, Some(1.0f32), Some(1.0f32)]);
        assert_eq!(result, expected);

        let xs = vec![None, None, Some(1.0f32), Some(1.0f32)].into_iter();
        let result = ewm_mean(xs, 0.5, false, 1);
        let expected = PrimitiveArray::from([None, None, Some(1.0f32), Some(1.0f32)]);
        assert_eq!(result, expected);

        let xs = vec![
            Some(2.0f32),
            Some(3.0f32),
            Some(5.0f32),
            Some(7.0f32),
            None,
            None,
            None,
            Some(4.0f32),
        ];
        let result = ewm_mean(xs, 0.5, false, 0);
        let expected = PrimitiveArray::from([
            Some(2.0f32),
            Some(2.5f32),
            Some(3.75f32),
            Some(5.375f32),
            Some(5.375f32),
            Some(5.375f32),
            Some(5.375f32),
            Some(4.6875f32),
        ]);
        assert_eq!(result, expected);

        let xs = vec![
            None,
            None,
            Some(5.0f32),
            Some(7.0f32),
            None,
            Some(2.0f32),
            Some(1.0f32),
            Some(4.0f32),
        ];
        let unadjusted_result = ewm_mean(xs.clone().into_iter(), 0.5, false, 1);
        let unadjusted_expected = PrimitiveArray::from([
            None,
            None,
            Some(5.0f32),
            Some(6.0f32),
            Some(6.0f32),
            Some(4.0f32),
            Some(2.5f32),
            Some(3.25f32),
        ]);
        assert_eq!(unadjusted_result, unadjusted_expected);
        let adjusted_result = ewm_mean(xs.clone().into_iter(), 0.5, true, 1);
        let adjusted_expected = PrimitiveArray::from([
            None,
            None,
            Some(5.0f32),
            Some(6.33333333f32),
            Some(6.33333333f32),
            Some(3.85714286f32),
            Some(2.3333335f32), // <-- pandas: 2.33333333
            Some(3.19354839f32),
        ]);
        assert_eq!(adjusted_result, adjusted_expected);

        let xs = vec![
            None,
            Some(1.0f32),
            Some(5.0f32),
            Some(7.0f32),
            None,
            Some(2.0f32),
            Some(1.0f32),
            Some(4.0f32),
        ]
        .into_iter();
        let result = ewm_mean(xs, 0.5, true, 1);
        let expected = PrimitiveArray::from([
            None,
            Some(1.0f32),
            Some(3.66666667f32),
            Some(5.57142857f32),
            Some(5.57142857f32),
            Some(3.66666667),
            Some(2.2903228f32), // <-- pandas: 2.29032258
            Some(3.15873016f32),
        ]);
        assert_eq!(result, expected);
    }
}
