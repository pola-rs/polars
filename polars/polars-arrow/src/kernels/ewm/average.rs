use std::ops::AddAssign;
use arrow::array::PrimitiveArray;
use arrow::types::NativeType;
use num::Float;

use crate::utils::CustomIterTools;


pub fn ewm_mean<T>(
    xs: &PrimitiveArray<T>,
    alpha: T,
    adjust: bool,
    min_periods: usize,
) -> PrimitiveArray<T>
where
    T: Float + NativeType + AddAssign,
{
    let one_sub_alpha = T::one() - alpha;

    let mut opt_mean = None;
    let mut non_null_cnt = 0usize;

    let wgt = if adjust { T::one() } else { alpha };
    let mut wgt_sum = if adjust { T::zero() } else { T::one() };

    xs.iter()
        .map(|opt_x| {
            if let Some(&x) = opt_x {
                non_null_cnt += 1;

                let prev_mean = opt_mean.unwrap_or(x);

                // TODO: this is `ignore_na = True`
                wgt_sum = one_sub_alpha * wgt_sum + wgt;

                opt_mean = Some(prev_mean + (x - prev_mean) * wgt / wgt_sum);
            }
            match non_null_cnt < min_periods {
                true => None,
                false => opt_mean,
            }
        })
        .collect_trusted()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ewm_mean_without_null() {
        let xs = PrimitiveArray::from([Some(1.0f32), Some(2.0f32), Some(3.0f32)]);

        for adjust in [false, true] {
            let result = ewm_mean(&xs, 0.5, adjust, 0);

            let expected = match adjust {
                false => PrimitiveArray::from([Some(1.0f32), Some(1.5f32), Some(2.25f32)]),
                true => PrimitiveArray::from([
                    Some(1.0f32),
                    Some(1.6666667f32), // <-- pandas gives 1.6666666666666667
                    Some(2.4285714285714284f32),
                ]),
            };
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_ewm_mean_with_null() {
        let xs = PrimitiveArray::from([Some(1.0f32), None, Some(1.0f32), Some(1.0f32)]);
        let result = ewm_mean(&xs, 0.5, false, 2);
        let expected = PrimitiveArray::from([None, None, Some(1.0f32), Some(1.0f32)]);
        assert_eq!(result, expected);

        let xs = PrimitiveArray::from([None, None, Some(1.0f32), Some(1.0f32)]);
        let result = ewm_mean(&xs, 0.5, false, 1);
        let expected = PrimitiveArray::from([None, None, Some(1.0f32), Some(1.0f32)]);
        assert_eq!(result, expected);

        let xs = PrimitiveArray::from([
            Some(2.0f32),
            Some(3.0f32),
            Some(5.0f32),
            Some(7.0f32),
            None,
            None,
            None,
            Some(4.0f32),
        ]);
        let result = ewm_mean(&xs, 0.5, false, 0);
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

        let xs = PrimitiveArray::from([
            None,
            None,
            Some(5.0f32),
            Some(7.0f32),
            None,
            Some(2.0f32),
            Some(1.0f32),
            Some(4.0f32),
        ]);
        let unadjusted_result = ewm_mean(&xs, 0.5, false, 1);
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
        let adjusted_result = ewm_mean(&xs, 0.5, true, 1);
        let adjusted_expected = PrimitiveArray::from([
            None,
            None,
            Some(5.0f32),
            Some(6.333333333333333f32),
            Some(6.333333333333333f32),
            Some(3.857142857142857f32),
            Some(2.3333333333333335f32),
            Some(3.1935482f32), // <-- pandas gives 3.19354838...
        ]);
        assert_eq!(adjusted_result, adjusted_expected);

        let xs = PrimitiveArray::from([
            None,
            Some(1.0f32),
            Some(5.0f32),
            Some(7.0f32),
            None,
            Some(2.0f32),
            Some(1.0f32),
            Some(4.0f32),
        ]);
        let result = ewm_mean(&xs, 0.5, true, 1);
        let expected = PrimitiveArray::from([
            None,
            Some(1.0f32),
            Some(3.6666666666666665f32),
            Some(5.571428571428571f32),
            Some(5.571428571428571f32),
            Some(3.6666665), // <-- pandas gives 3.6666666666666665
            Some(2.2903225806451615f32),
            Some(3.1587301587301586f32),
        ]);
        assert_eq!(result, expected);
    }
}
