use arrow::array::PrimitiveArray;
use arrow::types::NativeType;
use num::Float;

use crate::utils::CustomIterTools;

// See:
// https://github.com/pola-rs/polars/issues/2148
// https://stackoverflow.com/a/51392341/6717054

pub fn ewm_mean<T>(
    xs: &PrimitiveArray<T>,
    alpha: T,
    adjust: bool,
    min_periods: usize,
) -> PrimitiveArray<T>
where
    T: Float + NativeType,
{
    let mut denom = T::zero();
    let one_sub_alpha = T::one() - alpha;
    let mut non_null_cnt = 0usize;

    let mut opt_ewma = None;

    xs.iter()
        .map(|opt_x| {
            if let Some(&x) = opt_x {
                non_null_cnt += 1;

                let prev_ewma = opt_ewma.unwrap_or(x);

                let value = if adjust {
                    let numer = prev_ewma * denom * one_sub_alpha + x;
                    denom = T::one() + one_sub_alpha * denom;
                    numer / denom
                } else {
                    x * alpha + prev_ewma * one_sub_alpha
                };
                opt_ewma = Some(value);
            }
            match non_null_cnt < min_periods {
                true => None,
                false => opt_ewma,
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
                    Some(1.6666666666666667f32),
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
            Some(3.193548387096774f32),
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
            Some(3.6666666666666665f32),
            Some(2.2903225806451615f32),
            Some(3.1587301587301586f32),
        ]);
        assert_eq!(result, expected);
    }
}
