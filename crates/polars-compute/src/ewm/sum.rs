use arrow::array::{Array, PrimitiveArray};
use arrow::types::NativeType;

use crate::ewm::EwmStateUpdate;

pub fn ewm_sum<I, T>(
    xs: I,
    alpha: T,
    adjust: bool,
    min_periods: usize,
    ignore_nulls: bool,
) -> PrimitiveArray<T>
where
    I: IntoIterator<Item = Option<T>>,
    T: num_traits::Float + NativeType + std::ops::MulAssign,
{
    let mut state: EwmSumState<T> = EwmSumState::new(alpha, adjust, min_periods, ignore_nulls);
    state.update_iter(xs).collect()
}

pub struct EwmSumState<T> {
    sum: T,
    alpha: T,
    non_null_count: usize,
    min_periods: usize,
    ignore_nulls: bool,
}

impl<T> EwmSumState<T>
where
    T: NativeType + num_traits::Float + std::ops::MulAssign,
{
    pub fn new(alpha: T, _adjust: bool, min_periods: usize, ignore_nulls: bool) -> Self {
        Self {
            sum: T::zero(),
            alpha,
            non_null_count: 0,
            min_periods: min_periods.max(1),
            ignore_nulls,
        }
    }

    pub fn update(&mut self, values: &PrimitiveArray<T>) -> PrimitiveArray<T> {
        self.update_iter(values.iter().map(|x| x.copied()))
            .collect()
    }

    pub fn update_iter<I>(&mut self, values: I) -> impl Iterator<Item = Option<T>>
    where
        I: IntoIterator<Item = Option<T>>,
    {
        values.into_iter().map(move |opt_v| {
            if let Some(v) = opt_v {
                if self.non_null_count == 0 {
                    self.sum = v;
                } else {
                    self.sum *= T::one() - self.alpha;
                    self.sum = self.sum + v;
                }
                self.non_null_count += 1;
            } else if self.non_null_count > 0 && !self.ignore_nulls {
                self.sum *= T::one() - self.alpha;
            }

            (opt_v.is_some() && self.non_null_count >= self.min_periods).then_some(self.sum)
        })
    }
}

impl<T> EwmStateUpdate for EwmSumState<T>
where
    T: NativeType + num_traits::Float + std::ops::MulAssign,
{
    fn ewm_state_update(&mut self, values: &dyn Array) -> Box<dyn Array> {
        let values: &PrimitiveArray<T> = values.as_any().downcast_ref().unwrap();
        let out: PrimitiveArray<T> = self.update(values);
        out.boxed()
    }
}

#[cfg(test)]
mod test {
    use super::super::assert_allclose;
    use super::*;

    const ALPHA: f64 = 0.5;
    const EPS: f64 = 1e-15;

    #[test]
    fn test_ewm_sum_without_null() {
        let xs: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), Some(3.0)];

        for adjust in [false, true] {
            for ignore_nulls in [false, true] {
                for min_periods in [0, 1] {
                    let result = ewm_sum(xs.clone(), ALPHA, adjust, min_periods, ignore_nulls);
                    let expected =
                        PrimitiveArray::from(vec![Some(1.0f64), Some(2.5f64), Some(4.25f64)]);
                    assert_allclose!(result, expected, EPS);
                }

                let result = ewm_sum(xs.clone(), ALPHA, adjust, 2, ignore_nulls);
                let expected = PrimitiveArray::from(vec![None, Some(2.5f64), Some(4.25f64)]);
                assert_allclose!(result, expected, EPS);
            }
        }
    }

    #[test]
    fn test_ewm_sum_with_null() {
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

        for adjust in [false, true] {
            assert_allclose!(
                ewm_sum(xs1.clone(), ALPHA, adjust, 0, true),
                PrimitiveArray::from(vec![
                    None,
                    None,
                    Some(5.0),
                    Some(9.5),
                    None,
                    Some(6.75),
                    Some(4.375),
                    Some(6.1875),
                ]),
                EPS
            );

            assert_allclose!(
                ewm_sum(xs1.clone(), ALPHA, adjust, 0, false),
                PrimitiveArray::from(vec![
                    None,
                    None,
                    Some(5.0),
                    Some(9.5),
                    None,
                    Some(4.375),
                    Some(3.1875),
                    Some(5.59375),
                ]),
                EPS
            );
        }
    }
}
