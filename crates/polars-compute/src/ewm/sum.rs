use std::num::NonZeroUsize;

use polars_arrow::array::{Array, PrimitiveArray};
use polars_arrow::types::NativeType;

use crate::ewm::EwmStateUpdate;

pub fn ewm_sum<I, T>(xs: I, alpha: T, min_samples: usize, ignore_nulls: bool) -> PrimitiveArray<T>
where
    I: IntoIterator<Item = Option<T>>,
    T: num_traits::Float + NativeType + std::ops::MulAssign,
{
    let mut state: EwmSumState<T> = EwmSumState::new(alpha, min_samples, ignore_nulls);
    state.update_iter(xs).collect()
}

pub struct EwmSumState<T> {
    sum: T,
    alpha: T,
    non_null_count: usize,
    min_samples: usize,
    ignore_nulls: bool,
}

/// The running state of an [`EwmSumState`] that has seen a non-null value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EwmSumCarry<T> {
    pub sum: T,
    pub non_null_count: NonZeroUsize,
}

impl<T> EwmSumState<T>
where
    T: NativeType + num_traits::Float + std::ops::MulAssign,
{
    pub fn new(alpha: T, min_samples: usize, ignore_nulls: bool) -> Self {
        Self {
            sum: T::zero(),
            alpha,
            non_null_count: 0,
            min_samples: min_samples.max(1),
            ignore_nulls,
        }
    }

    /// Continue from `carry`, the [`Self::carry`] of a state with the same options.
    pub fn with_carry(self, carry: Option<EwmSumCarry<T>>) -> Self {
        match carry {
            Some(c) => Self {
                sum: c.sum,
                non_null_count: c.non_null_count.get(),
                ..self
            },
            None => Self::new(self.alpha, self.min_samples, self.ignore_nulls),
        }
    }

    /// The running state, `None` until the first non-null value.
    pub fn carry(&self) -> Option<EwmSumCarry<T>> {
        let non_null_count = NonZeroUsize::new(self.non_null_count)?;
        Some(EwmSumCarry {
            sum: self.sum,
            non_null_count,
        })
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

            (opt_v.is_some() && self.non_null_count >= self.min_samples).then_some(self.sum)
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

        for ignore_nulls in [false, true] {
            for min_samples in [0, 1] {
                let result = ewm_sum(xs.clone(), ALPHA, min_samples, ignore_nulls);
                let expected =
                    PrimitiveArray::from(vec![Some(1.0f64), Some(2.5f64), Some(4.25f64)]);
                assert_allclose!(result, expected, EPS);
            }

            let result = ewm_sum(xs.clone(), ALPHA, 2, ignore_nulls);
            let expected = PrimitiveArray::from(vec![None, Some(2.5f64), Some(4.25f64)]);
            assert_allclose!(result, expected, EPS);
        }
    }

    #[test]
    fn test_ewm_sum_continue_from_carry() {
        let xs = [
            None,
            Some(5.0f64),
            None,
            Some(7.0),
            Some(0.1),
            None,
            None,
            Some(-2.0),
            Some(1.0),
        ];
        let bits =
            |out: &[Option<f64>]| out.iter().map(|x| x.map(f64::to_bits)).collect::<Vec<_>>();

        for ignore_nulls in [false, true] {
            for min_samples in [1, 3] {
                let new = || EwmSumState::new(0.3, min_samples, ignore_nulls);
                let whole: Vec<_> = new().update_iter(xs).collect();

                for split in 0..=xs.len() {
                    let (head, tail) = xs.split_at(split);
                    let mut state = new();
                    let mut out: Vec<_> = state.update_iter(head.iter().copied()).collect();
                    let mut state = new().with_carry(state.carry());
                    out.extend(state.update_iter(tail.iter().copied()));
                    assert_eq!(bits(&out), bits(&whole));
                }
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

        assert_allclose!(
            ewm_sum(xs1.clone(), ALPHA, 0, true),
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
            ewm_sum(xs1.clone(), ALPHA, 0, false),
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
