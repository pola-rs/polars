use std::num::NonZeroUsize;

use polars_arrow::array::{Array, PrimitiveArray};
use polars_arrow::types::NativeType;

use crate::ewm::EwmStateUpdate;

pub fn ewm_mean<I, T>(
    xs: I,
    alpha: T,
    adjust: bool,
    min_samples: usize,
    ignore_nulls: bool,
) -> PrimitiveArray<T>
where
    I: IntoIterator<Item = Option<T>>,
    T: num_traits::Float + NativeType + std::ops::MulAssign,
{
    let mut state: EwmMeanState<T> = EwmMeanState::new(alpha, adjust, min_samples, ignore_nulls);
    state.update_iter(xs).collect()
}

pub struct EwmMeanState<T> {
    weighted_mean: T,
    weight: T,
    alpha: T,
    non_null_count: usize,
    adjust: bool,
    min_samples: usize,
    ignore_nulls: bool,
}

/// The running state of an [`EwmMeanState`] that has seen a non-null value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EwmMeanCarry<T> {
    pub weighted_mean: T,
    pub weight: T,
    pub non_null_count: NonZeroUsize,
}

impl<T> EwmMeanState<T>
where
    T: NativeType + num_traits::Float + std::ops::MulAssign,
{
    pub fn new(alpha: T, adjust: bool, min_samples: usize, ignore_nulls: bool) -> Self {
        Self {
            weighted_mean: T::zero(),
            weight: T::zero(),
            alpha,
            non_null_count: 0,
            adjust,
            min_samples: min_samples.max(1),
            ignore_nulls,
        }
    }

    /// Continue from `carry`, the [`Self::carry`] of a state with the same options.
    pub fn with_carry(self, carry: Option<EwmMeanCarry<T>>) -> Self {
        match carry {
            Some(c) => Self {
                weighted_mean: c.weighted_mean,
                weight: c.weight,
                non_null_count: c.non_null_count.get(),
                ..self
            },
            None => Self::new(self.alpha, self.adjust, self.min_samples, self.ignore_nulls),
        }
    }

    /// The running state, `None` until the first non-null value.
    pub fn carry(&self) -> Option<EwmMeanCarry<T>> {
        let non_null_count = NonZeroUsize::new(self.non_null_count)?;
        Some(EwmMeanCarry {
            weighted_mean: self.weighted_mean,
            weight: self.weight,
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
        let new_value_weight = if self.adjust { T::one() } else { self.alpha };

        values.into_iter().map(move |opt_v| {
            if self.non_null_count == 0
                && let Some(v) = opt_v
            {
                // Initialize
                self.non_null_count = 1;
                self.weighted_mean = v;
                self.weight = T::one();
            } else {
                if opt_v.is_some() || !self.ignore_nulls {
                    self.weight *= T::one() - self.alpha;
                }

                if let Some(new_v) = opt_v {
                    let new_weight = self.weight + new_value_weight;

                    let new_value_frac = new_value_weight / new_weight;
                    let mut new_mean =
                        self.weighted_mean + (new_v - self.weighted_mean) * new_value_frac;
                    if !new_mean.is_finite() && new_v.is_finite() && self.weighted_mean.is_finite()
                    {
                        // `new_v - weighted_mean` overflowed, the mean itself is in range.
                        new_mean = self.weighted_mean * (self.weight / new_weight)
                            + new_v * new_value_frac;
                    }
                    self.weighted_mean = new_mean;

                    self.weight = if self.adjust {
                        self.weight + T::one()
                    } else {
                        T::one()
                    };

                    self.non_null_count += 1;
                }
            }

            (opt_v.is_some() && self.non_null_count >= self.min_samples)
                .then_some(self.weighted_mean)
        })
    }
}

impl<T> EwmStateUpdate for EwmMeanState<T>
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
    fn test_ewm_mean_without_null() {
        let xs: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), Some(3.0)];
        for adjust in [false, true] {
            for ignore_nulls in [false, true] {
                for min_samples in [0, 1] {
                    let result = ewm_mean(xs.clone(), ALPHA, adjust, min_samples, ignore_nulls);
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
    fn test_ewm_mean_continue_from_carry() {
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

        for adjust in [false, true] {
            for ignore_nulls in [false, true] {
                for min_samples in [1, 3] {
                    let new = || EwmMeanState::new(0.3, adjust, min_samples, ignore_nulls);
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
