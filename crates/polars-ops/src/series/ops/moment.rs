use polars_compute::moment::{KurtosisState, SkewState, kurtosis, skew};
use polars_core::prelude::*;

use crate::prelude::SeriesSealed;

pub trait MomentSeries: SeriesSealed {
    /// Compute the sample skewness of a data set.
    ///
    /// For normally distributed data, the skewness should be about zero. For
    /// uni-modal continuous distributions, a skewness value greater than zero means
    /// that there is more weight in the right tail of the distribution. The
    /// function `skewtest` can be used to determine if the skewness value
    /// is close enough to zero, statistically speaking.
    ///
    /// see: [scipy](https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/stats.py#L1024)
    fn skew(&self, bias: bool) -> PolarsResult<Option<f64>> {
        let s = self.as_series();
        let s = s.cast(&DataType::Float64)?;
        let ca = s.f64().unwrap();

        let mut state = SkewState::default();
        for arr in ca.downcast_iter() {
            state.combine(&skew(arr));
        }
        Ok(state.finalize(bias))
    }

    /// Compute the kurtosis (Fisher or Pearson) of a dataset.
    ///
    /// Kurtosis is the fourth central moment divided by the square of the
    /// variance. If Fisher's definition is used, then 3.0 is subtracted from
    /// the result to give 0.0 for a normal distribution.
    /// If bias is `false` then the kurtosis is calculated using k statistics to
    /// eliminate bias coming from biased moment estimators
    ///
    /// see: [scipy](https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/stats.py#L1027)
    fn kurtosis(&self, fisher: bool, bias: bool) -> PolarsResult<Option<f64>> {
        let s = self.as_series();
        let s = s.cast(&DataType::Float64)?;
        let ca = s.f64().unwrap();

        let mut state = KurtosisState::default();
        for arr in ca.downcast_iter() {
            state.combine(&kurtosis(arr));
        }
        Ok(state.finalize(fisher, bias))
    }
}

impl MomentSeries for Series {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_skew() -> PolarsResult<()> {
        let s = Series::new(PlSmallStr::EMPTY, &[1, 2, 3, 4, 5, 23]);
        let s2 = Series::new(
            PlSmallStr::EMPTY,
            &[Some(1), Some(2), Some(3), None, Some(1)],
        );

        assert!((s.skew(false)?.unwrap() - 2.2905330058490514).abs() < 0.0001);
        assert!((s.skew(true)?.unwrap() - 1.6727687946848508).abs() < 0.0001);

        assert!((s2.skew(false)?.unwrap() - 0.8545630383279711).abs() < 0.0001);
        assert!((s2.skew(true)?.unwrap() - 0.49338220021815865).abs() < 0.0001);

        Ok(())
    }

    #[test]
    fn test_kurtosis() -> PolarsResult<()> {
        let s = Series::new(PlSmallStr::EMPTY, &[1, 2, 3, 4, 5, 23]);

        assert!((s.kurtosis(true, true)?.unwrap() - 0.9945668771797536).abs() < 0.0001);
        assert!((s.kurtosis(true, false)?.unwrap() - 5.400820058440946).abs() < 0.0001);
        assert!((s.kurtosis(false, true)?.unwrap() - 3.994566877179754).abs() < 0.0001);
        assert!((s.kurtosis(false, false)?.unwrap() - 8.400820058440946).abs() < 0.0001);

        let s2 = Series::new(
            PlSmallStr::EMPTY,
            &[Some(1), Some(2), Some(3), None, Some(1), Some(2), Some(3)],
        );
        assert!((s2.kurtosis(true, true)?.unwrap() - (-1.5)).abs() < 0.0001);
        assert!((s2.kurtosis(true, false)?.unwrap() - (-1.875)).abs() < 0.0001);
        assert!((s2.kurtosis(false, true)?.unwrap() - 1.5).abs() < 0.0001);
        assert!((s2.kurtosis(false, false)?.unwrap() - 1.125).abs() < 0.0001);

        Ok(())
    }
}
