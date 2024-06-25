use polars_core::prelude::*;

use crate::prelude::SeriesSealed;

fn moment_precomputed_mean(s: &Series, moment: usize, mean: f64) -> PolarsResult<Option<f64>> {
    // see: https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/stats.py#L922
    let out = match moment {
        0 => Some(1.0),
        1 => Some(0.0),
        _ => {
            let mut n_list = vec![moment];
            let mut current_n = moment;
            while current_n > 2 {
                if current_n % 2 == 1 {
                    current_n = (current_n - 1) / 2
                } else {
                    current_n /= 2
                }
                n_list.push(current_n)
            }

            let a_zero_mean = s.cast(&DataType::Float64)? - mean;

            let mut s = if n_list.pop().unwrap() == 1 {
                // TODO remove: false positive
                #[allow(clippy::redundant_clone)]
                a_zero_mean.clone()
            } else {
                (&a_zero_mean * &a_zero_mean)?
            };

            for n in n_list.iter().rev() {
                s = (&s * &s)?;
                if n % 2 == 1 {
                    s = (&s * &a_zero_mean)?;
                }
            }
            s.mean()
        },
    };
    Ok(out)
}

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

        let mean = match s.mean() {
            Some(mean) => mean,
            None => return Ok(None),
        };
        // we can unwrap because if it were None, we already return None above
        let m2 = moment_precomputed_mean(s, 2, mean)?.unwrap();
        let m3 = moment_precomputed_mean(s, 3, mean)?.unwrap();
        let zero = m2 <= (f64::EPSILON * mean).powf(2.0);
        let vals = match zero {
            true => f64::NAN,
            false => m3 / m2.powf(1.5),
        };
        let n = (s.len() - s.null_count()) as f64;
        let out = if !bias && !zero && n > 3.0 {
            ((n - 1.0) * n).sqrt() / (n - 2.0) * vals
        } else {
            vals
        };
        Ok(Some(out))
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

        let mean = match s.mean() {
            Some(mean) => mean,
            None => return Ok(None),
        };
        // we can unwrap because if it were None, we already return None above
        let m2 = moment_precomputed_mean(s, 2, mean)?.unwrap();
        let m4 = moment_precomputed_mean(s, 4, mean)?.unwrap();
        let zero = m2 <= (f64::EPSILON * mean).powf(2.0);
        let vals = match zero {
            true => f64::NAN,
            false => m4 / m2.powf(2.0),
        };
        let n = (s.len() - s.null_count()) as f64;
        let out = if !bias && !zero && n > 3.0 {
            3.0 + 1.0 / (n - 2.0) / (n - 3.0)
                * ((n.powf(2.0) - 1.0) * vals - 3.0 * (n - 1.0).powf(2.0))
        } else {
            vals
        };
        if fisher {
            Ok(Some(out - 3.0))
        } else {
            Ok(Some(out))
        }
    }
}

impl MomentSeries for Series {}

#[cfg(test)]
mod test {
    use super::*;

    fn moment(s: &Series, moment: usize) -> PolarsResult<Option<f64>> {
        match s.mean() {
            Some(mean) => moment_precomputed_mean(s, moment, mean),
            None => Ok(None),
        }
    }

    #[test]
    fn test_moment_compute() -> PolarsResult<()> {
        let s = Series::new("", &[1, 2, 3, 4, 5, 23]);

        assert_eq!(moment(&s, 0)?, Some(1.0));
        assert_eq!(moment(&s, 1)?, Some(0.0));
        assert!((moment(&s, 2)?.unwrap() - 57.22222222222223).abs() < 0.00001);
        assert!((moment(&s, 3)?.unwrap() - 724.0740740740742).abs() < 0.00001);

        Ok(())
    }

    #[test]
    fn test_skew() -> PolarsResult<()> {
        let s = Series::new("", &[1, 2, 3, 4, 5, 23]);
        let s2 = Series::new("", &[Some(1), Some(2), Some(3), None, Some(1)]);

        assert!((s.skew(false)?.unwrap() - 2.2905330058490514).abs() < 0.0001);
        assert!((s.skew(true)?.unwrap() - 1.6727687946848508).abs() < 0.0001);

        assert!((s2.skew(false)?.unwrap() - 0.8545630383279711).abs() < 0.0001);
        assert!((s2.skew(true)?.unwrap() - 0.49338220021815865).abs() < 0.0001);

        Ok(())
    }

    #[test]
    fn test_kurtosis() -> PolarsResult<()> {
        let s = Series::new("", &[1, 2, 3, 4, 5, 23]);

        assert!((s.kurtosis(true, true)?.unwrap() - 0.9945668771797536).abs() < 0.0001);
        assert!((s.kurtosis(true, false)?.unwrap() - 5.400820058440946).abs() < 0.0001);
        assert!((s.kurtosis(false, true)?.unwrap() - 3.994566877179754).abs() < 0.0001);
        assert!((s.kurtosis(false, false)?.unwrap() - 8.400820058440946).abs() < 0.0001);

        let s2 = Series::new(
            "",
            &[Some(1), Some(2), Some(3), None, Some(1), Some(2), Some(3)],
        );
        assert!((s2.kurtosis(true, true)?.unwrap() - (-1.5)).abs() < 0.0001);
        assert!((s2.kurtosis(true, false)?.unwrap() - (-1.875)).abs() < 0.0001);
        assert!((s2.kurtosis(false, true)?.unwrap() - 1.5).abs() < 0.0001);
        assert!((s2.kurtosis(false, false)?.unwrap() - 1.125).abs() < 0.0001);

        Ok(())
    }
}
