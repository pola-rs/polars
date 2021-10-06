use crate::prelude::*;

#[cfg_attr(docsrs, doc(cfg(feature = "moment")))]
fn moment_precomputed_mean(s: &Series, moment: usize, mean: f64) -> Result<Option<f64>> {
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
                a_zero_mean.clone()
            } else {
                &a_zero_mean * &a_zero_mean
            };

            for n in n_list.iter().rev() {
                s = &s * &s;
                if n % 2 == 1 {
                    s = &s * &a_zero_mean;
                }
            }
            s.mean()
        }
    };
    Ok(out)
}

impl Series {
    /// Compute the sample skewness of a data set.
    ///
    /// For normally distributed data, the skewness should be about zero. For
    /// unimodal continuous distributions, a skewness value greater than zero means
    /// that there is more weight in the right tail of the distribution. The
    /// function `skewtest` can be used to determine if the skewness value
    /// is close enough to zero, statistically speaking.
    ///
    /// see: https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/stats.py#L1024
    #[cfg_attr(docsrs, doc(cfg(feature = "moment")))]
    pub fn skew(&self, bias: bool) -> Result<Option<f64>> {
        let mean = match self.mean() {
            Some(mean) => mean,
            None => return Ok(None),
        };
        // we can unwrap because if it were None, we already return None above
        let m2 = moment_precomputed_mean(self, 2, mean)?.unwrap();
        let m3 = moment_precomputed_mean(self, 3, mean)?.unwrap();

        let out = m3 / m2.powf(1.5);

        if !bias {
            let n = self.len() as f64;
            Ok(Some(((n - 1.0) * n).sqrt() / (n - 2.0) * m3 / m2.powf(1.5)))
        } else {
            Ok(Some(out))
        }
    }

    /// Compute the kurtosis (Fisher or Pearson) of a dataset.
    ///
    /// Kurtosis is the fourth central moment divided by the square of the
    /// variance. If Fisher's definition is used, then 3.0 is subtracted from
    /// the result to give 0.0 for a normal distribution.
    /// If bias is `false` then the kurtosis is calculated using k statistics to
    /// eliminate bias coming from biased moment estimators
    ///
    /// see: https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/stats.py#L1027
    #[cfg_attr(docsrs, doc(cfg(feature = "moment")))]
    pub fn kurtosis(&self, fisher: bool, bias: bool) -> Result<Option<f64>> {
        let mean = match self.mean() {
            Some(mean) => mean,
            None => return Ok(None),
        };
        // we can unwrap because if it were None, we already return None above
        let m2 = moment_precomputed_mean(self, 2, mean)?.unwrap();
        let m4 = moment_precomputed_mean(self, 4, mean)?.unwrap();

        let out = if !bias {
            let n = self.len() as f64;
            1.0 / (n - 2.0) / (n - 3.0)
                * ((n.powf(2.0) - 1.0) * m4 / m2.powf(2.0) - 3.0 * (n - 1.0).powf(2.0))
        } else {
            m4 / m2.powf(2.0)
        };
        if fisher {
            Ok(Some(out - 3.0))
        } else {
            Ok(Some(out))
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    impl Series {
        fn moment(&self, moment: usize) -> Result<Option<f64>> {
            match self.mean() {
                Some(mean) => moment_precomputed_mean(self, moment, mean),
                None => Ok(None),
            }
        }
    }

    #[test]
    fn test_moment_compute() -> Result<()> {
        let s = Series::new("", &[1, 2, 3, 4, 5, 23]);

        assert_eq!(s.moment(0)?, Some(1.0));
        assert_eq!(s.moment(1)?, Some(0.0));
        assert!((s.moment(2)?.unwrap() - 57.22222222222223).abs() < 0.00001);
        assert!((s.moment(3)?.unwrap() - 724.0740740740742).abs() < 0.00001);

        Ok(())
    }

    #[test]
    fn test_skew() -> Result<()> {
        let s = Series::new("", &[1, 2, 3, 4, 5, 23]);
        assert!(s.skew(false)?.unwrap() - 2.2905330058490514 < 0.0001);
        assert!(s.skew(true)?.unwrap() - 2.2905330058490514 < 0.0001);
        Ok(())
    }

    #[test]
    fn test_kurtosis() -> Result<()> {
        let s = Series::new("", &[1, 2, 3, 4, 5, 23]);
        assert!(s.kurtosis(true, false)?.unwrap() - 5.400820058440946 < 0.0001);
        assert!(s.kurtosis(true, true)?.unwrap() - 0.9945668771797536 < 0.0001);
        Ok(())
    }
}
