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

            let a_zero_mean = s.cast::<Float64Type>()? - mean;

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
    #[cfg_attr(docsrs, doc(cfg(feature = "moment")))]
    pub fn skew(&self, bias: bool) -> Result<Option<f64>> {
        // see: https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/stats.py#L1024
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
}
