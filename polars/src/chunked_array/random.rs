use crate::prelude::*;
use num::{Float, NumCast};
use rand::distributions::Bernoulli;
use rand::prelude::*;
use rand_distr::{Distribution, Normal, StandardNormal, Uniform};
use rayon::prelude::*;

impl<T> ChunkedArray<T>
where
    ChunkedArray<T>: ChunkTake,
{
    /// Sample n datapoints from this ChunkedArray.
    pub fn sample_n(&self, n: usize) -> Result<Self> {
        if n > self.len() {
            return Err(PolarsError::ShapeMisMatch(
                "n is larger than the number of elements in this array".into(),
            ));
        }
        let len = self.len();
        let iter = (0..n).map(|_| Uniform::new(0, len).sample(&mut rand::thread_rng()));
        Ok(self.take(iter, Some(n)))
    }

    /// Sample a fraction between 0.0-1.0 of this ChunkedArray.
    pub fn sample_frac(&self, frac: f64) -> Result<Self> {
        let n = (self.len() as f64 * frac) as usize;
        self.sample_n(n)
    }
}

impl Series {
    /// Sample n datapoints from this Series.
    pub fn sample_n(&self, n: usize) -> Result<Self> {
        Ok(apply_method_all_series_and_return!(self, sample_n, [n],?))
    }

    /// Sample a fraction between 0.0-1.0 of this Series.
    pub fn sample_frac(&self, frac: f64) -> Result<Self> {
        let n = (self.len() as f64 * frac) as usize;
        self.sample_n(n)
    }
}

impl DataFrame {
    /// Sample n datapoints from this DataFrame.
    pub fn sample_n(&self, n: usize) -> Result<Self> {
        let columns = self
            .columns
            .par_iter()
            .map(|s| s.sample_n(n))
            .collect::<Result<_>>()?;
        Ok(DataFrame::new_no_checks(columns))
    }

    /// Sample a fraction between 0.0-1.0 of this DataFrame.
    pub fn sample_frac(&self, frac: f64) -> Result<Self> {
        let n = (self.height() as f64 * frac) as usize;
        self.sample_n(n)
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Float + NumCast,
{
    /// Create `ChunkedArray` with samples from a Normal distribution.
    pub fn rand_normal(name: &str, length: usize, mean: f64, std_dev: f64) -> Result<Self> {
        let normal = match Normal::new(mean, std_dev) {
            Ok(dist) => dist,
            Err(e) => return Err(PolarsError::RandError(format!("{:?}", e))),
        };
        let mut builder = PrimitiveChunkedBuilder::<T>::new(name, length);
        for _ in 0..length {
            let smpl = normal.sample(&mut rand::thread_rng());
            let smpl = NumCast::from(smpl).unwrap();
            builder.append_value(smpl)
        }
        Ok(builder.finish())
    }

    /// Create `ChunkedArray` with samples from a Standard Normal distribution.
    pub fn rand_standard_normal(name: &str, length: usize) -> Self {
        let mut builder = PrimitiveChunkedBuilder::<T>::new(name, length);
        for _ in 0..length {
            let smpl: f64 = thread_rng().sample(StandardNormal);
            let smpl = NumCast::from(smpl).unwrap();
            builder.append_value(smpl)
        }
        builder.finish()
    }

    /// Create `ChunkedArray` with samples from a Uniform distribution.
    pub fn rand_uniform(name: &str, length: usize, low: f64, high: f64) -> Self {
        let uniform = Uniform::new(low, high);
        let mut builder = PrimitiveChunkedBuilder::<T>::new(name, length);
        for _ in 0..length {
            let smpl = uniform.sample(&mut rand::thread_rng());
            let smpl = NumCast::from(smpl).unwrap();
            builder.append_value(smpl)
        }
        builder.finish()
    }
}

impl BooleanChunked {
    /// Create `ChunkedArray` with samples from a Bernoulli distribution.
    pub fn rand_bernoulli(name: &str, length: usize, p: f64) -> Result<Self> {
        let dist = match Bernoulli::new(p) {
            Ok(dist) => dist,
            Err(e) => return Err(PolarsError::RandError(format!("{:?}", e))),
        };
        let mut builder = BooleanChunkedBuilder::new(name, length);
        for _ in 0..length {
            let smpl = dist.sample(&mut rand::thread_rng());
            builder.append_value(smpl)
        }
        Ok(builder.finish())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sample() {
        let df = df![
            "foo" => &[1, 2, 3, 4, 5]
        ]
        .unwrap();

        assert!(df.sample_n(3).is_ok());
        assert!(df.sample_frac(0.4).is_ok());
        assert!(df.sample_frac(2.0).is_err());
    }
}
