use crate::prelude::*;
use num::{Float, NumCast};
use rand::distributions::Bernoulli;
use rand::prelude::*;
use rand_distr::{Distribution, Normal, StandardNormal, Uniform};

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
