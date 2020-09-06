use crate::prelude::*;
use num::{Float, NumCast};
use rand::prelude::*;
use rand_distr::{Distribution, Normal, StandardNormal};

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Float + NumCast,
{
    /// Create `ChunkedArray` with samples from a Normal distribution.
    pub fn rand_normal(name: &str, length: usize, mean: f64, std_dev: f64) -> Result<Self> {
        let normal = Normal::new(mean, std_dev)?;
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
}
