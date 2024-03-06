use num_traits::{Float, NumCast};
use polars_error::to_compute_err;
use rand::distributions::Bernoulli;
use rand::prelude::*;
use rand::seq::index::IndexVec;
use rand_distr::{Normal, Standard, StandardNormal, Uniform};

use crate::prelude::DataType::Float64;
use crate::prelude::*;
use crate::random::get_global_random_u64;
use crate::utils::NoNull;

fn create_rand_index_with_replacement(n: usize, len: usize, seed: Option<u64>) -> IdxCa {
    if len == 0 {
        return IdxCa::new_vec("", vec![]);
    }
    let mut rng = SmallRng::seed_from_u64(seed.unwrap_or_else(get_global_random_u64));
    let dist = Uniform::new(0, len as IdxSize);
    (0..n as IdxSize)
        .map(move |_| dist.sample(&mut rng))
        .collect_trusted::<NoNull<IdxCa>>()
        .into_inner()
}

fn create_rand_index_no_replacement(
    n: usize,
    len: usize,
    seed: Option<u64>,
    shuffle: bool,
) -> IdxCa {
    let mut rng = SmallRng::seed_from_u64(seed.unwrap_or_else(get_global_random_u64));
    let mut buf: Vec<IdxSize>;
    if n == len {
        buf = (0..len as IdxSize).collect();
        if shuffle {
            buf.shuffle(&mut rng)
        }
    } else {
        // TODO: avoid extra potential copy by vendoring rand::seq::index::sample,
        // or genericize take over slices over any unsigned type. The optimizer
        // should get rid of the extra copy already if IdxSize matches the IndexVec
        // size returned.
        buf = match rand::seq::index::sample(&mut rng, len, n) {
            IndexVec::U32(v) => v.into_iter().map(|x| x as IdxSize).collect(),
            IndexVec::USize(v) => v.into_iter().map(|x| x as IdxSize).collect(),
        };
    }
    IdxCa::new_vec("", buf)
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
    Standard: Distribution<T::Native>,
{
    pub fn init_rand(size: usize, null_density: f32, seed: Option<u64>) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed.unwrap_or_else(get_global_random_u64));
        (0..size)
            .map(|_| {
                if rng.gen::<f32>() < null_density {
                    None
                } else {
                    Some(rng.gen())
                }
            })
            .collect()
    }
}

fn ensure_shape(n: usize, len: usize, with_replacement: bool) -> PolarsResult<()> {
    polars_ensure!(
        with_replacement || n <= len,
        ShapeMismatch:
        "cannot take a larger sample than the total population when `with_replacement=false`"
    );
    Ok(())
}

impl Series {
    pub fn sample_n(
        &self,
        n: usize,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PolarsResult<Self> {
        ensure_shape(n, self.len(), with_replacement)?;
        if n == 0 {
            return Ok(self.clear());
        }
        let len = self.len();

        match with_replacement {
            true => {
                let idx = create_rand_index_with_replacement(n, len, seed);
                debug_assert_eq!(len, self.len());
                // SAFETY: we know that we never go out of bounds.
                unsafe { Ok(self.take_unchecked(&idx)) }
            },
            false => {
                let idx = create_rand_index_no_replacement(n, len, seed, shuffle);
                debug_assert_eq!(len, self.len());
                // SAFETY: we know that we never go out of bounds.
                unsafe { Ok(self.take_unchecked(&idx)) }
            },
        }
    }

    /// Sample a fraction between 0.0-1.0 of this [`ChunkedArray`].
    pub fn sample_frac(
        &self,
        frac: f64,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PolarsResult<Self> {
        let n = (self.len() as f64 * frac) as usize;
        self.sample_n(n, with_replacement, shuffle, seed)
    }

    pub fn shuffle(&self, seed: Option<u64>) -> Self {
        let len = self.len();
        let n = len;
        let idx = create_rand_index_no_replacement(n, len, seed, true);
        debug_assert_eq!(len, self.len());
        // SAFETY: we know that we never go out of bounds.
        unsafe { self.take_unchecked(&idx) }
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsDataType,
    ChunkedArray<T>: ChunkTake<IdxCa>,
{
    /// Sample n datapoints from this [`ChunkedArray`].
    pub fn sample_n(
        &self,
        n: usize,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PolarsResult<Self> {
        ensure_shape(n, self.len(), with_replacement)?;
        let len = self.len();

        match with_replacement {
            true => {
                let idx = create_rand_index_with_replacement(n, len, seed);
                debug_assert_eq!(len, self.len());
                // SAFETY: we know that we never go out of bounds.
                unsafe { Ok(self.take_unchecked(&idx)) }
            },
            false => {
                let idx = create_rand_index_no_replacement(n, len, seed, shuffle);
                debug_assert_eq!(len, self.len());
                // SAFETY: we know that we never go out of bounds.
                unsafe { Ok(self.take_unchecked(&idx)) }
            },
        }
    }

    /// Sample a fraction between 0.0-1.0 of this [`ChunkedArray`].
    pub fn sample_frac(
        &self,
        frac: f64,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PolarsResult<Self> {
        let n = (self.len() as f64 * frac) as usize;
        self.sample_n(n, with_replacement, shuffle, seed)
    }
}

impl DataFrame {
    /// Sample n datapoints from this [`DataFrame`].
    pub fn sample_n(
        &self,
        n: &Series,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
        n.len() == 1,
        ComputeError: "Sample size must be a single value."
        );

        let n = n.cast(&IDX_DTYPE)?;
        let n = n.idx()?;

        match n.get(0) {
            Some(n) => self.sample_n_literal(n as usize, with_replacement, shuffle, seed),
            None => {
                let new_cols = self.columns.iter().map(Series::clear).collect_trusted();
                Ok(unsafe { DataFrame::new_no_checks(new_cols) })
            },
        }
    }

    pub fn sample_n_literal(
        &self,
        n: usize,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PolarsResult<Self> {
        ensure_shape(n, self.height(), with_replacement)?;
        // All columns should used the same indices. So we first create the indices.
        let idx = match with_replacement {
            true => create_rand_index_with_replacement(n, self.height(), seed),
            false => create_rand_index_no_replacement(n, self.height(), seed, shuffle),
        };
        // SAFETY: the indices are within bounds.
        Ok(unsafe { self.take_unchecked(&idx) })
    }

    /// Sample a fraction between 0.0-1.0 of this [`DataFrame`].
    pub fn sample_frac(
        &self,
        frac: &Series,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
        frac.len() == 1,
        ComputeError: "Sample fraction must be a single value."
        );

        let frac = frac.cast(&Float64)?;
        let frac = frac.f64()?;

        match frac.get(0) {
            Some(frac) => {
                let n = (self.height() as f64 * frac) as usize;
                self.sample_n_literal(n, with_replacement, shuffle, seed)
            },
            None => {
                let new_cols = self.columns.iter().map(Series::clear).collect_trusted();
                Ok(unsafe { DataFrame::new_no_checks(new_cols) })
            },
        }
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Float,
{
    /// Create [`ChunkedArray`] with samples from a Normal distribution.
    pub fn rand_normal(name: &str, length: usize, mean: f64, std_dev: f64) -> PolarsResult<Self> {
        let normal = Normal::new(mean, std_dev).map_err(to_compute_err)?;
        let mut builder = PrimitiveChunkedBuilder::<T>::new(name, length);
        let mut rng = rand::thread_rng();
        for _ in 0..length {
            let smpl = normal.sample(&mut rng);
            let smpl = NumCast::from(smpl).unwrap();
            builder.append_value(smpl)
        }
        Ok(builder.finish())
    }

    /// Create [`ChunkedArray`] with samples from a Standard Normal distribution.
    pub fn rand_standard_normal(name: &str, length: usize) -> Self {
        let mut builder = PrimitiveChunkedBuilder::<T>::new(name, length);
        let mut rng = rand::thread_rng();
        for _ in 0..length {
            let smpl: f64 = rng.sample(StandardNormal);
            let smpl = NumCast::from(smpl).unwrap();
            builder.append_value(smpl)
        }
        builder.finish()
    }

    /// Create [`ChunkedArray`] with samples from a Uniform distribution.
    pub fn rand_uniform(name: &str, length: usize, low: f64, high: f64) -> Self {
        let uniform = Uniform::new(low, high);
        let mut builder = PrimitiveChunkedBuilder::<T>::new(name, length);
        let mut rng = rand::thread_rng();
        for _ in 0..length {
            let smpl = uniform.sample(&mut rng);
            let smpl = NumCast::from(smpl).unwrap();
            builder.append_value(smpl)
        }
        builder.finish()
    }
}

impl BooleanChunked {
    /// Create [`ChunkedArray`] with samples from a Bernoulli distribution.
    pub fn rand_bernoulli(name: &str, length: usize, p: f64) -> PolarsResult<Self> {
        let dist = Bernoulli::new(p).map_err(to_compute_err)?;
        let mut rng = rand::thread_rng();
        let mut builder = BooleanChunkedBuilder::new(name, length);
        for _ in 0..length {
            let smpl = dist.sample(&mut rng);
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

        // Default samples are random and don't require seeds.
        assert!(df
            .sample_n(&Series::new("s", &[3]), false, false, None)
            .is_ok());
        assert!(df
            .sample_frac(&Series::new("frac", &[0.4]), false, false, None)
            .is_ok());
        // With seeding.
        assert!(df
            .sample_n(&Series::new("s", &[3]), false, false, Some(0))
            .is_ok());
        assert!(df
            .sample_frac(&Series::new("frac", &[0.4]), false, false, Some(0))
            .is_ok());
        // Without replacement can not sample more than 100%.
        assert!(df
            .sample_frac(&Series::new("frac", &[2.0]), false, false, Some(0))
            .is_err());
        assert!(df
            .sample_n(&Series::new("s", &[3]), true, false, Some(0))
            .is_ok());
        assert!(df
            .sample_frac(&Series::new("frac", &[0.4]), true, false, Some(0))
            .is_ok());
        // With replacement can sample more than 100%.
        assert!(df
            .sample_frac(&Series::new("frac", &[2.0]), true, false, Some(0))
            .is_ok());
    }
}
