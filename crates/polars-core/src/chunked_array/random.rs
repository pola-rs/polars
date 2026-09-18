use rand::prelude::*;
use rand::seq::index::IndexVec;
use rand_distr::Uniform;

use crate::prelude::DataType::Float64;
use crate::prelude::*;
use crate::random::get_global_random_u64;
use crate::utils::NoNull;

fn create_rand_index_with_replacement(
    n: usize,
    len: usize,
    seed: Option<u64>,
    shuffle: Option<bool>,
) -> IdxCa {
    if len == 0 {
        return IdxCa::new_vec(PlSmallStr::EMPTY, vec![]);
    }
    let mut rng = SmallRng::seed_from_u64(seed.unwrap_or_else(get_global_random_u64));
    let dist = Uniform::new(0, len as IdxSize).unwrap();
    let idxs = (0..n as IdxSize)
        .map(move |_| dist.sample(&mut rng))
        .collect_trusted::<NoNull<IdxCa>>()
        .into_inner();
    if shuffle == Some(false) {
        idxs.sort(false)
    } else {
        idxs
    }
}

fn create_rand_index_no_replacement(
    n: usize,
    len: usize,
    seed: Option<u64>,
    shuffle: Option<bool>,
) -> IdxCa {
    let mut rng = SmallRng::seed_from_u64(seed.unwrap_or_else(get_global_random_u64));
    let mut buf: Vec<IdxSize>;
    if n == len {
        buf = (0..len as IdxSize).collect();
        // None and Some(false) coincide here because the natural output is already ordered and
        // forcing a shuffle would violate the fastest algorithm contract for None
        if let Some(true) = shuffle {
            buf.shuffle(&mut rng);
        }
    } else {
        // TODO: avoid extra potential copy by vendoring rand::seq::index::sample,
        // or genericize take over slices over any unsigned type. The optimizer
        // should get rid of the extra copy already if IdxSize matches the IndexVec
        // size returned.
        buf = match rand::seq::index::sample(&mut rng, len, n) {
            IndexVec::U32(v) => v.into_iter().map(|x| x as IdxSize).collect(),
            #[cfg(target_pointer_width = "64")]
            IndexVec::U64(v) => v.into_iter().map(|x| x as IdxSize).collect(),
        };
        // None and Some(true) coincide here because the rand::seq::index::sample
        // already returns indices in an unspecified order so neither needs additional work
        if let Some(false) = shuffle {
            buf.sort_unstable();
        }
    }
    IdxCa::new_vec(PlSmallStr::EMPTY, buf)
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
        shuffle: Option<bool>,
        seed: Option<u64>,
    ) -> PolarsResult<Self> {
        ensure_shape(n, self.len(), with_replacement)?;
        if n == 0 {
            return Ok(self.clear());
        }
        let len = self.len();

        match with_replacement {
            true => {
                let idx = create_rand_index_with_replacement(n, len, seed, shuffle);
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
        shuffle: Option<bool>,
        seed: Option<u64>,
    ) -> PolarsResult<Self> {
        let n = (self.len() as f64 * frac) as usize;
        self.sample_n(n, with_replacement, shuffle, seed)
    }

    pub fn shuffle(&self, seed: Option<u64>) -> Self {
        let len = self.len();
        let n = len;
        let idx = create_rand_index_no_replacement(n, len, seed, Some(true));
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
        shuffle: Option<bool>,
        seed: Option<u64>,
    ) -> PolarsResult<Self> {
        ensure_shape(n, self.len(), with_replacement)?;
        let len = self.len();

        match with_replacement {
            true => {
                let idx = create_rand_index_with_replacement(n, len, seed, shuffle);
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
        shuffle: Option<bool>,
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
        shuffle: Option<bool>,
        seed: Option<u64>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
        n.len() == 1,
        ComputeError: "Sample size must be a single value."
        );

        let n = n.strict_cast(&IDX_DTYPE)?;
        let n = n.idx()?;

        match n.get(0) {
            Some(n) => self.sample_n_literal(n as usize, with_replacement, shuffle, seed),
            None => Ok(self.clear()),
        }
    }

    pub fn sample_n_literal(
        &self,
        n: usize,
        with_replacement: bool,
        shuffle: Option<bool>,
        seed: Option<u64>,
    ) -> PolarsResult<Self> {
        ensure_shape(n, self.height(), with_replacement)?;
        // All columns should used the same indices. So we first create the indices.
        let idx = match with_replacement {
            true => create_rand_index_with_replacement(n, self.height(), seed, shuffle),
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
        shuffle: Option<bool>,
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
            None => Ok(self.clear()),
        }
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
        assert!(
            df.sample_n(
                &Series::new(PlSmallStr::from_static("s"), &[3]),
                false,
                None,
                None
            )
            .is_ok()
        );
        assert!(
            df.sample_frac(
                &Series::new(PlSmallStr::from_static("frac"), &[0.4]),
                false,
                None,
                None
            )
            .is_ok()
        );
        // With seeding.
        assert!(
            df.sample_n(
                &Series::new(PlSmallStr::from_static("s"), &[3]),
                false,
                None,
                Some(0)
            )
            .is_ok()
        );
        assert!(
            df.sample_frac(
                &Series::new(PlSmallStr::from_static("frac"), &[0.4]),
                false,
                None,
                Some(0)
            )
            .is_ok()
        );
        // Without replacement can not sample more than 100%.
        assert!(
            df.sample_frac(
                &Series::new(PlSmallStr::from_static("frac"), &[2.0]),
                false,
                None,
                Some(0)
            )
            .is_err()
        );
        assert!(
            df.sample_n(
                &Series::new(PlSmallStr::from_static("s"), &[3]),
                true,
                None,
                Some(0)
            )
            .is_ok()
        );
        assert!(
            df.sample_frac(
                &Series::new(PlSmallStr::from_static("frac"), &[0.4]),
                true,
                None,
                Some(0)
            )
            .is_ok()
        );
        // With replacement can sample more than 100%.
        assert!(
            df.sample_frac(
                &Series::new(PlSmallStr::from_static("frac"), &[2.0]),
                true,
                None,
                Some(0)
            )
            .is_ok()
        );
    }
}
