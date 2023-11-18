use polars_core::prelude::DataType::Float64;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, PartialEq, Debug, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum RandomMethod {
    Shuffle,
    Sample {
        is_fraction: bool,
        with_replacement: bool,
        shuffle: bool,
    },
}

impl Hash for RandomMethod {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state)
    }
}

pub(super) fn shuffle(s: &Series, seed: Option<u64>) -> PolarsResult<Series> {
    Ok(s.shuffle(seed))
}

pub(super) fn sample_frac(
    s: &[Series],
    with_replacement: bool,
    shuffle: bool,
    seed: Option<u64>,
) -> PolarsResult<Series> {
    let src = &s[0];
    let frac_s = &s[1];

    polars_ensure!(
        frac_s.len() == 1,
        ComputeError: "Sample fraction must be a single value."
    );

    let frac_s = frac_s.cast(&Float64)?;
    let frac = frac_s.f64()?;

    match frac.get(0) {
        Some(frac) => src.sample_frac(frac, with_replacement, shuffle, seed),
        None => Ok(Series::new_empty(src.name(), src.dtype())),
    }
}

pub(super) fn sample_n(
    s: &[Series],
    with_replacement: bool,
    shuffle: bool,
    seed: Option<u64>,
) -> PolarsResult<Series> {
    let src = &s[0];
    let n_s = &s[1];

    polars_ensure!(
        n_s.len() == 1,
        ComputeError: "Sample size must be a single value."
    );

    let n_s = n_s.cast(&IDX_DTYPE)?;
    let n = n_s.idx()?;

    match n.get(0) {
        Some(n) => src.sample_n(n as usize, with_replacement, shuffle, seed),
        None => Ok(Series::new_empty(src.name(), src.dtype())),
    }
}
