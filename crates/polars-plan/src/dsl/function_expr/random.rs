#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, PartialEq, Debug, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum RandomMethod {
    Shuffle,
    SampleN {
        n: usize,
        with_replacement: bool,
        shuffle: bool,
    },
    SampleFrac {
        frac: f64,
        with_replacement: bool,
        shuffle: bool,
    },
}

impl Hash for RandomMethod {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state)
    }
}

pub(super) fn random(s: &Series, method: RandomMethod, seed: Option<u64>) -> PolarsResult<Series> {
    match method {
        RandomMethod::Shuffle => Ok(s.shuffle(seed)),
        RandomMethod::SampleFrac {
            frac,
            with_replacement,
            shuffle,
        } => s.sample_frac(frac, with_replacement, shuffle, seed),
        RandomMethod::SampleN {
            n,
            with_replacement,
            shuffle,
        } => s.sample_n(n, with_replacement, shuffle, seed),
    }
}
