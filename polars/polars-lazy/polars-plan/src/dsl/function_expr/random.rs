use std::sync::atomic::Ordering;

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

pub(super) fn random(
    s: &Series,
    method: RandomMethod,
    atomic_seed: Option<&Arc<AtomicU64>>,
    seed: Option<u64>,
    fixed_seed: bool,
) -> PolarsResult<Series> {
    let seed = if fixed_seed {
        seed
    } else {
        // ensure seeds differ between groupby groups
        // otherwise all groups would be sampled the same
        atomic_seed
            .as_ref()
            .map(|atomic| atomic.fetch_add(1, Ordering::Relaxed))
    };
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
