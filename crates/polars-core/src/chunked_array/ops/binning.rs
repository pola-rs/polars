use std::hash::{Hash, Hasher};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::CHEAP_SERIES_HASH_LIMIT;
use crate::prelude::*;
use crate::utils::Wrap;

/// How interval binning delimits its bins.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug)]
pub enum IntervalSpec {
    /// Explicit breakpoints, strictly ascending.
    Breaks(Series),
    /// `n` equal-width bins spanning `[min, max]`.
    Count(usize),
}

/// How quantile and rank binning delimit their bins.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug)]
pub enum FractionSpec {
    /// Explicit strictly ascending fractions in `[0, 1]`.
    Explicit(Vec<f64>),
    /// `n` bins of equal probability, or of equal size for rank binning.
    Count(usize),
}

impl IntervalSpec {
    pub fn n_bins(&self) -> usize {
        match self {
            Self::Breaks(breaks) => breaks.len() + 1,
            Self::Count(n_bins) => *n_bins,
        }
    }

    pub fn breaks(&self) -> Option<&Series> {
        match self {
            Self::Breaks(breaks) => Some(breaks),
            Self::Count(_) => None,
        }
    }
}

impl FractionSpec {
    pub fn n_bins(&self) -> usize {
        match self {
            Self::Explicit(fractions) => fractions.len() + 1,
            Self::Count(n_bins) => *n_bins,
        }
    }
}

impl Hash for IntervalSpec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Breaks(breaks) => Wrap(breaks.slice(0, CHEAP_SERIES_HASH_LIMIT)).hash(state),
            Self::Count(n_bins) => n_bins.hash(state),
        }
    }
}

impl Hash for FractionSpec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Explicit(fractions) => bytemuck::cast_slice::<_, u64>(fractions).hash(state),
            Self::Count(n_bins) => n_bins.hash(state),
        }
    }
}
