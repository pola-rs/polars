use polars_core::CHEAP_SERIES_HASH_LIMIT;
use polars_core::utils::Wrap;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug, Hash)]
pub struct BinOptions {
    pub method: BinMethod,
    /// `None` emits integer indices (`labels=False` in Python).
    pub labels: Option<Vec<PlSmallStr>>,
    pub include_intervals: bool,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug)]
pub enum BinMethod {
    /// Explicit breakpoints, supports any orderable dtype.
    Intervals { breaks: Series, right_closed: bool },
    /// Equal-width bins over `[min, max]`, computed in the input dtype.
    UniformIntervals { n_bins: usize, right_closed: bool },
    /// Quantile probabilities in `[0, 1]`, strictly ascending.
    Quantiles { probs: Vec<f64>, right_closed: bool },
    /// Equiprobable bins, kept separate from Quantiles to allow an exact integer implementation.
    UniformQuantiles { n_bins: usize, right_closed: bool },
    /// Strictly ascending cumulative fractions in `[0, 1]`. Positional bins have no
    /// value boundary to close on.
    Ranks { fractions: Vec<f64> },
    /// Near-equal bins, with any remainder assigned to the first bins. Uniform
    /// fractions cannot express this distribution exactly.
    UniformRanks { n_bins: usize },
}

impl BinMethod {
    pub fn breaks(&self) -> Option<&Series> {
        match self {
            Self::Intervals { breaks, .. } => Some(breaks),
            _ => None,
        }
    }

    pub fn n_bins(&self) -> usize {
        match self {
            Self::Intervals { .. } => self.breaks().unwrap().len() + 1,
            Self::Quantiles { probs, .. } => probs.len() + 1,
            Self::Ranks { fractions } => fractions.len() + 1,
            Self::UniformIntervals { n_bins, .. }
            | Self::UniformQuantiles { n_bins, .. }
            | Self::UniformRanks { n_bins } => *n_bins,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Intervals { .. } | Self::UniformIntervals { .. } => "bin_intervals",
            Self::Quantiles { .. } | Self::UniformQuantiles { .. } => "bin_quantiles",
            Self::Ranks { .. } | Self::UniformRanks { .. } => "bin_ranks",
        }
    }

    /// Whether the specification requires numeric input.
    /// Quantiles are numeric-only despite only gathering values.
    pub fn requires_numeric_input(&self) -> bool {
        matches!(
            self,
            Self::UniformIntervals { .. } | Self::Quantiles { .. } | Self::UniformQuantiles { .. }
        )
    }
}

impl Hash for BinMethod {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Intervals {
                breaks,
                right_closed,
            } => {
                Hash::hash(
                    &Wrap(breaks.slice(0, CHEAP_SERIES_HASH_LIMIT).clone()),
                    state,
                );
                right_closed.hash(state);
            },
            Self::UniformIntervals {
                n_bins,
                right_closed,
            }
            | Self::UniformQuantiles {
                n_bins,
                right_closed,
            } => {
                n_bins.hash(state);
                right_closed.hash(state);
            },
            Self::Quantiles {
                probs,
                right_closed,
            } => {
                bytemuck::cast_slice::<_, u64>(probs).hash(state);
                right_closed.hash(state);
            },
            Self::Ranks { fractions } => bytemuck::cast_slice::<_, u64>(fractions).hash(state),
            Self::UniformRanks { n_bins } => n_bins.hash(state),
        }
    }
}
