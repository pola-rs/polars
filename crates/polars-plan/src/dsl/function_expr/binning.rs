use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;

use polars_core::CHEAP_SERIES_HASH_LIMIT;
pub use polars_core::chunked_array::ops::binning::FractionSpec;
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
#[derive(Clone, PartialEq, Debug, Hash)]
pub enum BinMethod {
    /// Bins delimited by values.
    Intervals {
        spec: DslIntervalSpec,
        right_closed: bool,
    },
    /// Bins delimited by quantiles of the input, keeps equal values in a single bin.
    Quantiles {
        spec: FractionSpec,
        right_closed: bool,
    },
    /// Bins delimited by position in sorted order, equal values split across bins.
    Ranks { spec: FractionSpec },
}

/// How interval binning delimits its bins, as the caller specified it.
///
/// The breakpoints are in whichever dtype they arrived in. We can not enforce ordering
/// here, e.g. because we would need to cast strings to Enum/Categorical.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug)]
pub enum DslIntervalSpec {
    /// Explicit breakpoints.
    Breaks(Series),
    /// `n` equal-width bins spanning `[min, max]`.
    Count(NonZeroUsize),
}

impl DslIntervalSpec {
    pub fn from_breaks(breaks: Series) -> Self {
        Self::Breaks(breaks)
    }

    pub fn from_count(count: usize) -> PolarsResult<Self> {
        Ok(Self::Count(NonZeroUsize::new(count).ok_or_else(
            || polars_err!(ComputeError: "binning requires at least one bin"),
        )?))
    }
}

impl Hash for DslIntervalSpec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Breaks(breaks) => Wrap(breaks.slice(0, CHEAP_SERIES_HASH_LIMIT)).hash(state),
            Self::Count(n_bins) => n_bins.hash(state),
        }
    }
}

impl BinMethod {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Intervals { .. } => "bin_intervals",
            Self::Quantiles { .. } => "bin_quantiles",
            Self::Ranks { .. } => "bin_ranks",
        }
    }

    /// Whether the specification requires numeric input.
    pub fn requires_numeric_input(&self) -> bool {
        match self {
            Self::Intervals { spec, .. } => matches!(spec, DslIntervalSpec::Count(_)),
            Self::Quantiles { .. } => true,
            Self::Ranks { .. } => false,
        }
    }
}
