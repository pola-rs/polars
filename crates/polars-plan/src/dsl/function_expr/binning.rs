pub use polars_core::chunked_array::ops::binning::{FractionSpec, IntervalSpec};
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
        spec: IntervalSpec,
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

impl BinMethod {
    pub fn breaks(&self) -> Option<&Series> {
        match self {
            Self::Intervals { spec, .. } => spec.breaks(),
            Self::Quantiles { .. } | Self::Ranks { .. } => None,
        }
    }

    pub fn n_bins(&self) -> usize {
        match self {
            Self::Intervals { spec, .. } => spec.n_bins(),
            Self::Quantiles { spec, .. } | Self::Ranks { spec } => spec.n_bins(),
        }
    }

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
            Self::Intervals { spec, .. } => matches!(spec, IntervalSpec::Count(_)),
            Self::Quantiles { .. } => true,
            Self::Ranks { .. } => false,
        }
    }
}
