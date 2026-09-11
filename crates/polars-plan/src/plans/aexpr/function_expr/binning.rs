pub use polars_core::chunked_array::ops::binning::{FractionSpec, IntervalSpec};

use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Debug, Hash)]
pub struct IRBinOptions {
    pub method: IRBinMethod,
    /// `None` emits integer indices (`labels=False` in Python).
    pub labels: Option<Vec<PlSmallStr>>,
    pub include_intervals: bool,
}

/// The resolved counterpart of [`BinMethod`](crate::dsl::BinMethod).
///
/// The breakpoint dtype is known, and their sorteness is validated.
#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Debug, Hash)]
pub enum IRBinMethod {
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

impl IRBinMethod {
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
}
