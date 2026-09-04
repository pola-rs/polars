use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::ops::Deref;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::CHEAP_SERIES_HASH_LIMIT;
use crate::prelude::*;
use crate::utils::Wrap;

/// Breakpoints delimiting bins by value.
///
/// Always free of nulls and non-decreasing.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug)]
pub struct Breaks(Series);

impl Breaks {
    pub fn new(breaks: Series) -> PolarsResult<Self> {
        polars_ensure!(
            breaks.null_count() == 0,
            ComputeError: "breakpoints cannot contain nulls"
        );
        let n = breaks.len();
        if n >= 2 {
            let non_decreasing = breaks.slice(0, n - 1).lt_eq(&breaks.slice(1, n - 1))?;
            polars_ensure!(
                non_decreasing.all(),
                ComputeError: "breakpoints must be non-decreasing"
            );
        }
        Ok(Self(breaks))
    }
}

/// Cumulative fractions delimiting bins by proportion.
///
/// Always non-decreasing and within `[0, 1]`.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug)]
pub struct Fractions(Vec<f64>);

impl Fractions {
    pub fn new(fractions: Vec<f64>) -> PolarsResult<Self> {
        for x in &fractions {
            polars_ensure!(
                (0.0..=1.0).contains(x),
                ComputeError: "fractions must be between 0.0 and 1.0, got {}", x
            );
        }
        polars_ensure!(
            fractions.is_sorted(),
            ComputeError: "fractions must be non-decreasing"
        );
        Ok(Self(fractions))
    }
}

/// How interval binning delimits its bins.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug, Hash)]
pub enum IntervalSpec {
    /// Explicit breakpoints.
    Breaks(Breaks),
    /// `n` equal-width bins spanning `[min, max]`.
    Count(NonZeroUsize),
}

/// How quantile and rank binning delimit their bins.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug, Hash)]
pub enum FractionSpec {
    /// Explicit fractions.
    Explicit(Fractions),
    /// `n` bins of equal probability, or of equal size for rank binning.
    Count(NonZeroUsize),
}

fn n_bins(n_bins: usize) -> PolarsResult<NonZeroUsize> {
    NonZeroUsize::new(n_bins)
        .ok_or_else(|| polars_err!(ComputeError: "binning requires at least one bin"))
}

impl IntervalSpec {
    pub fn from_breaks(breaks: Series) -> PolarsResult<Self> {
        Ok(Self::Breaks(Breaks::new(breaks)?))
    }

    pub fn from_count(count: usize) -> PolarsResult<Self> {
        Ok(Self::Count(n_bins(count)?))
    }

    pub fn n_bins(&self) -> usize {
        match self {
            Self::Breaks(breaks) => breaks.len() + 1,
            Self::Count(n_bins) => n_bins.get(),
        }
    }

    pub fn breaks(&self) -> Option<&Series> {
        match self {
            Self::Breaks(breaks) => Some(&breaks.0),
            Self::Count(_) => None,
        }
    }
}

impl FractionSpec {
    pub fn from_fractions(fractions: Vec<f64>) -> PolarsResult<Self> {
        Ok(Self::Explicit(Fractions::new(fractions)?))
    }

    pub fn from_count(count: usize) -> PolarsResult<Self> {
        Ok(Self::Count(n_bins(count)?))
    }

    pub fn n_bins(&self) -> usize {
        match self {
            Self::Explicit(fractions) => fractions.len() + 1,
            Self::Count(n_bins) => n_bins.get(),
        }
    }
}

impl Deref for Breaks {
    type Target = Series;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for Fractions {
    type Target = [f64];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Hash for Breaks {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Wrap(self.0.slice(0, CHEAP_SERIES_HASH_LIMIT)).hash(state)
    }
}

impl Hash for Fractions {
    fn hash<H: Hasher>(&self, state: &mut H) {
        bytemuck::cast_slice::<_, u64>(&self.0).hash(state)
    }
}
