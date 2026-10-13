//! Plan-time statistics.
//!
//! [`ScanStats`] is what a scan leaf resolves at plan time. [`node_stats`]
//! estimates the same shape for any node of the IR, and [`subplan_cost`] sums that
//! over a whole subtree.

mod cost;
#[cfg(any(feature = "python", test))]
mod dataset;
mod node;

use std::sync::Arc;

pub(crate) use cost::subplan_cost;
#[cfg(feature = "python")]
pub(crate) use dataset::dataset_scan_stats;
pub use node::{NodeStats, composite_key_domain, join_cardinality, key_domain, node_stats};
pub(crate) use node::{StatsCache, node_stats_with_cache};
#[allow(clippy::disallowed_types)]
use polars_utils::aliases::PlHashMap;
use polars_utils::pl_str::PlSmallStr;

use crate::plans::IR;

/// Relative error for an estimate that carries no better information.
pub(crate) const DEFAULT_REL_ERR: f32 = 0.5;

/// A cardinality: a count that may be unknown, guaranteed, or estimated.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum Card {
    #[default]
    Unknown,
    /// Guaranteed by the source, e.g. a parquet footer or an Iceberg manifest.
    Exact(u64),
    /// Derived. `rel_err` is a rough 1-sigma relative error.
    Approx { value: u64, rel_err: f32 },
}

impl Card {
    /// An estimate with the default relative error.
    pub fn approx(value: u64) -> Self {
        Card::Approx {
            value,
            rel_err: DEFAULT_REL_ERR,
        }
    }

    /// The value, exact or estimated.
    pub fn value(self) -> Option<u64> {
        match self {
            Card::Unknown => None,
            Card::Exact(v) | Card::Approx { value: v, .. } => Some(v),
        }
    }

    /// The value, only when it is known to within `max_rel_err`.
    pub fn confident(self, max_rel_err: f32) -> Option<u64> {
        match self {
            Card::Unknown => None,
            Card::Exact(v) => Some(v),
            Card::Approx { value, rel_err } => (rel_err <= max_rel_err).then_some(value),
        }
    }

    /// Turn a guarantee into an estimate.
    pub fn demote(self, rel_err: f32) -> Self {
        match self {
            Card::Unknown => Card::Unknown,
            Card::Exact(value) => Card::Approx { value, rel_err },
            Card::Approx { value, rel_err: e } => Card::Approx {
                value,
                rel_err: e.max(rel_err),
            },
        }
    }

    /// [`Card::demote`] with the default relative error.
    pub fn demote_default(self) -> Self {
        self.demote(DEFAULT_REL_ERR)
    }

    /// Apply `f` to the value, keeping the confidence.
    pub fn map(self, f: impl FnOnce(u64) -> u64) -> Self {
        match self {
            Card::Unknown => Card::Unknown,
            Card::Exact(v) => Card::Exact(f(v)),
            Card::Approx { value, rel_err } => Card::Approx {
                value: f(value),
                rel_err,
            },
        }
    }
}

/// Statistics for one column, keyed on the output column name, i.e. after column
/// mapping.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct ScanColumnStats {
    /// Number of distinct values.
    pub distinct: Card,
    pub null_count: Card,
    /// Average width of one value in the source, in bytes.
    pub avg_byte_width: Option<f32>,
    /// Inclusive value range of an integer column, folded over the chunks that were
    /// read. When the chunks were sampled, it is widened by the mean gap between
    /// them and is an estimate.
    #[cfg_attr(feature = "serde", serde(default))]
    pub int_range: Option<(i128, i128)>,
    /// `int_range` misses sources whose statistics were not read, or comes from a
    /// single sampled row group, so values far outside it may exist.
    #[cfg_attr(feature = "serde", serde(default))]
    pub int_range_partial: bool,
}

impl ScanColumnStats {
    /// Values the column could hold, from its integer range.
    pub fn int_domain(&self) -> Option<f64> {
        let (min, max) = self.int_range?;
        (max >= min).then(|| range_width(min, max))
    }
}

/// Values in `min..=max`. The count can exceed `i128`, e.g. for a 128-bit decimal.
pub(crate) fn range_width(min: i128, max: i128) -> f64 {
    max.checked_sub(min)
        .and_then(|width| width.checked_add(1))
        .map_or(max as f64 - min as f64 + 1.0, |width| width as f64)
}

// We don't index
#[allow(clippy::disallowed_types)]
pub type ScanColumnStatsMap = PlHashMap<PlSmallStr, ScanColumnStats>;

/// Plan-time statistics for a scan.
///
/// Derivative: takes no part in plan equality or hashing.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct ScanStats {
    #[cfg_attr(feature = "serde", serde(default))]
    pub rows: Card,
    #[cfg_attr(feature = "serde", serde(default))]
    #[cfg_attr(
        feature = "dsl-schema",
        schemars(with = "Option<Vec<(PlSmallStr, ScanColumnStats)>>")
    )]
    columns: Option<Arc<ScanColumnStatsMap>>,
}

impl ScanStats {
    pub fn unknown() -> Self {
        Self::default()
    }

    pub fn new(rows: Card) -> Self {
        ScanStats {
            rows,
            columns: None,
        }
    }

    pub fn exact_rows(rows: u64) -> Self {
        Self::new(Card::Exact(rows))
    }

    pub fn approx_rows(rows: u64) -> Self {
        Self::new(Card::approx(rows))
    }

    pub fn with_columns(mut self, columns: ScanColumnStatsMap) -> Self {
        self.columns = (!columns.is_empty()).then(|| Arc::new(columns));
        self
    }

    pub fn column(&self, name: &str) -> Option<&ScanColumnStats> {
        self.columns.as_ref()?.get(name)
    }
}

/// Row count of a leaf node.
pub fn leaf_row_count(ir: &IR) -> Card {
    match ir {
        IR::Scan { file_info, .. } => file_info.stats.rows,
        IR::DataFrameScan { df, .. } => Card::Exact(df.height() as u64),
        _ => Card::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_domain_beyond_i128() {
        let max = 10i128.pow(38) - 1;
        let stats = ScanColumnStats {
            int_range: Some((-max, max)),
            ..Default::default()
        };
        assert_eq!(stats.int_domain(), Some(2e38));
    }
}
