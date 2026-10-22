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
pub use node::{
    NodeStats, composite_key_domain, join_cardinality, key_domain, node_stats, unique_side_rows,
};
pub(crate) use node::{StatsCache, node_stats_with_cache};
use polars_core::prelude::Scalar;
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

    fn rel_err(self) -> f32 {
        match self {
            Card::Unknown => f32::INFINITY,
            Card::Exact(_) => 0.0,
            Card::Approx { rel_err, .. } => rel_err,
        }
    }

    /// Combine two counts with `f`, keeping the lower confidence.
    fn combine(self, other: Self, f: impl FnOnce(u64, u64) -> u64) -> Self {
        match (self, other) {
            (Card::Exact(l), Card::Exact(r)) => Card::Exact(f(l, r)),
            (l, r) => match (l.value(), r.value()) {
                (Some(lv), Some(rv)) => Card::Approx {
                    value: f(lv, rv),
                    rel_err: l.rel_err().max(r.rel_err()),
                },
                _ => Card::Unknown,
            },
        }
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
    /// The statistics each sampled row group records of this column (bounds, counts
    /// and sizes), present only when `POLARS_SCAN_CHUNK_STATISTICS` is enabled.
    #[cfg_attr(feature = "serde", serde(skip))]
    #[cfg_attr(feature = "dsl-schema", schemars(skip))]
    /// One entry per sampled row group, ordered by source file, and by row group within a
    /// file.
    pub sampled_row_groups: Option<Arc<[RowGroupStats]>>,
}

/// How many bytes one stored value of a parquet column takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueWidth {
    /// Every value takes this many bytes, e.g. 4 for INT32 and 8 for INT64.
    Fixed(u32),
    /// Booleans, which are stored as bits.
    Bits,
    /// Byte arrays, whose values differ in length.
    Variable,
}

/// The statistics of one parquet column in one row group.
#[derive(Clone, Debug, PartialEq)]
pub struct RowGroupStats {
    pub rows: u64,
    pub null_count: Option<u64>,
    /// The number of distinct values, as read from the row group's statistics. Most
    /// writers don't write this.
    pub distinct_count: Option<u64>,
    /// How the column chunk stores its values, which writers choose differently for
    /// some types, e.g. a decimal as INT32 or as a fixed-length byte array.
    pub value_width: ValueWidth,
    /// The size in bytes of the column chunk once decompressed.
    pub uncompressed_size: u64,
    /// The size in bytes of the column chunk's dictionary page as stored on disk
    /// (compressed), if it has one.
    pub dictionary_page_size: Option<u64>,
    /// The minimum and maximum value, typed as the column, or as its values for a
    /// categorical column. `None` when the row group doesn't record both exactly, or
    /// they can't be converted to that type.
    pub bounds: Option<(Scalar, Scalar)>,
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

    /// Statistics of one scan that reads the sources of `parts` one after another. A column
    /// keeps statistics only if every part has them.
    pub(crate) fn concat(parts: &[&Self]) -> Self {
        let rows = parts
            .iter()
            .map(|s| s.rows)
            .reduce(|l, r| l.combine(r, u64::saturating_add))
            .unwrap_or_default();

        let Some(maps) = parts
            .iter()
            .map(|s| s.columns.as_deref())
            .collect::<Option<Vec<_>>>()
        else {
            return Self::new(rows);
        };

        let columns =
            maps[0]
                .iter()
                .filter_map(|(name, first)| {
                    let mut out = first.clone();
                    let mut width_rows = parts[0].rows.value().unwrap_or(0) as f64;
                    for (map, part) in maps.iter().zip(parts).skip(1) {
                        let col = map.get(name)?;
                        let part_rows = part.rows.value().unwrap_or(0) as f64;
                        out.distinct = out.distinct.combine(col.distinct, u64::max);
                        out.null_count =
                            out.null_count.combine(col.null_count, u64::saturating_add);
                        out.avg_byte_width =
                            out.avg_byte_width.zip(col.avg_byte_width).map(|(l, r)| {
                                let total = width_rows + part_rows;
                                if total == 0.0 {
                                    (l + r) / 2.0
                                } else {
                                    ((l as f64 * width_rows + r as f64 * part_rows) / total) as f32
                                }
                            });
                        out.int_range = out.int_range.zip(col.int_range).map(
                            |((l_min, l_max), (r_min, r_max))| (l_min.min(r_min), l_max.max(r_max)),
                        );
                        out.int_range_partial |= col.int_range_partial;
                        width_rows += part_rows;
                    }
                    Some((name.clone(), out))
                })
                .collect();

        Self::new(rows).with_columns(columns)
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
