//! Plan-time statistics for scan leaves.

use crate::plans::IR;

/// Relative error for an estimate that carries no better information.
const DEFAULT_REL_ERR: f32 = 0.5;

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

/// Plan-time statistics for a scan.
///
/// Derivative: takes no part in plan equality or hashing.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct ScanStats {
    #[cfg_attr(feature = "serde", serde(default))]
    pub rows: Card,
}

impl ScanStats {
    pub fn unknown() -> Self {
        Self::default()
    }

    pub fn new(rows: Card) -> Self {
        ScanStats { rows }
    }

    pub fn exact_rows(rows: u64) -> Self {
        Self::new(Card::Exact(rows))
    }

    pub fn approx_rows(rows: u64) -> Self {
        Self::new(Card::approx(rows))
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
