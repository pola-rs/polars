//! Row-count estimates for join ordering.
//!
//! The only statistic read from the plan is a scan's row count; the rest are
//! constants and assumptions. Anything that cannot be estimated makes the whole
//! cluster non-reorderable rather than producing a guess.

use polars_utils::arena::{Arena, Node};
use recursive::recursive;

use crate::plans::{AExpr, IR, MintermIter};

/// Fallback selectivity for a filter conjunct with no better estimate.
const DEFAULT_SELECTIVITY: f64 = 0.2;

/// Floor for any estimate.
///
/// A leaf estimated at zero rows would compare smaller than everything else and
/// dominate every ordering decision.
const MIN_CARDINALITY: f64 = 1.0;

/// Row-count estimates for one leaf of a join cluster.
#[derive(Debug, Clone, Copy)]
pub(super) struct LeafStats {
    /// Estimated rows, after any filter pushed into the leaf.
    pub(super) filtered: f64,
    /// Estimated rows, before those filters.
    ///
    /// Used as the distinct-count proxy for this leaf's join keys. A join divides by
    /// the size of the key domain rather than the filtered relation, which is what
    /// carries a dimension's selectivity over to the fact table.
    pub(super) unfiltered: f64,
}

/// Estimate the rows produced by a leaf subtree.
///
/// `None` means the subtree is not modelled (a group-by, a union, a non-file
/// source), which leaves the whole cluster alone.
#[recursive]
pub(super) fn leaf_stats(
    node: Node,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> Option<LeafStats> {
    match ir_arena.get(node) {
        IR::Scan {
            file_info,
            predicate,
            ..
        } => {
            // `row_estimation.1` is `usize::MAX` when the source could not be counted.
            let rows = file_info.row_estimation.1;
            if rows == usize::MAX {
                return None;
            }
            let unfiltered = rows as f64;
            let filtered = match predicate {
                None => unfiltered,
                Some(p) => apply_selectivity(unfiltered, n_conjuncts(p.node(), expr_arena)),
            };
            Some(LeafStats {
                filtered,
                unfiltered,
            })
        },

        // A filter above the scan is equivalent to one pushed into it; predicate
        // pushdown normally leaves none here, but a leaf need not be a bare scan.
        IR::Filter { input, predicate } => {
            let inner = leaf_stats(*input, ir_arena, expr_arena)?;
            Some(LeafStats {
                filtered: apply_selectivity(
                    inner.filtered,
                    n_conjuncts(predicate.node(), expr_arena),
                ),
                unfiltered: inner.unfiltered,
            })
        },

        // Row-preserving, so they pass both estimates through untouched.
        IR::SimpleProjection { input, .. } | IR::Cache { input, .. } => {
            leaf_stats(*input, ir_arena, expr_arena)
        },

        // Anything else (group-by, union, distinct, sort with slice, python scan, ...)
        // is not modelled. Do not guess.
        _ => None,
    }
}

/// Number of `AND` conjuncts in a predicate.
fn n_conjuncts(node: Node, expr_arena: &Arena<AExpr>) -> u32 {
    MintermIter::new(node, expr_arena).count() as u32
}

fn apply_selectivity(rows: f64, n_conjuncts: u32) -> f64 {
    (rows * DEFAULT_SELECTIVITY.powi(n_conjuncts as i32)).max(MIN_CARDINALITY)
}

/// Estimate the rows produced by joining two relations of the given sizes.
///
/// `key_domain_product` is the product, over every equi-key pair bridging the two
/// sides, of that key's domain size (see [`key_domain`]).
///
/// This is `|A| * |B| / NDV(key)`, extended over multiple keys. With exact distinct
/// counts the divisor would be the `max` of the two sides. Only row counts are
/// available, which bound distinct counts from above and are tight only on the unique
/// side, so [`key_domain`] takes the `min` instead: the smaller relation is assumed
/// to hold the key uniquely.
pub(super) fn join_cardinality(left: f64, right: f64, key_domain_product: f64) -> f64 {
    // Every factor comes from `key_domain`, which floors at `MIN_CARDINALITY`, so
    // the product is always >= 1.
    (left * right / key_domain_product).max(MIN_CARDINALITY)
}

/// Domain size of a key joining two leaves, under the primary-key assumption.
pub(super) fn key_domain(left: &LeafStats, right: &LeafStats) -> f64 {
    left.unfiltered.min(right.unfiltered).max(MIN_CARDINALITY)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(unfiltered: f64, filtered: f64) -> LeafStats {
        LeafStats {
            filtered,
            unfiltered,
        }
    }

    /// Joining a heavily filtered dimension must shrink the fact table so that it
    /// sorts ahead of the alternatives.
    #[test]
    fn filtered_dimension_shrinks_the_fact_table() {
        let inventory = leaf(11_745_000.0, 11_745_000.0);
        // date_dim filtered from 73049 rows down to roughly 60.
        let date_dim = leaf(73_049.0, 60.0);
        let item = leaf(18_000.0, 18_000.0);
        let warehouse = leaf(5.0, 5.0);

        let with_date = join_cardinality(
            inventory.filtered,
            date_dim.filtered,
            key_domain(&inventory, &date_dim),
        );
        let with_item = join_cardinality(
            inventory.filtered,
            item.filtered,
            key_domain(&inventory, &item),
        );
        let with_warehouse = join_cardinality(
            inventory.filtered,
            warehouse.filtered,
            key_domain(&inventory, &warehouse),
        );

        // 11.7M * 60 / 73049
        assert!((with_date - 9_646.0).abs() < 50.0, "got {with_date}");
        // An unfiltered dimension joined on its key reproduces the fact table.
        assert!((with_item - 11_745_000.0).abs() < 1.0, "got {with_item}");
        assert!(
            (with_warehouse - 11_745_000.0).abs() < 1.0,
            "got {with_warehouse}"
        );

        assert!(with_date < with_item && with_date < with_warehouse);
    }

    /// Once the filtered dimension is folded in, the remaining joins must not
    /// re-inflate the intermediate.
    #[test]
    fn unfiltered_dimensions_do_not_inflate() {
        let inventory = leaf(11_745_000.0, 11_745_000.0);
        let item = leaf(18_000.0, 18_000.0);
        let acc = 9_646.0;

        let out = join_cardinality(acc, item.filtered, key_domain(&inventory, &item));
        assert!((out - acc).abs() < 1.0, "got {out}");
    }

    #[test]
    fn selectivity_compounds_per_conjunct_and_never_reaches_zero() {
        assert!((apply_selectivity(1000.0, 1) - 200.0).abs() < 1e-9);
        assert!((apply_selectivity(1000.0, 2) - 40.0).abs() < 1e-9);
        // However many conjuncts pile up, an estimate never reaches zero.
        assert_eq!(apply_selectivity(1.0, 40), MIN_CARDINALITY);
    }
}
