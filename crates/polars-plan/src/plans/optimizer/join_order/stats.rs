//! Row-count estimates for join ordering.
//!
//! The only statistics read from the plan are row counts: a scan's estimate and an
//! in-memory frame's height. Everything else is constants and assumptions. Anything
//! that cannot be estimated makes the whole cluster non-reorderable rather than
//! producing a guess.

use polars_utils::arena::{Arena, Node};
use recursive::recursive;

use crate::plans::{AExpr, ExprIR, IR, MintermIter};

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
/// `None` means the subtree is not modelled (a union, a sort with a slice, a python scan),
/// which leaves the whole cluster alone.
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

        // An in-memory frame carries its rows with it, so the count is exact.
        IR::DataFrameScan { df, .. } => {
            let rows = (df.height() as f64).max(MIN_CARDINALITY);
            Some(LeafStats {
                filtered: rows,
                unfiltered: rows,
            })
        },

        // A group-by emits one row per distinct key, which is never more than its
        // input and is exactly one when there is no key at all.
        IR::GroupBy {
            input,
            keys,
            options,
            apply,
            ..
        } => {
            // A user function may emit any number of rows per group, and a
            // rolling or dynamic group-by counts windows rather than keys.
            if apply.is_some() || options.is_rolling() || options.is_dynamic() {
                return None;
            }
            let inner = leaf_stats(*input, ir_arena, expr_arena)?;
            Some(one_row_per_group(inner, keys.len(), options.slice))
        },

        // `unique` keeps one row per distinct combination of its subset, which is a
        // group-by over those columns. No subset means every column.
        IR::Distinct { input, options } => {
            let n_keys = match &options.subset {
                Some(subset) => subset.len(),
                None => ir_arena.get(*input).schema(ir_arena).len(),
            };
            let inner = leaf_stats(*input, ir_arena, expr_arena)?;
            Some(one_row_per_group(inner, n_keys, options.slice))
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

        // A `select` keeps its input's height when every expression does. Aggregates
        // collapse the frame to one row, and a mix of the two broadcasts the scalars
        // back up to the height of the rest.
        IR::Select { input, expr, .. } => {
            if expr.is_empty() || !expr.iter().all(|e| keeps_height(e, expr_arena)) {
                return None;
            }
            let inner = leaf_stats(*input, ir_arena, expr_arena)?;
            if expr.iter().all(|e| e.is_scalar(expr_arena)) {
                return Some(LeafStats {
                    filtered: MIN_CARDINALITY,
                    unfiltered: MIN_CARDINALITY,
                });
            }
            Some(inner)
        },

        // `with_columns` adds to the frame it is given rather than replacing it, so
        // the height is the input's even when every expression is a scalar.
        IR::HStack { input, exprs, .. } => leaf_stats(*input, ir_arena, expr_arena),

        // Anything else (union, sort with a slice, python scan, ...) is not
        // modelled. Do not guess.
        _ => None,
    }
}

/// Whether an expression leaves the frame's height alone, either by producing one
/// value per row or by producing a scalar that broadcasts back to it.
///
/// Anything else — an explode, a range, a slice — sets the height from something
/// other than the input, and is not modelled.
fn keeps_height(expr: &ExprIR, expr_arena: &Arena<AExpr>) -> bool {
    expr.is_length_preserving(expr_arena) || expr.is_scalar(expr_arena)
}

/// Estimates for a node emitting one row per distinct combination of `n_keys`
/// columns, optionally sliced.
fn one_row_per_group(inner: LeafStats, n_keys: usize, slice: Option<(i64, usize)>) -> LeafStats {
    let cap = match slice {
        Some((_, len)) => len as f64,
        None => f64::INFINITY,
    };
    LeafStats {
        filtered: n_groups(inner.filtered, n_keys).min(cap),
        unfiltered: n_groups(inner.unfiltered, n_keys).min(cap),
    }
}

/// Estimated distinct combinations of `n_keys` grouping keys over `rows` rows.
///
/// No distinct counts are available, so this interpolates between the two ends that
/// are known: no keys is a single group, and any number of keys is at most one group
/// per row. Each key added moves the estimate closer to the input.
fn n_groups(rows: f64, n_keys: usize) -> f64 {
    if n_keys == 0 {
        return MIN_CARDINALITY;
    }
    let exponent = n_keys as f64 / (n_keys as f64 + 1.0);
    rows.powf(exponent).clamp(MIN_CARDINALITY, rows)
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

    /// The two ends of the group-count estimate are known exactly; only what lies
    /// between them is assumed.
    #[test]
    fn group_count_is_bounded_by_one_and_the_input() {
        // No key at all is a global aggregate.
        assert_eq!(n_groups(1_000_000.0, 0), 1.0);
        // One group per row is the most a group-by can emit.
        assert!(n_groups(1000.0, 8) <= 1000.0);
        assert_eq!(n_groups(1.0, 3), 1.0);
        // Adding a key can only move the estimate towards the input.
        let by_keys: Vec<f64> = (1..6).map(|k| n_groups(1_000_000.0, k)).collect();
        assert!(by_keys.windows(2).all(|w| w[0] < w[1]), "{by_keys:?}");
        assert!(by_keys.iter().all(|&g| g < 1_000_000.0), "{by_keys:?}");
    }

    #[test]
    fn selectivity_compounds_per_conjunct_and_never_reaches_zero() {
        assert!((apply_selectivity(1000.0, 1) - 200.0).abs() < 1e-9);
        assert!((apply_selectivity(1000.0, 2) - 40.0).abs() < 1e-9);
        // However many conjuncts pile up, an estimate never reaches zero.
        assert_eq!(apply_selectivity(1.0, 40), MIN_CARDINALITY);
    }
}
