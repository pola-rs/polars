//! Cardinality estimates for a node of the IR.
//!
//! Anything that cannot be estimated yields `None` rather than a guess, so a
//! caller can tell "no estimate" apart from "estimated to be small".

use std::sync::Arc;

use polars_utils::aliases::PlIndexMap;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::slice_enum::Slice;
use recursive::recursive;

use super::{Card, DEFAULT_REL_ERR, ScanColumnStats, ScanColumnStatsMap, leaf_row_count};
use crate::plans::{
    AExpr, ExprIR, IR, IRBooleanFunction, IRFunctionExpr, MintermIter, into_column,
};

/// Fallback selectivity for a filter conjunct with no better estimate.
const DEFAULT_SELECTIVITY: f64 = 0.2;

/// Largest relative error an NDV may carry and still steer a decision.
///
/// A weak distinct count is worse than none: it moves join order and group
/// counts confidently in a direction nothing supports. Anything derived carries
/// [`DEFAULT_REL_ERR`]; a source claiming less confidence than that is ignored.
const MAX_NDV_REL_ERR: f32 = DEFAULT_REL_ERR;

/// Floor for any estimate.
///
/// A leaf estimated at zero rows would compare smaller than everything else and
/// dominate every ordering decision.
const MIN_CARDINALITY: f64 = 1.0;

/// Cardinality estimates for the rows a node emits.
#[derive(Debug, Clone, Default)]
pub struct NodeStats {
    /// Estimated rows, after any filter pushed into the leaf.
    pub filtered: f64,
    /// Estimated rows, before those filters.
    ///
    /// Used as the distinct-count proxy for this leaf's join keys. A join divides by
    /// the size of the key domain rather than the filtered relation, which is what
    /// carries a dimension's selectivity over to the fact table.
    pub unfiltered: f64,
    /// Per output column, sparse. An absent column is unknown.
    columns: Option<Arc<ScanColumnStatsMap>>,
}

/// Estimate the rows a subtree produces.
///
/// `None` means the subtree is not modelled (a python scan, an opaque function, a
/// gather by a computed index).
#[recursive]
pub fn node_stats(
    node: Node,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> Option<NodeStats> {
    let ir = ir_arena.get(node);
    match ir {
        IR::Scan {
            predicate,
            unified_scan_args,
            ..
        } => {
            let unfiltered = leaf_row_count(ir).value()? as f64;
            // The slice is applied before the predicate, so it narrows first.
            let mut filtered = match &unified_scan_args.pre_slice {
                None => unfiltered,
                Some(slice) => sliced(unfiltered, slice.clone()),
            };
            let columns = scan_columns(ir);
            if let Some(p) = predicate {
                filtered = apply_predicate(
                    filtered,
                    unfiltered,
                    p.node(),
                    expr_arena,
                    columns.as_deref(),
                );
            }
            Some(NodeStats {
                filtered,
                unfiltered,
                columns,
            })
        },

        // An in-memory frame carries its rows with it, so the count is exact.
        IR::DataFrameScan { .. } => {
            let rows = (leaf_row_count(ir).value()? as f64).max(MIN_CARDINALITY);
            Some(NodeStats::of_rows(rows))
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
            let inner = node_stats(*input, ir_arena, expr_arena)?;
            let names: Vec<&PlSmallStr> = keys.iter().map(|k| k.output_name()).collect();
            Some(one_row_per_group(inner, &names, options.slice))
        },

        // `unique` keeps one row per distinct combination of its subset, which is a
        // group-by over those columns. No subset means every column.
        IR::Distinct { input, options } => {
            let input_schema;
            let names: Vec<&PlSmallStr> = match &options.subset {
                Some(subset) => subset.iter().collect(),
                None => {
                    input_schema = ir_arena.get(*input).schema(ir_arena).into_owned();
                    input_schema.iter_names().collect()
                },
            };
            let inner = node_stats(*input, ir_arena, expr_arena)?;
            Some(one_row_per_group(inner, &names, options.slice))
        },

        // A filter above the scan is equivalent to one pushed into it; predicate
        // pushdown normally leaves none here, but a leaf need not be a bare scan.
        IR::Filter { input, predicate } => {
            let inner = node_stats(*input, ir_arena, expr_arena)?;
            let filtered = apply_predicate(
                inner.filtered,
                inner.unfiltered,
                predicate.node(),
                expr_arena,
                inner.columns.as_deref(),
            );
            Some(inner.selecting(filtered))
        },

        // Row-preserving, so they pass both estimates through untouched.
        IR::SimpleProjection { input, .. } | IR::Cache { input, .. } => {
            node_stats(*input, ir_arena, expr_arena)
        },

        // A sort keeps every row it is given, and carries its own slice for a top-k.
        IR::Sort { input, slice, .. } => {
            let inner = node_stats(*input, ir_arena, expr_arena)?;
            Some(match slice {
                None => inner,
                // The dynamic predicate reaches the same rows sooner, it does not
                // change which ones they are.
                Some((offset, len, _)) => {
                    let rows = sliced(inner.filtered, (*offset, *len));
                    inner.selecting(rows)
                },
            })
        },

        // A slice narrows the relation the way a filter does.
        IR::Slice { input, offset, len } => {
            let inner = node_stats(*input, ir_arena, expr_arena)?;
            let rows = sliced(inner.filtered, (*offset, *len as usize));
            Some(inner.selecting(rows))
        },

        // A union stacks its inputs, so it holds every row and every key of all of
        // them. Every input has to be measurable for the sum to mean anything.
        IR::Union { inputs, options } => {
            let mut acc = NodeStats::default();
            for input in inputs {
                let inner = node_stats(*input, ir_arena, expr_arena)?;
                acc.filtered += inner.filtered;
                acc.unfiltered += inner.unfiltered;
            }
            acc.filtered = acc.filtered.max(MIN_CARDINALITY);
            acc.unfiltered = acc.unfiltered.max(MIN_CARDINALITY);
            Some(match options.slice {
                None => acc,
                Some(slice) => {
                    let rows = sliced(acc.filtered, slice);
                    acc.selecting(rows)
                },
            })
        },

        // A gather emits one row per index, and its indices are a frame of their own,
        // so its height is that frame's.
        IR::Gather { input, idxs, .. } => {
            let idxs = node_stats(*idxs, ir_arena, expr_arena)?;
            let inner = node_stats(*input, ir_arena, expr_arena)?;
            Some(NodeStats {
                // A gather may repeat a row, so it can be taller than its input, but
                // it reaches no key the input did not already hold.
                filtered: idxs.filtered,
                unfiltered: inner.unfiltered,
                columns: inner.columns,
            })
        },

        // A `select` keeps its input's height when every expression does. Aggregates
        // collapse the frame to one row, and a mix of the two broadcasts the scalars
        // back up to the height of the rest.
        IR::Select { input, expr, .. } => {
            if expr.is_empty() || !expr.iter().all(|e| keeps_height(e, expr_arena)) {
                return None;
            }
            let inner = node_stats(*input, ir_arena, expr_arena)?;
            if expr.iter().all(|e| e.is_scalar(expr_arena)) {
                return Some(NodeStats::of_rows(MIN_CARDINALITY));
            }
            let columns = passed_through_columns(&inner, expr, expr_arena);
            Some(NodeStats { columns, ..inner })
        },

        // `with_columns` adds to the frame it is given rather than replacing it, so
        // the height is the input's even when every expression is a scalar.
        IR::HStack { input, exprs, .. } => {
            let inner = node_stats(*input, ir_arena, expr_arena)?;
            let columns = shadowed_columns(&inner, exprs, expr_arena);
            Some(NodeStats { columns, ..inner })
        },

        // Anything else (python scan, opaque function, unpivot, ...) is not
        // modelled. Do not guess.
        _ => None,
    }
}

impl NodeStats {
    /// The same leaf with fewer of its rows selected. The key domain is untouched:
    /// the other side of a join still references all of it.
    fn selecting(mut self, filtered: f64) -> Self {
        self.filtered = filtered;
        self
    }

    /// Estimates for a relation of `rows` rows carrying no column statistics.
    pub fn of_rows(rows: f64) -> Self {
        Self {
            filtered: rows,
            unfiltered: rows,
            columns: None,
        }
    }

    /// Statistics for one output column.
    pub fn column(&self, name: &str) -> Option<&ScanColumnStats> {
        self.columns.as_ref()?.get(name)
    }

    /// Distinct values in `name`, when known well enough to steer a decision.
    ///
    /// Never more than the rows the node emits.
    fn ndv(&self, name: &str) -> Option<f64> {
        let distinct = self.column(name)?.distinct.confident(MAX_NDV_REL_ERR)?;
        Some((distinct as f64).clamp(MIN_CARDINALITY, self.unfiltered))
    }

    /// Distinct values in `key`, when it is a plain column with a known NDV.
    fn key_ndv(&self, key: &ExprIR) -> Option<f64> {
        self.ndv(key.output_name_inner().get()?)
    }

    /// Distinct combinations of `keys`, or `None` unless every one is known.
    ///
    /// The product assumes the keys are independent, which is an upper bound; the
    /// caller caps it at the row count.
    fn key_ndv_product(&self, keys: &[&PlSmallStr]) -> Option<f64> {
        keys.iter()
            .map(|name| self.ndv(name))
            .try_fold(1.0, |acc, ndv| Some(acc * ndv?))
    }
}

/// Column statistics of a scan, keyed on its output names.
///
/// A column mapping renames between the file and the IR schema. Resolving it needs
/// the physical-id lookup that lives in the multi-scan reader, which this crate
/// cannot reach, so a mapped scan reports nothing.
fn scan_columns(ir: &IR) -> Option<Arc<ScanColumnStatsMap>> {
    let IR::Scan {
        file_info,
        unified_scan_args,
        ..
    } = ir
    else {
        return None;
    };
    unified_scan_args.column_mapping.is_none().then_some(())?;
    file_info.stats.columns.clone()
}

/// Statistics for the outputs of `exprs` that are a column of the input under its
/// own name or an alias of it.
///
/// Anything computed describes values the input statistics say nothing about.
fn passed_through_columns(
    inner: &NodeStats,
    exprs: &[ExprIR],
    expr_arena: &Arena<AExpr>,
) -> Option<Arc<ScanColumnStatsMap>> {
    let mut kept = ScanColumnStatsMap::default();
    for e in exprs {
        let source = into_column(e.node(), expr_arena)?;
        if let Some(stats) = inner.column(source) {
            kept.insert(e.output_name().clone(), stats.clone());
        }
    }
    (!kept.is_empty()).then(|| Arc::new(kept))
}

/// The input's statistics, less every column `exprs` overwrites with a computed
/// one. `with_columns` keeps the rest of the frame as it was.
fn shadowed_columns(
    inner: &NodeStats,
    exprs: &[ExprIR],
    expr_arena: &Arena<AExpr>,
) -> Option<Arc<ScanColumnStatsMap>> {
    let columns = inner.columns.as_ref()?;
    let overwrites = |e: &ExprIR| {
        into_column(e.node(), expr_arena) != Some(e.output_name())
            && columns.contains_key(e.output_name())
    };
    if !exprs.iter().any(overwrites) {
        return Some(Arc::clone(columns));
    }
    let kept: ScanColumnStatsMap = columns
        .iter()
        .filter(|(name, _)| !exprs.iter().any(|e| e.output_name() == *name && overwrites(e)))
        .map(|(name, stats)| (name.clone(), stats.clone()))
        .collect();
    (!kept.is_empty()).then(|| Arc::new(kept))
}

/// Whether an expression leaves the frame's height alone, either by producing one
/// value per row or by producing a scalar that broadcasts back to it.
///
/// Anything else — an explode, a range, a slice — sets the height from something
/// other than the input, and is not modelled.
fn keeps_height(expr: &ExprIR, expr_arena: &Arena<AExpr>) -> bool {
    expr.is_length_preserving(expr_arena) || expr.is_scalar(expr_arena)
}

/// Estimates for a node emitting one row per distinct combination of `keys`,
/// optionally sliced.
///
/// The output columns are the keys, each holding as many distinct values as the
/// node has rows.
fn one_row_per_group(
    inner: NodeStats,
    keys: &[&PlSmallStr],
    slice: Option<(i64, usize)>,
) -> NodeStats {
    let ndv = inner.key_ndv_product(keys);
    let mut groups = NodeStats {
        filtered: n_groups(inner.filtered, keys.len(), ndv),
        unfiltered: n_groups(inner.unfiltered, keys.len(), ndv),
        columns: None,
    };
    if let Some(slice) = slice {
        // Grouping is what set the key domain here, so the slice narrows that too.
        groups.filtered = sliced(groups.filtered, slice);
        groups.unfiltered = sliced(groups.unfiltered, slice);
    }
    groups.columns = single_key_column(keys, groups.filtered);
    groups
}

/// A group-by over one key emits that key's distinct values, so its output holds
/// exactly one of each. Several keys say nothing about any one of them.
fn single_key_column(keys: &[&PlSmallStr], rows: f64) -> Option<Arc<ScanColumnStatsMap>> {
    let [name] = keys else { return None };
    let mut map = ScanColumnStatsMap::default();
    map.insert(
        (*name).clone(),
        ScanColumnStats {
            distinct: Card::approx(rows as u64),
            null_count: Card::Unknown,
            avg_byte_width: None,
        },
    );
    Some(Arc::new(map))
}

/// Rows left after applying `slice` to a relation of `rows` rows.
fn sliced(rows: f64, slice: impl Into<Slice>) -> f64 {
    let bounded = slice.into().restrict_to_bounds(rows as usize);
    (bounded.len() as f64).max(MIN_CARDINALITY)
}

/// Estimated distinct combinations of `n_keys` grouping keys over `rows` rows.
///
/// No distinct counts are available, so this interpolates between the two ends that
/// are known: no keys is a single group, and any number of keys is at most one group
/// per row. Each key added moves the estimate closer to the input.
fn n_groups(rows: f64, n_keys: usize, ndv: Option<f64>) -> f64 {
    if n_keys == 0 {
        return MIN_CARDINALITY;
    }
    if let Some(ndv) = ndv {
        return ndv.clamp(MIN_CARDINALITY, rows);
    }
    let exponent = n_keys as f64 / (n_keys as f64 + 1.0);
    rows.powf(exponent).clamp(MIN_CARDINALITY, rows)
}

/// Rows left after `predicate`, estimated one `AND` conjunct at a time.
///
/// `rows` is what the conjuncts narrow. `unfiltered` is the relation the column
/// statistics describe, which a slice may already have cut into.
fn apply_predicate(
    rows: f64,
    unfiltered: f64,
    predicate: Node,
    expr_arena: &Arena<AExpr>,
    columns: Option<&ScanColumnStatsMap>,
) -> f64 {
    let mut selectivity = 1.0;
    for conjunct in MintermIter::new(predicate, expr_arena) {
        selectivity *= conjunct_selectivity(conjunct, expr_arena, columns, unfiltered)
            .unwrap_or(DEFAULT_SELECTIVITY);
    }
    (rows * selectivity).max(MIN_CARDINALITY)
}

/// Fraction of rows one conjunct keeps, or `None` when nothing describes it.
fn conjunct_selectivity(
    conjunct: Node,
    expr_arena: &Arena<AExpr>,
    columns: Option<&ScanColumnStatsMap>,
    rows: f64,
) -> Option<f64> {
    let AExpr::Function {
        input,
        function: IRFunctionExpr::Boolean(function),
        ..
    } = expr_arena.get(conjunct)
    else {
        return None;
    };
    let [arg] = input.as_slice() else {
        return None;
    };
    let name = into_column(arg.node(), expr_arena)?;

    let nulls = columns?.get(name)?.null_count.confident(0.0)? as f64;
    let null_fraction = (nulls / rows.max(MIN_CARDINALITY)).clamp(0.0, 1.0);
    match function {
        IRBooleanFunction::IsNull => Some(null_fraction),
        IRBooleanFunction::IsNotNull => Some(1.0 - null_fraction),
        _ => None,
    }
}

/// Estimate the rows produced by joining two relations of the given sizes.
///
/// `key_domain_product` is the product, over every equi-key pair bridging the two
/// sides, of that key's domain size (see [`key_domain`]).
///
/// This is `|A| * |B| / DISTINCT(key)`, extended over multiple keys. With exact distinct
/// counts the divisor would be the `max` of the two sides. Only row counts are
/// available, which bound distinct counts from above and are tight only on the unique
/// side, so [`key_domain`] takes the `min` instead: the smaller relation is assumed
/// to hold the key uniquely.
pub fn join_cardinality(left: f64, right: f64, key_domain_product: f64) -> f64 {
    // Every factor comes from `key_domain`, which floors at `MIN_CARDINALITY`, so
    // the product is always >= 1.
    (left * right / key_domain_product).max(MIN_CARDINALITY)
}

/// Domain size of the key joining two leaves.
///
/// With distinct counts for both sides the domain is the larger of the two: every
/// value one side holds is in the domain, whether or not the other side has it.
/// Without them, row counts bound the domain from above and are tight only on the
/// unique side, so the smaller relation is assumed to hold the key uniquely.
pub fn key_domain(
    left: &NodeStats,
    left_key: &ExprIR,
    right: &NodeStats,
    right_key: &ExprIR,
) -> f64 {
    let domain = match (left.key_ndv(left_key), right.key_ndv(right_key)) {
        (Some(l), Some(r)) => l.max(r),
        _ => left.unfiltered.min(right.unfiltered),
    };
    domain.max(MIN_CARDINALITY)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(unfiltered: f64, filtered: f64) -> NodeStats {
        NodeStats {
            filtered,
            unfiltered,
            columns: None,
        }
    }

    impl NodeStats {
        /// The same leaf, with statistics for one of its columns.
        fn with_column(mut self, name: &str, stats: ScanColumnStats) -> Self {
            let mut map = self.columns.as_deref().cloned().unwrap_or_default();
            map.insert(PlSmallStr::from_str(name), stats);
            self.columns = Some(Arc::new(map));
            self
        }
    }

    /// A leaf whose join key has a known distinct count.
    fn leaf_with_ndv(unfiltered: f64, filtered: f64, key: &str, ndv: u64) -> NodeStats {
        leaf(unfiltered, filtered).with_column(
            key,
            ScanColumnStats {
                distinct: Card::Exact(ndv),
                ..Default::default()
            },
        )
    }

    fn key(name: &str) -> ExprIR {
        ExprIR::new(
            Node::default(),
            crate::plans::OutputName::ColumnLhs(PlSmallStr::from_str(name)),
        )
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
            key_domain(&inventory, &key("k"), &date_dim, &key("k")),
        );
        let with_item = join_cardinality(
            inventory.filtered,
            item.filtered,
            key_domain(&inventory, &key("k"), &item, &key("k")),
        );
        let with_warehouse = join_cardinality(
            inventory.filtered,
            warehouse.filtered,
            key_domain(&inventory, &key("k"), &warehouse, &key("k")),
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

        let out = join_cardinality(acc, item.filtered, key_domain(&inventory, &key("k"), &item, &key("k")));
        assert!((out - acc).abs() < 1.0, "got {out}");
    }

    /// The two ends of the group-count estimate are known exactly; only what lies
    /// between them is assumed.
    #[test]
    fn group_count_is_bounded_by_one_and_the_input() {
        // No key at all is a global aggregate.
        assert_eq!(n_groups(1_000_000.0, 0, None), 1.0);
        // One group per row is the most a group-by can emit.
        assert!(n_groups(1000.0, 8, None) <= 1000.0);
        assert_eq!(n_groups(1.0, 3, None), 1.0);
        // Adding a key can only move the estimate towards the input.
        let by_keys: Vec<f64> = (1..6).map(|k| n_groups(1_000_000.0, k, None)).collect();
        assert!(by_keys.windows(2).all(|w| w[0] < w[1]), "{by_keys:?}");
        assert!(by_keys.iter().all(|&g| g < 1_000_000.0), "{by_keys:?}");
    }

    #[test]
    fn slicing_counts_from_either_end_and_never_reaches_zero() {
        // Plain head.
        assert_eq!(sliced(100.0, (0i64, 10)), 10.0);
        // A slice longer than what is left of the relation.
        assert_eq!(sliced(100.0, (95i64, 10)), 5.0);
        // A negative offset counts back from the end.
        assert_eq!(sliced(100.0, (-5i64, 10)), 5.0);
        // One reaching past the start keeps only the part that overlaps.
        assert_eq!(sliced(100.0, (-105i64, 10)), 5.0);
        // Nothing left over floors, rather than producing an estimate that would
        // dominate every ordering decision.
        assert_eq!(sliced(100.0, (-200i64, 10)), MIN_CARDINALITY);
        assert_eq!(sliced(100.0, (200i64, 10)), MIN_CARDINALITY);
        assert_eq!(sliced(100.0, (0i64, 0)), MIN_CARDINALITY);
    }

    /// A slice narrows the relation but not the key domain the other side joins
    /// against, so it carries its selectivity over the way a filter does.
    #[test]
    fn a_sliced_dimension_shrinks_the_fact_table() {
        let inventory = leaf(11_745_000.0, 11_745_000.0);
        let date_dim = leaf(73_049.0, 73_049.0);
        let rows = sliced(date_dim.filtered, (0i64, 10));
        let head = date_dim.selecting(rows);

        let out = join_cardinality(
            inventory.filtered,
            head.filtered,
            key_domain(&inventory, &key("k"), &head, &key("k")),
        );
        // 11.7M * 10 / 73049. Dividing by the ten rows left would reproduce the
        // fact table whole.
        assert!((out - 1608.0).abs() < 5.0, "got {out}");
        assert!(out < inventory.filtered);
    }

    /// With distinct counts on both sides the domain is the larger of them, not the
    /// smaller relation.
    #[test]
    fn key_domain_takes_the_larger_distinct_count() {
        let fact = leaf_with_ndv(1_000_000.0, 1_000_000.0, "k", 500);
        let dim = leaf_with_ndv(800.0, 800.0, "k", 800);

        assert_eq!(key_domain(&fact, &key("k"), &dim, &key("k")), 800.0);
        // Without them the smaller relation is assumed to hold the key uniquely.
        let opaque = leaf(800.0, 800.0);
        assert_eq!(key_domain(&fact, &key("k"), &opaque, &key("k")), 800.0);
    }

    /// A distinct count that is only a rough bound must not steer the domain.
    #[test]
    fn a_low_confidence_distinct_count_is_ignored() {
        let dim = leaf(800.0, 800.0).with_column(
            "k",
            ScanColumnStats {
                distinct: Card::Approx {
                    value: 4,
                    rel_err: 1.0,
                },
                ..Default::default()
            },
        );

        let fact = leaf_with_ndv(1_000_000.0, 1_000_000.0, "k", 500);
        // The rough count of 4 would otherwise dominate as the max.
        assert_eq!(key_domain(&fact, &key("k"), &dim, &key("k")), 800.0);
    }

    /// A known distinct count replaces the interpolation between one group and one
    /// group per row.
    #[test]
    fn group_count_uses_a_known_distinct_count() {
        assert_eq!(n_groups(1_000_000.0, 1, Some(12.0)), 12.0);
        // Never more groups than rows, however the keys combine.
        assert_eq!(n_groups(10.0, 2, Some(400.0)), 10.0);
        // The interpolation still covers an unknown key.
        assert!(n_groups(1_000_000.0, 1, None) > 12.0);
    }

    #[test]
    fn null_counts_drive_is_null_selectivity() {
        let stats = leaf(1000.0, 1000.0).with_column(
            "a",
            ScanColumnStats {
                null_count: Card::Exact(250),
                ..Default::default()
            },
        );
        let map = stats.columns.as_deref().unwrap();

        let mut expr_arena = Arena::new();
        let col = expr_arena.add(AExpr::Column(PlSmallStr::from_str("a")));
        let is_null = expr_arena.add(AExpr::Function {
            input: vec![ExprIR::from_node(col, &expr_arena)],
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsNull),
            options: crate::prelude::FunctionOptions::elementwise(),
        });

        // A quarter of the rows are null, against the flat 0.2 fallback.
        let kept = apply_predicate(1000.0, 1000.0, is_null, &expr_arena, Some(map));
        assert!((kept - 250.0).abs() < 1e-9, "got {kept}");
        // An unknown column falls back.
        let kept = apply_predicate(1000.0, 1000.0, is_null, &expr_arena, None);
        assert!((kept - 200.0).abs() < 1e-9, "got {kept}");
    }

    #[test]
    fn selectivity_compounds_per_conjunct_and_never_reaches_zero() {
        // Two opaque conjuncts, so both fall back to the flat selectivity.
        let mut expr_arena = Arena::new();
        let a = expr_arena.add(AExpr::Column(PlSmallStr::from_str("a")));
        let b = expr_arena.add(AExpr::Column(PlSmallStr::from_str("b")));
        let and = expr_arena.add(AExpr::BinaryExpr {
            left: a,
            op: crate::plans::Operator::And,
            right: b,
        });

        let one = apply_predicate(1000.0, 1000.0, a, &expr_arena, None);
        assert!((one - 200.0).abs() < 1e-9, "got {one}");
        let two = apply_predicate(1000.0, 1000.0, and, &expr_arena, None);
        assert!((two - 40.0).abs() < 1e-9, "got {two}");
        // However many conjuncts pile up, an estimate never reaches zero.
        assert_eq!(apply_predicate(1.0, 1.0, and, &expr_arena, None), MIN_CARDINALITY);
    }
}
