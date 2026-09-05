//! Cardinality estimates for a node of the IR.
//!
//! Anything that cannot be estimated yields `None` rather than a guess, so a caller
//! can tell "no estimate" apart from "estimated to be small".

use std::sync::Arc;

use polars_utils::aliases::InitHashMaps;
#[expect(clippy::disallowed_types)] // We don't iterate over it.
use polars_utils::aliases::PlHashMap;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::slice_enum::Slice;
use recursive::recursive;

use super::{Card, DEFAULT_REL_ERR, ScanColumnStats, ScanColumnStatsMap, leaf_row_count};
use crate::plans::{
    AExpr, ExprIR, IR, IRBooleanFunction, IRFunctionExpr, JoinTypeOptionsIR, MintermIter,
    into_column,
};
use crate::prelude::{JoinType, Operator};

// We don't iterate over it.
#[expect(clippy::disallowed_types)]
pub(super) type StatsCache = PlHashMap<Node, Option<NodeStats>>;

/// Fallback selectivity for a filter conjunct with no better estimate.
const DEFAULT_SELECTIVITY: f64 = 0.2;

/// Largest relative error an NDV may carry and still steer a decision.
const MAX_NDV_REL_ERR: f32 = DEFAULT_REL_ERR;

const MIN_CARDINALITY: f64 = 1.0;

/// Cardinality estimates for the rows a node emits.
#[derive(Debug, Clone, Default)]
pub struct NodeStats {
    /// Estimated rows, after any filter pushed into the leaf.
    pub filtered: f64,
    /// Estimated rows, before those filters.
    /// We want to know this as we divide a join by the number
    /// of distinct values in the original set.
    pub unfiltered: f64,
    /// Rows the node cannot emit more than. A guarantee, not an estimate.
    max_rows: Option<f64>,
    /// Per output column, sparse. An absent column is unknown.
    columns: Option<Arc<ScanColumnStatsMap>>,
}

/// Estimate the rows a subplan produces.
///
/// `None` means the subtree is not modelled (a python scan, an opaque function, a
/// gather by a computed index).
pub fn node_stats(
    node: Node,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> Option<NodeStats> {
    node_stats_with_cache(node, ir_arena, expr_arena, &mut StatsCache::new())
}

/// [`node_stats`], reusing what `cache` already holds.
///
/// Every node is estimated from its inputs, so a caller asking about many nodes of
/// one subplan would otherwise re-walk the same descendants once per ancestor. The
/// cache is keyed on [`Node`] and is only valid while the arenas are unchanged.
#[recursive]
pub(super) fn node_stats_with_cache(
    node: Node,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    cache: &mut StatsCache,
) -> Option<NodeStats> {
    if let Some(hit) = cache.get(&node) {
        return hit.clone();
    }
    let ir = ir_arena.get(node);
    let stats = match ir {
        IR::Scan {
            predicate,
            unified_scan_args,
            ..
        } => {
            let rows = leaf_row_count(ir);
            let unfiltered = rows.value()? as f64;
            let mut max_rows = match rows {
                Card::Exact(rows) => Some(rows as f64),
                _ => None,
            };
            // The slice is applied before the predicate, so it narrows first.
            let mut filtered = unfiltered;
            if let Some(slice) = &unified_scan_args.pre_slice {
                filtered = sliced(filtered, slice.clone());
                max_rows = max_rows.map(|m| sliced(m, slice.clone()));
            }
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
                max_rows,
                columns,
            })
        },

        IR::DataFrameScan { .. } => {
            let rows = (leaf_row_count(ir).value()? as f64).max(MIN_CARDINALITY);
            Some(NodeStats::of_rows(rows))
        },
        IR::GroupBy {
            input,
            keys,
            options,
            apply,
            ..
        } => {
            // Those can produce more groups.
            if apply.is_some() || options.is_rolling() || options.is_dynamic() {
                return None;
            }
            let inner = node_stats_with_cache(*input, ir_arena, expr_arena, cache)?;
            let names: Vec<&PlSmallStr> = keys.iter().map(|k| k.output_name()).collect();
            Some(one_row_per_group(inner, &names, options.slice))
        },
        IR::Distinct { input, options } => {
            let input_schema;
            let names: Vec<&PlSmallStr> = match &options.subset {
                Some(subset) => subset.iter().collect(),
                None => {
                    input_schema = ir_arena.get(*input).schema(ir_arena).into_owned();
                    input_schema.iter_names().collect()
                },
            };
            let inner = node_stats_with_cache(*input, ir_arena, expr_arena, cache)?;
            Some(one_row_per_group(inner, &names, options.slice))
        },
        IR::Filter { input, predicate } => {
            let inner = node_stats_with_cache(*input, ir_arena, expr_arena, cache)?;
            let filtered = apply_predicate(
                inner.filtered,
                inner.unfiltered,
                predicate.node(),
                expr_arena,
                inner.columns.as_deref(),
            );
            Some(inner.filter(filtered))
        },
        IR::SimpleProjection { input, .. } | IR::Cache { input, .. } => {
            node_stats_with_cache(*input, ir_arena, expr_arena, cache)
        },
        IR::Sort { input, slice, .. } => {
            let inner = node_stats_with_cache(*input, ir_arena, expr_arena, cache)?;
            Some(match slice {
                None => inner,
                Some((offset, len, _)) => inner.slice((*offset, *len)),
            })
        },
        IR::Slice { input, offset, len } => {
            let inner = node_stats_with_cache(*input, ir_arena, expr_arena, cache)?;
            Some(inner.slice((*offset, *len as usize)))
        },
        IR::Union { inputs, options } => {
            let mut acc = NodeStats::of_rows(0.0);
            for input in inputs {
                let inner = node_stats_with_cache(*input, ir_arena, expr_arena, cache)?;
                acc.filtered += inner.filtered;
                acc.unfiltered += inner.unfiltered;
                acc.max_rows = acc.max_rows.zip(inner.max_rows).map(|(a, b)| a + b);
            }
            acc.filtered = acc.filtered.max(MIN_CARDINALITY);
            acc.unfiltered = acc.unfiltered.max(MIN_CARDINALITY);
            acc.max_rows = acc.max_rows.map(|m| m.max(MIN_CARDINALITY));
            Some(match options.slice {
                None => acc,
                Some(slice) => acc.slice(slice),
            })
        },
        // Both sides are divided by the shared domain of a key pair, see
        // `join_cardinality`.
        IR::Join {
            input_left,
            input_right,
            options,
            ..
        } => {
            // As-of, inequality and range matches are not modelled.
            let JoinTypeOptionsIR::Equi { on } = &options.options else {
                return None;
            };
            let left = node_stats_with_cache(*input_left, ir_arena, expr_arena, cache)?;
            let right = node_stats_with_cache(*input_right, ir_arena, expr_arena, cache)?;

            let key_domains = composite_key_domain(
                on.iter()
                    .map(|(left_key, right_key)| key_domain(&left, left_key, &right, right_key)),
                left.unfiltered.max(right.unfiltered),
            );
            let how = &options.args.how;
            let rows = |l: f64, r: f64| join_rows(how, l, r, join_cardinality(l, r, key_domains));

            let stats = NodeStats {
                filtered: rows(left.filtered, right.filtered)?,
                unfiltered: rows(left.unfiltered, right.unfiltered)?,
                max_rows: join_max_rows(how, &left, &right),
                columns: join_columns(&left, &right),
            };
            Some(match options.args.slice {
                None => stats,
                Some(slice) => stats.slice(slice),
            })
        },
        // A gather emits one row per index, and its indices are a frame of their own,
        // so its height is that frame's.
        IR::Gather { input, idxs, .. } => {
            let idxs = node_stats_with_cache(*idxs, ir_arena, expr_arena, cache)?;
            let inner = node_stats_with_cache(*input, ir_arena, expr_arena, cache)?;
            Some(NodeStats {
                // A gather may repeat a row, so it can be taller than its input, but
                // it reaches no key the input did not already hold.
                filtered: idxs.filtered,
                unfiltered: inner.unfiltered,
                max_rows: idxs.max_rows,
                columns: inner.columns,
            })
        },
        IR::Select { input, expr, .. } => {
            if expr.is_empty() || !expr.iter().all(|e| keeps_height(e, expr_arena)) {
                return None;
            }
            let inner = node_stats_with_cache(*input, ir_arena, expr_arena, cache)?;
            if expr.iter().all(|e| e.is_scalar(expr_arena)) {
                return Some(NodeStats::of_rows(MIN_CARDINALITY));
            }
            let columns = passed_through_columns(&inner, expr, expr_arena);
            Some(NodeStats { columns, ..inner })
        },
        IR::HStack { input, exprs, .. } => {
            let inner = node_stats_with_cache(*input, ir_arena, expr_arena, cache)?;
            let columns = shadowed_columns(&inner, exprs, expr_arena);
            Some(NodeStats { columns, ..inner })
        },
        _ => None,
    };
    cache.insert(node, stats.clone());
    stats
}

impl NodeStats {
    /// The same leaf with fewer of its rows selected. The key domain is untouched.
    fn filter(mut self, filtered: f64) -> Self {
        self.filtered = filtered;
        self
    }

    /// The same leaf narrowed by `slice`, applied to the estimate and the bound.
    fn slice(mut self, slice: impl Into<Slice> + Clone) -> Self {
        self.max_rows = self.max_rows.map(|m| sliced(m, slice.clone()));
        let filtered = sliced(self.filtered, slice);
        self.filter(filtered)
    }

    /// Estimates for a relation of exactly `rows` rows carrying no column
    /// statistics.
    pub fn of_rows(rows: f64) -> Self {
        Self {
            filtered: rows,
            unfiltered: rows,
            max_rows: Some(rows),
            columns: None,
        }
    }

    /// Rows the node is guaranteed not to exceed.
    pub fn max_rows(&self) -> Option<f64> {
        self.max_rows
    }

    /// Statistics for one output column.
    pub fn column(&self, name: &str) -> Option<&ScanColumnStats> {
        self.columns.as_ref()?.get(name)
    }

    /// Distinct values in `name`, when known well enough to steer a decision.
    ///
    /// Never more than the rows the node emits.
    fn distinct_count_key(&self, name: &str) -> Option<f64> {
        let distinct = self.column(name)?.distinct.confident(MAX_NDV_REL_ERR)?;
        Some((distinct as f64).clamp(MIN_CARDINALITY, self.unfiltered))
    }

    /// Distinct values in `key`, when it is a plain column with a known NDV.
    fn distinct_count(&self, key: &ExprIR) -> Option<f64> {
        self.distinct_count_key(key.output_name_inner().get()?)
    }

    /// Values `key` could hold, from its integer range. Estimates its distinct
    /// count from above, for a plain column of a scan that carried min/max.
    fn int_domain(&self, key: &ExprIR) -> Option<f64> {
        let domain = self.column(key.output_name_inner().get()?)?.int_domain()?;
        Some(domain.clamp(MIN_CARDINALITY, self.unfiltered))
    }

    /// Distinct combinations of `keys`, or `None` unless every one is known.
    ///
    /// The product assumes the keys are independent, which is an upper bound; the
    /// caller caps it at the row count.
    fn key_distinct_count_product(&self, keys: &[&PlSmallStr]) -> Option<f64> {
        keys.iter()
            .map(|name| self.distinct_count_key(name))
            .try_fold(1.0, |acc, ndv| Some(acc * ndv?))
    }
}

/// Rows a join of the given type emits, given the sizes of its sides and the rows an
/// inner join of them would emit. `None` for a join type that is not modelled.
fn join_rows(how: &JoinType, left: f64, right: f64, inner: f64) -> Option<f64> {
    let rows = match how {
        JoinType::Inner => inner,
        // An outer side keeps every row it has, matched or not.
        JoinType::Left => inner.max(left),
        JoinType::Right => inner.max(right),
        JoinType::Full => inner.max(left + right),
        // A semi-join emits the left rows that matched, an anti-join the rest.
        #[cfg(feature = "semi_anti_join")]
        JoinType::Semi => inner.min(left),
        #[cfg(feature = "semi_anti_join")]
        JoinType::Anti => left - inner.min(left),
        JoinType::Cross => left * right,
        // As-of, inequality and range matches are not modelled.
        _ => return None,
    };
    Some(rows.max(MIN_CARDINALITY))
}

/// A bound on the rows a join emits, for the types that have one. An equi-join can
/// repeat a row of either side once per match on the other, so most of them do not.
fn join_max_rows(how: &JoinType, left: &NodeStats, right: &NodeStats) -> Option<f64> {
    match how {
        // Every pair, and no more.
        JoinType::Cross => Some(left.max_rows? * right.max_rows?),
        // Both keep a subset of the left side.
        #[cfg(feature = "semi_anti_join")]
        JoinType::Semi | JoinType::Anti => left.max_rows,
        _ => None,
    }
}

/// Column statistics of a join output: those of both sides, minus any name they
/// share. A shared name is either coalesced or suffixed, and which side the output
/// column came from is no longer clear.
fn join_columns(left: &NodeStats, right: &NodeStats) -> Option<Arc<ScanColumnStatsMap>> {
    match (&left.columns, &right.columns) {
        (None, None) => None,
        (Some(columns), None) | (None, Some(columns)) => Some(columns.clone()),
        (Some(left), Some(right)) => {
            let mut merged = ScanColumnStatsMap::default();
            let mut take_from = |from: &ScanColumnStatsMap, other: &ScanColumnStatsMap| {
                for (name, stats) in from.iter().filter(|(name, _)| !other.contains_key(*name)) {
                    merged.insert(name.clone(), stats.clone());
                }
            };
            take_from(left, right);
            take_from(right, left);
            Some(Arc::new(merged))
        },
    }
}

/// Column statistics of a scan, keyed on its output names.
///
/// A scan with a column mapping reports nothing: resolving file names to output
/// names needs the physical-id lookup in the multi-scan reader.
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
/// own name or an alias of it. Anything computed holds values the input statistics
/// do not describe.
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
        .filter(|(name, _)| {
            !exprs
                .iter()
                .any(|e| e.output_name() == *name && overwrites(e))
        })
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
    let ndv = inner.key_distinct_count_product(keys);
    let mut groups = NodeStats {
        filtered: n_groups(inner.filtered, keys.len(), ndv),
        unfiltered: n_groups(inner.unfiltered, keys.len(), ndv),
        max_rows: inner.max_rows,
        columns: None,
    };
    if let Some(slice) = slice {
        // Grouping is what set the key domain here, so the slice narrows that too.
        groups.filtered = sliced(groups.filtered, slice);
        groups.unfiltered = sliced(groups.unfiltered, slice);
        groups.max_rows = groups.max_rows.map(|m| sliced(m, slice));
    }
    groups.columns = single_key_column(keys, groups.filtered);
    groups
}

/// A group-by over one key holds exactly one row per distinct value of it. Several
/// keys say nothing about any one of them.
fn single_key_column(keys: &[&PlSmallStr], rows: f64) -> Option<Arc<ScanColumnStatsMap>> {
    let [name] = keys else { return None };
    let mut map = ScanColumnStatsMap::default();
    map.insert(
        (*name).clone(),
        ScanColumnStats {
            distinct: Card::approx(rows as u64),
            null_count: Card::Unknown,
            avg_byte_width: None,
            int_range: None,
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
/// Without `ndv` this interpolates between the two ends that are known: no keys is a
/// single group, and any number of keys is at most one group per row. Each key added
/// moves the estimate closer to the input.
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
    let (name, function) = match expr_arena.get(conjunct) {
        AExpr::Function {
            input,
            function: IRFunctionExpr::Boolean(function),
            ..
        } => {
            let [arg] = input.as_slice() else {
                return None;
            };
            (into_column(arg.node(), expr_arena)?, function)
        },
        // Pushing an equi-join key into one of its sides leaves `x == x` behind. That
        // is a null check on the key, not the arbitrary comparison it looks like.
        AExpr::BinaryExpr {
            left,
            op: Operator::Eq,
            right,
        } => {
            let name = into_column(*left, expr_arena)?;
            if name != into_column(*right, expr_arena)? {
                return None;
            }
            (name, &IRBooleanFunction::IsNotNull)
        },
        _ => return None,
    };

    let Some(nulls) = columns
        .and_then(|columns| columns.get(name))
        .and_then(|stats| stats.null_count.confident(0.0))
    else {
        // Nothing describes the column. Most frames are mostly non-null; how many rows
        // hold a null is anyone's guess.
        return match function {
            IRBooleanFunction::IsNotNull => Some(1.0),
            _ => None,
        };
    };
    let null_fraction = (nulls as f64 / rows.max(MIN_CARDINALITY)).clamp(0.0, 1.0);
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
/// This is `|A| * |B| / DISTINCT(key)`, extended over multiple keys.
pub fn join_cardinality(left: f64, right: f64, key_domain_product: f64) -> f64 {
    // Every factor comes from `key_domain`, which floors at `MIN_CARDINALITY`, so
    // the product is always >= 1.
    (left * right / key_domain_product).max(MIN_CARDINALITY)
}

/// Domain size of a composite key, from the domains of the columns making it up.
///
/// The parts multiply, which holds only if they vary independently. `max_rows` bounds
/// the result, since neither side holds more distinct combinations than it has rows.
pub fn composite_key_domain(parts: impl Iterator<Item = f64>, max_rows: f64) -> f64 {
    parts.product::<f64>().min(max_rows).max(MIN_CARDINALITY)
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
    let domain = match (
        left.distinct_count(left_key),
        right.distinct_count(right_key),
    ) {
        (Some(l), Some(r)) => l.max(r),
        // Both the uniqueness assumption and an integer range estimate the domain
        // from above, so the tighter one is the better estimate.
        _ => {
            let rows = left.unfiltered.min(right.unfiltered);
            match key_int_domain(left, left_key, right, right_key) {
                Some(range) => rows.min(range),
                None => rows,
            }
        },
    };
    domain.max(MIN_CARDINALITY)
}

/// Domain implied by the integer ranges of the keys, from whichever sides have one.
fn key_int_domain(
    left: &NodeStats,
    left_key: &ExprIR,
    right: &NodeStats,
    right_key: &ExprIR,
) -> Option<f64> {
    match (left.int_domain(left_key), right.int_domain(right_key)) {
        (Some(l), Some(r)) => Some(l.max(r)),
        (l, r) => l.or(r),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(unfiltered: f64, filtered: f64) -> NodeStats {
        NodeStats {
            filtered,
            unfiltered,
            ..Default::default()
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

        let out = join_cardinality(
            acc,
            item.filtered,
            key_domain(&inventory, &key("k"), &item, &key("k")),
        );
        assert!((out - acc).abs() < 1.0, "got {out}");
    }

    /// Every join type is bounded by what its match semantics allow, whatever the
    /// inner-join estimate says.
    #[test]
    fn join_types_stay_within_their_bounds() {
        let (left, right) = (1000.0, 20.0);

        // A match that keeps few rows: the outer sides still keep all of theirs.
        let selective = 5.0;
        assert_eq!(
            join_rows(&JoinType::Inner, left, right, selective),
            Some(5.0)
        );
        assert_eq!(
            join_rows(&JoinType::Left, left, right, selective),
            Some(left)
        );
        assert_eq!(
            join_rows(&JoinType::Right, left, right, selective),
            Some(right)
        );
        assert_eq!(
            join_rows(&JoinType::Full, left, right, selective),
            Some(left + right)
        );

        // A match that fans out: an outer join is at least as large as the inner one.
        let fanning = 5000.0;
        assert_eq!(
            join_rows(&JoinType::Left, left, right, fanning),
            Some(fanning)
        );
        assert_eq!(
            join_rows(&JoinType::Cross, left, right, fanning),
            Some(20_000.0)
        );
    }

    /// A semi-join is a subset of its left side and an anti-join is the complement,
    /// so the two must add up to it however far off the inner estimate is.
    #[cfg(feature = "semi_anti_join")]
    #[test]
    fn semi_and_anti_partition_the_left_side() {
        let left = 1000.0;
        for inner in [0.0, 5.0, 400.0, 1000.0, 5000.0] {
            let semi = join_rows(&JoinType::Semi, left, 20.0, inner).unwrap();
            let anti = join_rows(&JoinType::Anti, left, 20.0, inner).unwrap();
            assert!(semi <= left && anti <= left, "{inner}: {semi} / {anti}");
            // Both floor at one row, so they only sum to the left side above that.
            assert!(semi + anti <= left + MIN_CARDINALITY, "{inner}");
        }
    }

    /// A join type we do not model must say so rather than fall back to a guess.
    #[cfg(feature = "iejoin")]
    #[test]
    fn unmodelled_join_types_have_no_estimate() {
        assert_eq!(join_rows(&JoinType::IEJoin, 1000.0, 20.0, 5.0), None);
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
        let head = date_dim.filter(rows);

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
        assert_eq!(
            apply_predicate(1.0, 1.0, and, &expr_arena, None),
            MIN_CARDINALITY
        );
    }
}
