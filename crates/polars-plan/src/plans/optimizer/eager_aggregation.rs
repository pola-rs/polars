//! Aggregating the right side of a join by its join keys before the join.
//!
//! `GroupBy(L ⋈ R, keys from L, aggs over R)` equals
//! `GroupBy(L ⋈ GroupBy(R, R's join keys, partial aggs), keys, final aggs)` when every
//! aggregate splits into a partial one per join key and a final one that combines them.
//! Each L row then meets one pre-aggregated row per key instead of every R row of that key.
//!
//! Only `count`, `len`, `sum`, `min` and `max` of plain R columns split here. A left join
//! gives an unmatched L row nulls for R's columns; the split is only right if the aggregate
//! sees those nulls as they are, which is why the aggregate input must be a plain column.
//! The partial aggregates also run on R rows the join would have dropped, so they must not
//! be able to fail where the original did not.

use std::sync::Arc;

use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::unique_column_name;

use super::join_utils::unconstrained;
use crate::plans::schema::join_right_output_names;
use crate::plans::stats::{
    NodeStats, StatsCache, composite_key_domain, node_stats_with_cache,
};
use crate::plans::{
    AExpr, ExprIR, IR, IRAggExpr, IRBooleanFunction, IRBuilder, IRFunctionExpr, JoinTypeOptionsIR,
    LiteralValue, OutputName, ProjectionOptions,
};
use crate::prelude::{GroupbyOptions, JoinType, Operator};

/// Rewrite throughout the plan, returning the new root.
pub(super) fn push_group_by_below_joins(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> Node {
    super::join_pushthrough::run_pass(root, ir_arena, expr_arena, |node, ir_arena, expr_arena| {
        try_push(node, ir_arena, expr_arena).unwrap_or(node)
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum LeafKind {
    Count,
    Len,
    Sum,
    Min { propagate_nans: bool },
    Max { propagate_nans: bool },
}

/// One aggregate the rewrite splits, and the R column it reads (none for `len`).
#[derive(Clone, PartialEq, Eq, Hash)]
struct Leaf {
    kind: LeafKind,
    column: Option<PlSmallStr>,
}

fn try_push(node: Node, ir_arena: &mut Arena<IR>, expr_arena: &mut Arena<AExpr>) -> Option<Node> {
    let IR::GroupBy {
        input: join,
        keys,
        aggs,
        schema: output_schema,
        maintain_order,
        options,
        apply,
    } = ir_arena.get(node)
    else {
        return None;
    };
    if *maintain_order
        || apply.is_some()
        || options.is_rolling()
        || options.is_dynamic()
        || options.slice.is_some()
    {
        return None;
    }
    // Projection pushdown often leaves a selection of columns between the two, which may
    // rename them (e.g. to a join suffix). Name seen by the group by -> join output name.
    let (join, renamed): (Node, Option<PlIndexMap<PlSmallStr, PlSmallStr>>) = match ir_arena
        .get(*join)
    {
        IR::SimpleProjection { input, columns } => (
            *input,
            Some(
                columns
                    .iter_names()
                    .map(|name| (name.clone(), name.clone()))
                    .collect(),
            ),
        ),
        IR::Select { input, expr, .. } => (
            *input,
            Some(
                expr.iter()
                    .map(|e| match expr_arena.get(e.node()) {
                        AExpr::Column(column) => Some((e.output_name().clone(), column.clone())),
                        _ => None,
                    })
                    .collect::<Option<_>>()?,
            ),
        ),
        _ => (*join, None),
    };
    let join_name = |name: &PlSmallStr| match &renamed {
        None => Some(name.clone()),
        Some(renamed) => renamed.get(name).cloned(),
    };
    let IR::Join {
        input_left: left,
        input_right: right,
        options: join_options,
        ..
    } = ir_arena.get(join)
    else {
        return None;
    };
    let (left, right) = (*left, *right);
    let how = join_options.args.how.clone();
    if !matches!(how, JoinType::Inner | JoinType::Left)
        || !unconstrained(&join_options.args)
        || join_options.args.nulls_equal
        || !join_options.runtime_filters.is_empty()
    {
        return None;
    }
    let JoinTypeOptionsIR::Equi {
        on,
        fused_predicate: None,
    } = &join_options.options
    else {
        return None;
    };
    // A computed key would be applied twice: once by the partial group by and again by the
    // join, which keeps its key expressions.
    let (left_keys, right_keys): (Vec<_>, Vec<_>) = on
        .iter()
        .map(|(l, r)| {
            Some((
                l.plain_column(expr_arena)?.clone(),
                r.plain_column(expr_arena)?.clone(),
            ))
        })
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .unzip();
    if right_keys.is_empty() {
        return None;
    }

    let left_schema = ir_arena.get(left).schema(ir_arena).into_owned();
    let right_schema = ir_arena.get(right).schema(ir_arena).into_owned();
    // Join output name -> R column, for the R columns the join keeps.
    let right_names = join_right_output_names(&left_schema, &right_schema, join_options).ok()?;
    let right_by_output: PlIndexMap<PlSmallStr, PlSmallStr> = right_names
        .into_iter()
        .zip(right_schema.iter_names())
        .filter_map(|(output, name)| Some((output?, name.clone())))
        .collect();
    // Name seen by the group by -> R column.
    let from_right: PlIndexMap<PlSmallStr, PlSmallStr> = match &renamed {
        None => right_by_output,
        Some(renamed) => renamed
            .iter()
            .filter_map(|(seen, output)| Some((seen.clone(), right_by_output.get(output)?.clone())))
            .collect(),
    };

    // Group key -> the L column it is.
    let mut key_columns = Vec::with_capacity(keys.len());
    for key in keys {
        let name = join_name(key.plain_column(expr_arena)?)?;
        if !left_schema.contains(&name) {
            return None;
        }
        key_columns.push((key.output_name().clone(), name));
    }

    let mut leaves = Vec::new();
    for agg in aggs {
        if !collect_leaves(
            agg.node(),
            expr_arena,
            &from_right,
            &right_schema,
            &mut leaves,
        ) {
            return None;
        }
    }
    if leaves.is_empty() {
        return None;
    }

    if !polars_config::config().eager_aggregation_skip_gate()
        && !gate_passes(left, right, &left_keys, &right_keys, ir_arena, expr_arena)
    {
        return None;
    }

    let (keys, aggs, output_schema) = (keys.clone(), aggs.clone(), output_schema.clone());
    let join_options = join_options.clone();

    // Partial aggregates over R, one per distinct leaf.
    let mut partial_names: PlIndexMap<Leaf, PlSmallStr> = PlIndexMap::default();
    for (_, leaf) in &leaves {
        if !partial_names.contains_key(leaf) {
            partial_names.insert(leaf.clone(), unique_column_name());
        }
    }
    let partial_aggs = partial_names
        .iter()
        .map(|(leaf, name)| {
            let node = partial_agg(leaf, &right_keys[0], expr_arena);
            ExprIR::new(node, OutputName::Alias(name.clone()))
        })
        .collect();

    // R rows with a null key match nothing.
    let mut not_null = None;
    for key in &right_keys {
        let column = ExprIR::from_column_name(key.clone(), expr_arena);
        let function = IRFunctionExpr::Boolean(IRBooleanFunction::IsNotNull);
        let is_not_null = expr_arena.add(AExpr::Function {
            input: vec![column],
            options: function.function_options(),
            function,
        });
        not_null = Some(match not_null {
            None => is_not_null,
            Some(left) => expr_arena.add(AExpr::BinaryExpr {
                left,
                op: Operator::And,
                right: is_not_null,
            }),
        });
    }
    let not_null = not_null?;
    let filtered_right = ir_arena.add(IR::Filter {
        input: right,
        predicate: ExprIR::from_node(not_null, expr_arena),
    });
    let partial_keys = right_keys
        .iter()
        .map(|key| ExprIR::from_column_name(key.clone(), expr_arena))
        .collect();
    let partial = IRBuilder::new(filtered_right, expr_arena, ir_arena)
        .group_by(
            partial_keys,
            partial_aggs,
            None,
            false,
            Arc::new(GroupbyOptions::default()),
        )
        .ok()?
        .node();

    // Final aggregates over the join output: the same expressions with each leaf replaced
    // by the aggregate that combines its partials.
    let is_left = matches!(how, JoinType::Left);
    let mut final_leaves = PlIndexMap::default();
    for (node, leaf) in &leaves {
        let partial_column = expr_arena.add(AExpr::Column(partial_names[leaf].clone()));
        final_leaves.insert(*node, final_agg(leaf, partial_column, is_left, expr_arena));
    }
    let final_aggs = aggs
        .iter()
        .map(|agg| {
            let node = rebuild(agg.node(), &final_leaves, expr_arena);
            ExprIR::new(node, OutputName::Alias(agg.output_name().clone()))
        })
        .collect();

    // The group by reads its keys by the names it saw before, and the partials.
    let mut columns = Vec::with_capacity(key_columns.len() + partial_names.len());
    let mut seen = PlIndexSet::default();
    for (seen_name, column) in &key_columns {
        if seen.insert(seen_name.clone()) {
            let node = expr_arena.add(AExpr::Column(column.clone()));
            columns.push(ExprIR::new(node, OutputName::Alias(seen_name.clone())));
        }
    }
    for name in partial_names.values() {
        columns.push(ExprIR::from_column_name(name.clone(), expr_arena));
    }
    let new_join = IRBuilder::new(left, expr_arena, ir_arena)
        .join(partial, join_options)
        .project(columns, ProjectionOptions::default())
        .node();
    let new_group_by = IRBuilder::new(new_join, expr_arena, ir_arena)
        .group_by(
            keys,
            final_aggs,
            None,
            false,
            Arc::new(GroupbyOptions::default()),
        )
        .ok()?
        .node();

    // Anything the rewrite got wrong about names or types shows up here.
    let new_schema = ir_arena.get(new_group_by).schema(ir_arena);
    if **new_schema != *output_schema {
        return None;
    }
    Some(new_group_by)
}

/// Fire only if at least this share of R's non-null rows is expected to find a match.
/// Below it, most of the pre-aggregation works on rows the join, or its runtime filters,
/// would have dropped.
const MIN_MATCHED_SHARE: f64 = 0.75;
/// Fire only if each partial group is expected to fold at least this many R rows.
const MIN_ROWS_PER_KEY: f64 = 1.5;
/// Leave small inputs alone.
const MIN_RIGHT_ROWS: f64 = 100_000.0;

/// What the gate needs to know about one side of the join and its keys.
struct SideStats {
    /// Rows after the side's own filters.
    rows: f64,
    /// Rows before them.
    unfiltered: f64,
    /// Distinct keys before the filters.
    ndv: f64,
    /// Share of rows with a null in any key.
    null_share: f64,
    /// Inclusive range of a single integer key.
    range: Option<(i128, i128)>,
}

impl SideStats {
    fn new(stats: &NodeStats, keys: &[PlSmallStr]) -> Option<Self> {
        let unfiltered = stats.unfiltered.max(1.0);
        let ndv = match keys {
            [key] => stats.key_distinct_estimate(key)?,
            _ => composite_key_domain(
                keys.iter()
                    .map(|key| stats.key_distinct_estimate(key))
                    .collect::<Option<Vec<_>>>()?
                    .into_iter(),
                unfiltered,
            ),
        }
        .max(1.0);
        let not_null = keys
            .iter()
            .map(|key| {
                let nulls = stats
                    .column(key)
                    .and_then(|c| c.null_count.confident(0.0))
                    .unwrap_or(0);
                1.0 - (nulls as f64 / unfiltered).clamp(0.0, 1.0)
            })
            .product::<f64>();
        let range = match keys {
            [key] => stats.column(key).and_then(|c| c.int_range),
            _ => None,
        };
        Some(Self {
            rows: stats.filtered,
            unfiltered,
            ndv,
            null_share: 1.0 - not_null,
            range,
        })
    }

    /// Share of rows the side's filters keep.
    fn kept(&self) -> f64 {
        (self.rows / self.unfiltered).clamp(0.0, 1.0)
    }

    /// Non-null rows per key before the filters.
    fn rows_per_key(&self) -> f64 {
        self.unfiltered * (1.0 - self.null_share) / self.ndv
    }

    /// Chance that a key keeps at least one row after the filters, taking the filter as
    /// independent of the key.
    fn key_survival(&self) -> f64 {
        1.0 - (1.0 - self.kept()).powf(self.rows_per_key())
    }
}

/// Share of R's keys that L also holds, both before filters. With integer ranges on both
/// sides the keys are taken as spread evenly over their range; otherwise the smaller key
/// set is taken to lie inside the larger.
fn key_overlap(left: &SideStats, right: &SideStats) -> f64 {
    let overlap = match (left.range, right.range) {
        (Some((l_min, l_max)), Some((r_min, r_max))) => {
            let length = |(min, max): (i128, i128)| (max - min + 1).max(1) as f64;
            let density_left = left.ndv / length((l_min, l_max));
            let density_right = right.ndv / length((r_min, r_max));
            let shared = (l_max.min(r_max) - l_min.max(r_min) + 1).max(0) as f64;
            density_left.min(density_right) * shared / right.ndv
        },
        _ => left.ndv / right.ndv,
    };
    overlap.clamp(0.0, 1.0)
}

/// Whether the rewrite is expected to pay off.
fn gate_passes(
    left: Node,
    right: Node,
    left_keys: &[PlSmallStr],
    right_keys: &[PlSmallStr],
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> bool {
    let mut cache = StatsCache::new();
    let mut side = |node, keys| {
        let stats = node_stats_with_cache(node, ir_arena, expr_arena, &mut cache)?;
        SideStats::new(&stats, keys)
    };
    let (Some(left), Some(right)) = (side(left, left_keys), side(right, right_keys)) else {
        if polars_config::config().verbose() {
            eprintln!("eager aggregation: no key statistics: skipped");
        }
        return false;
    };

    let matched_share = key_overlap(&left, &right) * left.key_survival();
    let right_rows = right.rows * (1.0 - right.null_share);
    let right_keys_left = right.ndv * right.key_survival();
    let rows_per_key = right_rows / right_keys_left.max(1.0);
    let passes = matched_share >= MIN_MATCHED_SHARE
        && rows_per_key >= MIN_ROWS_PER_KEY
        && right_rows >= MIN_RIGHT_ROWS;
    if polars_config::config().verbose() {
        eprintln!(
            "eager aggregation: matched share {matched_share:.3}, rows per key {rows_per_key:.2}, \
             right rows {right_rows:.0}: {}",
            if passes { "fires" } else { "skipped" }
        );
    }
    passes
}

/// Checks that `node` is built from splittable aggregates of R columns, joined by
/// elementwise operations, and records each aggregate.
fn collect_leaves(
    node: Node,
    expr_arena: &Arena<AExpr>,
    from_right: &PlIndexMap<PlSmallStr, PlSmallStr>,
    right_schema: &Schema,
    leaves: &mut Vec<(Node, Leaf)>,
) -> bool {
    let right_column = |input: Node, allowed: fn(&DataType) -> bool| {
        let AExpr::Column(name) = expr_arena.get(input) else {
            return None;
        };
        let column = from_right.get(name)?;
        allowed(right_schema.get(column)?).then(|| column.clone())
    };
    let leaf = match expr_arena.get(node) {
        AExpr::Len => Some(Leaf {
            kind: LeafKind::Len,
            column: None,
        }),
        AExpr::Agg(agg) => {
            let (kind, input, allowed): (_, _, fn(&DataType) -> bool) = match agg {
                IRAggExpr::Count {
                    input,
                    include_nulls: false,
                } => (LeafKind::Count, *input, |_| true),
                IRAggExpr::Sum(input) => (LeafKind::Sum, *input, sum_is_infallible),
                IRAggExpr::Min {
                    input,
                    propagate_nans,
                } => (
                    LeafKind::Min {
                        propagate_nans: *propagate_nans,
                    },
                    *input,
                    min_max_is_supported,
                ),
                IRAggExpr::Max {
                    input,
                    propagate_nans,
                } => (
                    LeafKind::Max {
                        propagate_nans: *propagate_nans,
                    },
                    *input,
                    min_max_is_supported,
                ),
                _ => return false,
            };
            let Some(column) = right_column(input, allowed) else {
                return false;
            };
            Some(Leaf {
                kind,
                column: Some(column),
            })
        },
        _ => None,
    };
    if let Some(leaf) = leaf {
        leaves.push((node, leaf));
        return true;
    }

    // Above the aggregates: one value per group in, one value out.
    let ae = expr_arena.get(node);
    let allowed = match ae {
        AExpr::Literal(value) => value.is_scalar(),
        AExpr::Cast { .. } | AExpr::BinaryExpr { .. } | AExpr::Ternary { .. } => true,
        AExpr::Function { options, .. } => options.is_elementwise(),
        _ => false,
    };
    if !allowed {
        return false;
    }
    let mut inputs = Vec::new();
    ae.inputs(&mut inputs);
    inputs
        .into_iter()
        .all(|input| collect_leaves(input, expr_arena, from_right, right_schema, leaves))
}

/// Integer sums wrap the same way in any grouping; a decimal sum raises on overflow.
fn sum_is_infallible(dtype: &DataType) -> bool {
    dtype.is_integer() || dtype.is_float()
}

fn min_max_is_supported(dtype: &DataType) -> bool {
    dtype.is_primitive_numeric()
        || matches!(
            dtype,
            DataType::Date | DataType::Datetime(_, _) | DataType::Duration(_) | DataType::Time
        )
}

/// A partial count over an R key without a match has no counterpart in the original, so it
/// must not be able to fail: counts are summed as u64, which `SumCounts` then checks.
fn partial_agg(leaf: &Leaf, first_key: &PlSmallStr, expr_arena: &mut Arena<AExpr>) -> Node {
    let mut column = || expr_arena.add(AExpr::Column(leaf.column.clone().unwrap()));
    let agg = match leaf.kind {
        // R's keys are not null under the partial group by, so this counts its rows.
        LeafKind::Len => {
            let key = expr_arena.add(AExpr::Column(first_key.clone()));
            IRAggExpr::Sum(count_as_u64(key, expr_arena))
        },
        LeafKind::Count => {
            let input = column();
            IRAggExpr::Sum(count_as_u64(input, expr_arena))
        },
        LeafKind::Sum => IRAggExpr::Sum(column()),
        LeafKind::Min { propagate_nans } => IRAggExpr::Min {
            input: column(),
            propagate_nans,
        },
        LeafKind::Max { propagate_nans } => IRAggExpr::Max {
            input: column(),
            propagate_nans,
        },
    };
    expr_arena.add(AExpr::Agg(agg))
}

/// `input.is_not_null().cast(UInt64)`: one for each value that `count` counts.
fn count_as_u64(input: Node, expr_arena: &mut Arena<AExpr>) -> Node {
    let function = IRFunctionExpr::Boolean(IRBooleanFunction::IsNotNull);
    let is_not_null = expr_arena.add(AExpr::Function {
        input: vec![ExprIR::from_node(input, expr_arena)],
        options: function.function_options(),
        function,
    });
    expr_arena.add(AExpr::Cast {
        expr: is_not_null,
        dtype: DataType::UInt64,
        options: CastOptions::Strict,
    })
}

fn final_agg(leaf: &Leaf, partial: Node, is_left: bool, expr_arena: &mut Arena<AExpr>) -> Node {
    let agg = match leaf.kind {
        LeafKind::Count => IRAggExpr::SumCounts(partial),
        // An unmatched L row in a left join is one row of the original group.
        LeafKind::Len if is_left => {
            let one = expr_arena.add(AExpr::Literal(LiteralValue::Scalar(Scalar::from(1u64))));
            let function = IRFunctionExpr::FillNull;
            let filled = expr_arena.add(AExpr::Function {
                input: vec![
                    ExprIR::from_node(partial, expr_arena),
                    ExprIR::from_node(one, expr_arena),
                ],
                options: function.function_options(),
                function,
            });
            IRAggExpr::SumCounts(filled)
        },
        LeafKind::Len => IRAggExpr::SumCounts(partial),
        LeafKind::Sum => IRAggExpr::Sum(partial),
        LeafKind::Min { propagate_nans } => IRAggExpr::Min {
            input: partial,
            propagate_nans,
        },
        LeafKind::Max { propagate_nans } => IRAggExpr::Max {
            input: partial,
            propagate_nans,
        },
    };
    expr_arena.add(AExpr::Agg(agg))
}

/// Copy of the expression at `node` with every leaf in `replaced` swapped out.
fn rebuild(node: Node, replaced: &PlIndexMap<Node, Node>, expr_arena: &mut Arena<AExpr>) -> Node {
    if let Some(&new) = replaced.get(&node) {
        return new;
    }
    let ae = expr_arena.get(node).clone();
    let mut inputs = Vec::new();
    ae.inputs(&mut inputs);
    if inputs.is_empty() {
        return node;
    }
    let inputs: Vec<Node> = inputs
        .into_iter()
        .map(|input| rebuild(input, replaced, expr_arena))
        .collect();
    expr_arena.add(ae.replace_inputs(&inputs))
}
