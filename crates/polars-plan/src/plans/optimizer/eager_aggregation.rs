//! Aggregating the right side of a join by its join keys before the join.
//!
//! `GroupBy(L ⋈ R, keys from L, aggs over R)` equals
//! `GroupBy(L ⋈ GroupBy(R, R's join keys, partial aggs), keys, final aggs)` when every
//! aggregate splits into a partial one per join key and a final one that combines them.
//! Each L row then meets one pre-aggregated row per key instead of every R row of that key.
//!
//! Only `count`, `len`, `sum`, `min` and `max` of R columns, or of plain arithmetic over them,
//! split here. A left join gives an unmatched L row nulls for R's columns; the split is only
//! right if the aggregate sees those nulls as nulls, which is why the input may not turn a
//! null into a value. The partial aggregates also run on R rows the join would have dropped,
//! so they must not be able to fail where the original did not.

use std::sync::Arc;

use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::unique_column_name;

use super::join_utils::unconstrained;
use crate::plans::schema::join_right_output_names;
use crate::plans::stats::{NodeStats, StatsCache, composite_key_domain, node_stats_with_cache};
use crate::plans::{
    AExpr, AExprBuilder, ExprIR, IR, IRAggExpr, IRBuilder, IRFunctionExpr, JoinTypeOptionsIR,
    LiteralValue, OutputName, ToFieldContext, aexpr_to_leaf_names_iter,
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

/// One aggregate the rewrite splits, and its input in R's column names (none for `len`).
#[derive(Clone, Copy)]
struct Leaf {
    kind: LeafKind,
    input: Option<Node>,
}

impl Leaf {
    fn is_equal_to(&self, other: &Leaf, expr_arena: &Arena<AExpr>) -> bool {
        self.kind == other.kind
            && match (self.input, other.input) {
                (None, None) => true,
                (Some(a), Some(b)) => expr_arena
                    .get(a)
                    .is_expr_equal_to(expr_arena.get(b), expr_arena),
                _ => false,
            }
    }
}

/// A node on the way from the group by down to the join the rewrite targets.
enum Step {
    /// A `SimpleProjection`, or a `Select` of plain columns.
    Projection(Node),
    /// An inner join above the target. The path continues into one of its inputs.
    Join { node: Node, path_is_left: bool },
}

impl Step {
    fn node(&self) -> Node {
        match self {
            Step::Projection(node) | Step::Join { node, .. } => *node,
        }
    }
}

/// A join above the target, as the gate sees it.
struct Ancestor {
    /// The input the path continues into, and its keys.
    path: Node,
    path_keys: Vec<PlSmallStr>,
    /// The other input, and its keys.
    other: Node,
    other_keys: Vec<PlSmallStr>,
}

/// Columns the group by and the joins above the target read, named as at the node the
/// walk has reached.
struct Names {
    /// Group keys.
    keys: Vec<PlSmallStr>,
    /// Columns the aggregates read: name seen by the group by, and name here.
    leaf_columns: Vec<(PlSmallStr, PlSmallStr)>,
    /// Keys of the joins above.
    ancestor_keys: Vec<PlSmallStr>,
}

impl Names {
    fn all(&self) -> impl Iterator<Item = &PlSmallStr> {
        self.keys
            .iter()
            .chain(self.leaf_columns.iter().map(|(_, name)| name))
            .chain(self.ancestor_keys.iter())
    }

    /// Follows a `Select` of plain columns: output name -> input column.
    fn through_select(&mut self, renamed: &PlIndexMap<PlSmallStr, PlSmallStr>) -> Option<()> {
        let names = self
            .keys
            .iter_mut()
            .chain(self.leaf_columns.iter_mut().map(|(_, name)| name))
            .chain(self.ancestor_keys.iter_mut());
        for name in names {
            *name = renamed.get(name)?.clone();
        }
        Some(())
    }
}

/// The join the partial aggregation goes under.
struct Target {
    /// L: the input the group keys come from. R: the input that is aggregated.
    left: Node,
    right: Node,
    left_keys: Vec<PlSmallStr>,
    right_keys: Vec<PlSmallStr>,
    /// Whether R is the join's left input.
    right_is_left: bool,
    /// Name seen by the group by -> R column.
    from_right: PlIndexMap<PlSmallStr, PlSmallStr>,
}

fn try_push(node: Node, ir_arena: &mut Arena<IR>, expr_arena: &mut Arena<AExpr>) -> Option<Node> {
    let IR::GroupBy {
        input,
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

    let mut cache = StatsCache::new();
    let (target, join, steps) = find_target(*input, keys, aggs, ir_arena, expr_arena, &mut cache)?;

    let right_schema = ir_arena.get(target.right).schema(ir_arena).into_owned();
    let mut leaves = Vec::new();
    for agg in aggs {
        if !collect_leaves(
            agg.node(),
            expr_arena,
            &target.from_right,
            &right_schema,
            &mut leaves,
        ) {
            return None;
        }
    }
    if leaves.is_empty() {
        return None;
    }

    // A partial count over an R key without a match has no counterpart in the original, so
    // it must not be able to fail. `count` raises above `IdxSize::MAX`, which R's row bound
    // rules out; without a bound (R is a join), counts are summed as u64 instead.
    let wide_counts = node_stats_with_cache(target.right, ir_arena, expr_arena, &mut cache)
        .and_then(|stats| stats.max_rows())
        .is_none_or(|max_rows| max_rows > IdxSize::MAX as f64);

    let (keys, aggs, output_schema) = (keys.clone(), aggs.clone(), output_schema.clone());
    let IR::Join {
        options: join_options,
        ..
    } = ir_arena.get(join)
    else {
        unreachable!()
    };
    let join_options = join_options.clone();

    // Partial aggregates over R, one per distinct leaf.
    let mut partials: Vec<(Leaf, PlSmallStr)> = Vec::new();
    let mut partial_of = Vec::with_capacity(leaves.len());
    for (_, leaf) in &leaves {
        let index = match partials
            .iter()
            .position(|(seen, _)| seen.is_equal_to(leaf, expr_arena))
        {
            Some(index) => index,
            None => {
                partials.push((*leaf, unique_column_name()));
                partials.len() - 1
            },
        };
        partial_of.push(index);
    }
    let partial = partial_group_by(&target, &partials, wide_counts, ir_arena, expr_arena)?;

    // Final aggregates over the join output: the same expressions with each leaf replaced
    // by the aggregate that combines its partials.
    let is_left = matches!(join_options.args.how, JoinType::Left);
    let mut final_leaves = PlIndexMap::default();
    for ((node, leaf), index) in leaves.iter().zip(partial_of) {
        let partial_column = expr_arena.add(AExpr::Column(partials[index].1.clone()));
        final_leaves.insert(
            *node,
            final_agg(leaf, partial_column, is_left, wide_counts, expr_arena),
        );
    }
    let final_aggs = aggs
        .iter()
        .map(|agg| {
            let node = rebuild(agg.node(), &final_leaves, expr_arena);
            ExprIR::new(node, OutputName::Alias(agg.output_name().clone()))
        })
        .collect();

    let new_join = if target.right_is_left {
        IRBuilder::new(partial, expr_arena, ir_arena).join(target.left, join_options)
    } else {
        IRBuilder::new(target.left, expr_arena, ir_arena).join(partial, join_options)
    }
    .node();
    let partial_names: Vec<PlSmallStr> = partials.into_iter().map(|(_, name)| name).collect();
    let new_input = rebuild_path(&steps, new_join, &partial_names, ir_arena, expr_arena)?;
    let new_group_by = IRBuilder::new(new_input, expr_arena, ir_arena)
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

/// Walks down from the group by's input to the first join that can take the partial
/// aggregation and passes the gate. Returns it, the join node, and the nodes above it.
fn find_target(
    input: Node,
    keys: &[ExprIR],
    aggs: &[ExprIR],
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    cache: &mut StatsCache,
) -> Option<(Target, Node, Vec<Step>)> {
    let mut names = Names {
        keys: keys
            .iter()
            .map(|key| key.plain_column(expr_arena).cloned())
            .collect::<Option<_>>()?,
        leaf_columns: aggs
            .iter()
            .flat_map(|agg| aexpr_to_leaf_names_iter(agg.node(), expr_arena))
            .collect::<PlIndexSet<_>>()
            .into_iter()
            .map(|name| (name.clone(), name.clone()))
            .collect(),
        ancestor_keys: Vec::new(),
    };
    let mut steps = Vec::new();
    let mut ancestors = Vec::new();
    let mut current = input;
    loop {
        match ir_arena.get(current) {
            IR::SimpleProjection { input, columns } => {
                if !names.all().all(|name| columns.contains(name)) {
                    return None;
                }
                steps.push(Step::Projection(current));
                current = *input;
            },
            IR::Select { input, expr, .. } => {
                let renamed = expr
                    .iter()
                    .map(|e| match expr_arena.get(e.node()) {
                        AExpr::Column(column) => Some((e.output_name().clone(), column.clone())),
                        _ => None,
                    })
                    .collect::<Option<PlIndexMap<_, _>>>()?;
                names.through_select(&renamed)?;
                steps.push(Step::Projection(current));
                current = *input;
            },
            IR::Join { .. } => {
                if let Some(target) = target_at(current, &names, ir_arena, expr_arena) {
                    let skip_gate = polars_config::config().eager_aggregation_skip_gate();
                    if skip_gate || gate_passes(&target, &ancestors, ir_arena, expr_arena, cache) {
                        return Some((target, current, steps));
                    }
                }
                let (ancestor, path_is_left) = ancestor_at(current, &names, ir_arena, expr_arena)?;
                below_ancestor(&mut names, &ancestor, ir_arena)?;
                steps.push(Step::Join {
                    node: current,
                    path_is_left,
                });
                current = ancestor.path;
                ancestors.push(ancestor);
            },
            _ => return None,
        }
    }
}

/// `GroupBy(Filter(R, keys not null), R's join keys, partials)`. R rows with a null key match
/// nothing.
fn partial_group_by(
    target: &Target,
    partials: &[(Leaf, PlSmallStr)],
    wide_counts: bool,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<Node> {
    let right_keys = &target.right_keys;
    let partial_aggs = partials
        .iter()
        .map(|(leaf, name)| {
            let node = partial_agg(leaf, &right_keys[0], wide_counts, expr_arena);
            ExprIR::new(node, OutputName::Alias(name.clone()))
        })
        .collect();

    let mut not_null = None;
    for key in right_keys {
        let is_not_null = AExprBuilder::col(key.clone(), expr_arena)
            .is_not_null(expr_arena)
            .node();
        not_null = Some(match not_null {
            None => is_not_null,
            Some(left) => expr_arena.add(AExpr::BinaryExpr {
                left,
                op: Operator::And,
                right: is_not_null,
            }),
        });
    }
    let filtered_right = ir_arena.add(IR::Filter {
        input: target.right,
        predicate: ExprIR::from_node(not_null?, expr_arena),
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
    Some(partial)
}

/// Rebuilds the nodes above the target join bottom-up over `new_join`. Projections keep the
/// columns that still exist and pass the partials through.
fn rebuild_path(
    steps: &[Step],
    new_join: Node,
    partial_names: &[PlSmallStr],
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<Node> {
    let mut new_node = new_join;
    for step in steps.iter().rev() {
        let input_schema = ir_arena.get(new_node).schema(ir_arena).into_owned();
        new_node = match (step, ir_arena.get(step.node())) {
            (Step::Projection(_), IR::SimpleProjection { columns, .. }) => {
                let names: Vec<_> = columns
                    .iter_names()
                    .filter(|name| input_schema.contains(name))
                    .chain(partial_names)
                    .cloned()
                    .collect();
                IRBuilder::new(new_node, expr_arena, ir_arena)
                    .project_simple(names)
                    .ok()?
                    .node()
            },
            (Step::Projection(_), IR::Select { expr, options, .. }) => {
                let options = *options;
                let mut exprs: Vec<ExprIR> = expr
                    .iter()
                    .filter(|e| match expr_arena.get(e.node()) {
                        AExpr::Column(column) => input_schema.contains(column),
                        _ => false,
                    })
                    .cloned()
                    .collect();
                exprs.extend(
                    partial_names
                        .iter()
                        .map(|name| ExprIR::from_column_name(name.clone(), expr_arena)),
                );
                IRBuilder::new(new_node, expr_arena, ir_arena)
                    .project(exprs, options)
                    .node()
            },
            (
                Step::Join { path_is_left, .. },
                IR::Join {
                    input_left,
                    input_right,
                    options,
                    ..
                },
            ) => {
                let options = options.clone();
                if *path_is_left {
                    let other = *input_right;
                    IRBuilder::new(new_node, expr_arena, ir_arena).join(other, options)
                } else {
                    let other = *input_left;
                    IRBuilder::new(other, expr_arena, ir_arena).join(new_node, options)
                }
                .node()
            },
            _ => unreachable!(),
        };
    }
    Some(new_node)
}

/// Plain-column keys of an equi join the rewrite may cross or target: no fused predicate,
/// nulls never equal, nothing that pins it to its inputs. A computed key would be applied
/// twice: once by the partial group by and again by the join, which keeps its key
/// expressions.
fn join_keys(
    join: Node,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> Option<(Vec<PlSmallStr>, Vec<PlSmallStr>)> {
    let IR::Join { options, .. } = ir_arena.get(join) else {
        return None;
    };
    if !unconstrained(&options.args)
        || options.args.nulls_equal
        || !options.runtime_filters.is_empty()
    {
        return None;
    }
    let JoinTypeOptionsIR::Equi {
        on,
        fused_predicate: None,
    } = &options.options
    else {
        return None;
    };
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
    (!right_keys.is_empty()).then_some((left_keys, right_keys))
}

fn shares_a_name(a: &Schema, b: &Schema) -> bool {
    a.iter_names().any(|name| b.contains(name))
}

/// The target at `join`, if the partial aggregation can go under it: every aggregated
/// column comes from one input R, and everything else read above comes from the other. The
/// partial aggregation keeps R's join keys, so they may also be group keys.
fn target_at(
    join: Node,
    names: &Names,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> Option<Target> {
    let (left_keys, right_keys) = join_keys(join, ir_arena, expr_arena)?;
    let IR::Join {
        input_left,
        input_right,
        options,
        ..
    } = ir_arena.get(join)
    else {
        return None;
    };
    let how = &options.args.how;
    let left_schema = ir_arena.get(*input_left).schema(ir_arena);
    let right_schema = ir_arena.get(*input_right).schema(ir_arena);
    // A join above that filters on a key of this join could have pruned R before.
    let on_a_key = |name: &PlSmallStr| left_keys.contains(name) || right_keys.contains(name);
    if names.ancestor_keys.iter().any(on_a_key) {
        return None;
    }

    // R on the right. The join may suffix R's names; L's names stay as they are.
    if matches!(how, JoinType::Inner | JoinType::Left) {
        let output_names = join_right_output_names(&left_schema, &right_schema, options).ok()?;
        let right_by_output: PlIndexMap<PlSmallStr, PlSmallStr> = output_names
            .into_iter()
            .zip(right_schema.iter_names())
            .filter_map(|(output, name)| Some((output?, name.clone())))
            .collect();
        let from_right = names
            .leaf_columns
            .iter()
            .map(|(seen, name)| Some((seen.clone(), right_by_output.get(name)?.clone())))
            .collect::<Option<PlIndexMap<_, _>>>();
        let from_left = |name: &PlSmallStr| left_schema.contains(name);
        let right_join_key = |name: &PlSmallStr| {
            right_by_output
                .get(name)
                .is_some_and(|column| right_keys.contains(column))
        };
        if let Some(from_right) = from_right
            && names
                .keys
                .iter()
                .all(|key| from_left(key) || right_join_key(key))
            && names.ancestor_keys.iter().all(from_left)
        {
            return Some(Target {
                left: *input_left,
                right: *input_right,
                left_keys,
                right_keys,
                right_is_left: false,
                from_right,
            });
        }
    }

    // R on the left of an inner join. With a shared name the join would suffix L's names.
    if matches!(how, JoinType::Inner) && !shares_a_name(&left_schema, &right_schema) {
        let from_right = names
            .leaf_columns
            .iter()
            .map(|(seen, name)| {
                left_schema
                    .contains(name)
                    .then(|| (seen.clone(), name.clone()))
            })
            .collect::<Option<PlIndexMap<_, _>>>()?;
        let from_left = |name: &PlSmallStr| right_schema.contains(name);
        if names
            .keys
            .iter()
            .all(|key| from_left(key) || left_keys.contains(key))
            && names.ancestor_keys.iter().all(from_left)
        {
            return Some(Target {
                left: *input_right,
                right: *input_left,
                left_keys: right_keys,
                right_keys: left_keys,
                right_is_left: true,
                from_right,
            });
        }
    }
    None
}

/// `join` as a join above the target: an inner join whose aggregated columns all come from
/// one input, which is where the path continues.
fn ancestor_at(
    join: Node,
    names: &Names,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> Option<(Ancestor, bool)> {
    let (left_keys, right_keys) = join_keys(join, ir_arena, expr_arena)?;
    let IR::Join {
        input_left,
        input_right,
        options,
        ..
    } = ir_arena.get(join)
    else {
        return None;
    };
    // Only aggregated columns say which input the path continues into; `len()` reads none.
    if !matches!(options.args.how, JoinType::Inner) || names.leaf_columns.is_empty() {
        return None;
    }
    let left_schema = ir_arena.get(*input_left).schema(ir_arena);
    let right_schema = ir_arena.get(*input_right).schema(ir_arena);
    // Without shared names the join renames nothing, so names pass through unchanged.
    if shares_a_name(&left_schema, &right_schema) {
        return None;
    }
    let all_in = |schema: &Schema| {
        names
            .leaf_columns
            .iter()
            .all(|(_, name)| schema.contains(name))
    };
    let (path_is_left, ancestor) = if all_in(&left_schema) {
        (
            true,
            Ancestor {
                path: *input_left,
                path_keys: left_keys,
                other: *input_right,
                other_keys: right_keys,
            },
        )
    } else if all_in(&right_schema) {
        (
            false,
            Ancestor {
                path: *input_right,
                path_keys: right_keys,
                other: *input_left,
                other_keys: left_keys,
            },
        )
    } else {
        return None;
    };
    Some((ancestor, path_is_left))
}

/// Names below an ancestor join: what its other input provides is read above the target,
/// not from it.
fn below_ancestor(names: &mut Names, ancestor: &Ancestor, ir_arena: &Arena<IR>) -> Option<()> {
    let path_schema = ir_arena.get(ancestor.path).schema(ir_arena);
    let other_schema = ir_arena.get(ancestor.other).schema(ir_arena);
    for list in [&mut names.keys, &mut names.ancestor_keys] {
        list.retain(|name| !other_schema.contains(name));
        if !list.iter().all(|name| path_schema.contains(name)) {
            return None;
        }
    }
    names
        .ancestor_keys
        .extend(ancestor.path_keys.iter().cloned());
    Some(())
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
    /// `null_shares` holds the share of nulls in each key.
    fn new(stats: &NodeStats, keys: &[PlSmallStr], null_shares: &[f64]) -> Option<Self> {
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
        let not_null = null_shares.iter().map(|share| 1.0 - share).product::<f64>();
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
    /// independent of the key. A side that is a join can drop keys without a filter, so
    /// this is also capped by its rows: it holds no more keys than rows.
    fn key_survival(&self) -> f64 {
        let independent = 1.0 - (1.0 - self.kept()).powf(self.rows_per_key());
        independent.min(self.rows / self.ndv)
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

/// The node below `node` whose own column `name` is, through joins and projections.
fn key_origin(
    mut node: Node,
    name: &PlSmallStr,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> (Node, PlSmallStr) {
    let mut name = name.clone();
    loop {
        node = match ir_arena.get(node) {
            IR::Join {
                input_left,
                input_right,
                options,
                ..
            } => {
                // A join keeps the left names and may suffix the right ones.
                let left_schema = ir_arena.get(*input_left).schema(ir_arena);
                if left_schema.contains(&name) {
                    *input_left
                } else {
                    let right_schema = ir_arena.get(*input_right).schema(ir_arena);
                    let Ok(output_names) =
                        join_right_output_names(&left_schema, &right_schema, options)
                    else {
                        break;
                    };
                    let column = output_names
                        .into_iter()
                        .zip(right_schema.iter_names())
                        .find_map(|(output, column)| (output? == name).then(|| column.clone()));
                    match column {
                        Some(column) => {
                            name = column;
                            *input_right
                        },
                        None => break,
                    }
                }
            },
            IR::SimpleProjection { input, .. } => *input,
            IR::Select { input, expr, .. } => {
                let column = expr.iter().find_map(|e| match expr_arena.get(e.node()) {
                    AExpr::Column(column) if e.output_name() == &name => Some(column.clone()),
                    _ => None,
                });
                match column {
                    Some(column) => {
                        name = column;
                        *input
                    },
                    None => break,
                }
            },
            _ => break,
        };
    }
    (node, name)
}

/// Whether the rewrite is expected to pay off.
fn gate_passes(
    target: &Target,
    ancestors: &[Ancestor],
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    cache: &mut StatsCache,
) -> bool {
    let mut side = |node, keys: &[PlSmallStr]| {
        let stats = node_stats_with_cache(node, ir_arena, expr_arena, cache)?;
        // A column's statistics still describe the input it comes from, not a join's
        // output, so its nulls are counted against that input's rows.
        let null_shares = keys
            .iter()
            .map(|key| {
                let (origin, name) = key_origin(node, key, ir_arena, expr_arena);
                let origin = node_stats_with_cache(origin, ir_arena, expr_arena, cache)?;
                let nulls = origin
                    .column(&name)
                    .and_then(|c| c.null_count.confident(0.0))
                    .unwrap_or(0);
                Some((nulls as f64 / origin.unfiltered.max(1.0)).clamp(0.0, 1.0))
            })
            .collect::<Option<Vec<_>>>()?;
        SideStats::new(&stats, keys, &null_shares)
    };
    let (Some(left), Some(right)) = (
        side(target.left, &target.left_keys),
        side(target.right, &target.right_keys),
    ) else {
        if polars_config::config().verbose() {
            eprintln!("eager aggregation: no key statistics: skipped");
        }
        return false;
    };
    // Share of the path's rows each join above keeps: the matched share, pointed at the
    // path. Not output over input rows, which counts duplicate matches.
    let mut keeps = Vec::with_capacity(ancestors.len());
    for ancestor in ancestors {
        let (Some(path), Some(other)) = (
            side(ancestor.path, &ancestor.path_keys),
            side(ancestor.other, &ancestor.other_keys),
        ) else {
            if polars_config::config().verbose() {
                eprintln!("eager aggregation: no key statistics above the join: skipped");
            }
            return false;
        };
        keeps.push(key_overlap(&other, &path) * other.key_survival() * (1.0 - path.null_share));
    }

    let matched_share = key_overlap(&left, &right) * left.key_survival();
    let path_share = matched_share * keeps.iter().product::<f64>();
    let right_rows = right.rows * (1.0 - right.null_share);
    let right_keys_left = right.ndv * right.key_survival();
    let rows_per_key = right_rows / right_keys_left.max(1.0);
    let passes = path_share >= MIN_MATCHED_SHARE
        && rows_per_key >= MIN_ROWS_PER_KEY
        && right_rows >= MIN_RIGHT_ROWS;
    if polars_config::config().verbose() {
        eprintln!(
            "eager aggregation: matched share {matched_share:.3}, kept above {keeps:.3?}, \
             rows per key {rows_per_key:.2}, right rows {right_rows:.0}: {}",
            if passes { "fires" } else { "skipped" }
        );
    }
    passes
}

/// Checks that `node` is built from splittable aggregates of R columns, joined by
/// elementwise operations, and records each aggregate.
fn collect_leaves(
    node: Node,
    expr_arena: &mut Arena<AExpr>,
    from_right: &PlIndexMap<PlSmallStr, PlSmallStr>,
    right_schema: &Schema,
    leaves: &mut Vec<(Node, Leaf)>,
) -> bool {
    let leaf = match expr_arena.get(node) {
        AExpr::Len => Some((LeafKind::Len, None)),
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
            Some((kind, Some((input, allowed))))
        },
        _ => None,
    };
    if let Some((kind, input)) = leaf {
        let input = match input {
            None => None,
            Some((input, allowed)) => {
                match right_input(input, expr_arena, from_right, right_schema, allowed) {
                    Some(input) => Some(input),
                    None => return false,
                }
            },
        };
        leaves.push((node, Leaf { kind, input }));
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

/// The input of an aggregate, in R's column names: a plain R column with an `allowed` dtype,
/// or arithmetic over R columns (see `arithmetic_input`).
fn right_input(
    input: Node,
    expr_arena: &mut Arena<AExpr>,
    from_right: &PlIndexMap<PlSmallStr, PlSmallStr>,
    right_schema: &Schema,
    allowed: fn(&DataType) -> bool,
) -> Option<Node> {
    if let AExpr::Column(name) = expr_arena.get(input) {
        let column = from_right.get(name)?.clone();
        allowed(right_schema.get(&column)?).then_some(())?;
        return Some(expr_arena.add(AExpr::Column(column)));
    }

    let mut has_column = false;
    let input = arithmetic_input(input, expr_arena, from_right, &mut has_column)?;
    if !has_column {
        return None;
    }
    // Every node must be a primitive integer or float, including the input of a cast:
    // arithmetic on other types can raise (e.g. a Decimal division by zero).
    let ctx = ToFieldContext::new(expr_arena, right_schema);
    let mut stack = vec![input];
    while let Some(node) = stack.pop() {
        let ae = expr_arena.get(node);
        let dtype = ae.to_dtype(&ctx).ok()?.materialize_unknown(true).ok()?;
        if !(dtype.is_integer() || dtype.is_float()) {
            return None;
        }
        ae.inputs(&mut stack);
    }
    let dtype = expr_arena.get(input).to_dtype(&ctx).ok()?;
    allowed(&dtype).then_some(input)
}

/// A copy of `node` in R's column names, if it uses only R columns, scalar literals, `+`,
/// `-`, `*`, `/` and non-strict casts. These give null for a null input and cannot raise
/// on integers and floats, so running them on R rows the join would drop, or on a left
/// join's null-padded row, changes nothing.
fn arithmetic_input(
    node: Node,
    expr_arena: &mut Arena<AExpr>,
    from_right: &PlIndexMap<PlSmallStr, PlSmallStr>,
    has_column: &mut bool,
) -> Option<Node> {
    let ae = match expr_arena.get(node).clone() {
        AExpr::Column(name) => {
            *has_column = true;
            AExpr::Column(from_right.get(&name)?.clone())
        },
        AExpr::Literal(value) => return value.is_scalar().then_some(node),
        AExpr::BinaryExpr { left, op, right }
            if matches!(
                op,
                Operator::Plus | Operator::Minus | Operator::Multiply | Operator::TrueDivide
            ) =>
        {
            AExpr::BinaryExpr {
                left: arithmetic_input(left, expr_arena, from_right, has_column)?,
                op,
                right: arithmetic_input(right, expr_arena, from_right, has_column)?,
            }
        },
        AExpr::Cast {
            expr,
            dtype,
            options: CastOptions::NonStrict,
        } => AExpr::Cast {
            expr: arithmetic_input(expr, expr_arena, from_right, has_column)?,
            dtype,
            options: CastOptions::NonStrict,
        },
        _ => return None,
    };
    Some(expr_arena.add(ae))
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

/// With `wide_counts`, counts are summed as u64, which `SumCounts` then checks.
fn partial_agg(
    leaf: &Leaf,
    first_key: &PlSmallStr,
    wide_counts: bool,
    expr_arena: &mut Arena<AExpr>,
) -> Node {
    let leaf_input = || leaf.input.unwrap();
    let agg = match leaf.kind {
        LeafKind::Len if !wide_counts => return expr_arena.add(AExpr::Len),
        // R's keys are not null under the partial group by, so this counts its rows.
        LeafKind::Len => {
            let key = expr_arena.add(AExpr::Column(first_key.clone()));
            IRAggExpr::Sum(count_as_u64(key, expr_arena))
        },
        LeafKind::Count if !wide_counts => IRAggExpr::Count {
            input: leaf_input(),
            include_nulls: false,
        },
        LeafKind::Count => {
            let input = leaf_input();
            IRAggExpr::Sum(count_as_u64(input, expr_arena))
        },
        LeafKind::Sum => IRAggExpr::Sum(leaf_input()),
        LeafKind::Min { propagate_nans } => IRAggExpr::Min {
            input: leaf_input(),
            propagate_nans,
        },
        LeafKind::Max { propagate_nans } => IRAggExpr::Max {
            input: leaf_input(),
            propagate_nans,
        },
    };
    expr_arena.add(AExpr::Agg(agg))
}

/// `input.is_not_null().cast(UInt64)`: one for each value that `count` counts.
fn count_as_u64(input: Node, expr_arena: &mut Arena<AExpr>) -> Node {
    AExprBuilder::new_from_node(input)
        .is_not_null(expr_arena)
        .cast(DataType::UInt64, expr_arena)
        .node()
}

fn final_agg(
    leaf: &Leaf,
    partial: Node,
    is_left: bool,
    wide_counts: bool,
    expr_arena: &mut Arena<AExpr>,
) -> Node {
    let agg = match leaf.kind {
        LeafKind::Count => IRAggExpr::SumCounts(partial),
        // An unmatched L row in a left join is one row of the original group.
        LeafKind::Len if is_left => {
            let one = if wide_counts {
                Scalar::from(1u64)
            } else {
                Scalar::new_idxsize(1)
            };
            let one = expr_arena.add(AExpr::Literal(LiteralValue::Scalar(one)));
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
