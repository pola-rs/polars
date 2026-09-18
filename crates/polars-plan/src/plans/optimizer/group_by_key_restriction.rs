//! Keeping only the groups a join will use, before they are formed.
//!
//! `S ⋈ GroupBy(R, K)` equals `S ⋈ GroupBy(R ⋉ S, K)` when every join key on the
//! grouped side is a grouping column: the join drops the groups whose keys have no
//! match in `S`, and the semi join drops exactly the rows of those groups. The other
//! groups are formed from the same rows as before, so every aggregate is unchanged.
//! `S` is computed once and shared through a cache.

use std::sync::Arc;

use polars_core::prelude::{PlIndexMap, PlIndexSet};
use polars_defs::join::JoinBuildSide;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::unique_id::UniqueId;

use super::predicate_pushdown::utils::map_column_references;
use crate::plans::iterator::ArenaExprIter;
use crate::plans::stats::{StatsCache, node_stats_with_cache};
use crate::plans::{
    AExpr, ExprIR, ExprPushdownGroup, IR, JoinOptionsIR, JoinTypeOptionsIR, MintermIter,
    aexpr_to_leaf_names_iter, is_inherently_nondeterministic,
};
use crate::prelude::{JoinArgs, JoinType};

/// Largest share of the grouped rows the semi join may be estimated to keep.
const MAX_KEPT_FRACTION: f64 = 0.5;
/// The semi join sends the grouped rows on unfiltered once its build side holds more
/// than this share of their estimate.
const MAX_BUILD_SHARE: f64 = 1.0 / 8.0;

/// Rewrite every inner join over a group-by throughout the plan, in place.
pub(super) fn restrict_grouped_join_inputs(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) {
    let mut stats = StatsCache::default();
    let mut seen = PlIndexSet::default();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        stack.extend(ir_arena.get(node).get_inputs());
        if matches!(ir_arena.get(node), IR::Join { .. }) {
            restrict(node, ir_arena, expr_arena, &mut stats);
        }
    }
}

/// Elementwise, fallible or not: an expression whose result for a row does not
/// depend on the other rows.
fn elementwise(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    let mut group = ExprPushdownGroup::Pushable;
    group.update_with_expr_rec(expr_arena.get(node), expr_arena, None);
    !group.blocks_pushdown(false)
}

/// Elementwise and infallible: an expression that may see fewer rows. A user
/// function is opaque, so it is taken to be able to raise.
fn pushable(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    let mut group = ExprPushdownGroup::Pushable;
    group.update_with_expr_rec(expr_arena.get(node), expr_arena, None);
    matches!(group, ExprPushdownGroup::Pushable)
        && !expr_arena
            .iter(node)
            .any(|(_, ae)| matches!(ae, AExpr::AnonymousFunction { .. }))
}

fn all_conjuncts_pushable(predicate: &ExprIR, expr_arena: &Arena<AExpr>) -> bool {
    MintermIter::new(predicate.node(), expr_arena).all(|node| pushable(node, expr_arena))
}

/// Whether `node` is in the subtree under `root`.
fn depends_on(root: Node, node: Node, ir_arena: &Arena<IR>) -> bool {
    let mut seen = PlIndexSet::default();
    let mut stack = vec![root];
    while let Some(current) = stack.pop() {
        if current == node {
            return true;
        }
        if seen.insert(current) {
            stack.extend(ir_arena.get(current).get_inputs());
        }
    }
    false
}

/// The grouped side of an inner join that passed every check but cost.
struct Grouped {
    group_by: Node,
    /// The nodes between the join and the group-by, top down.
    chain: Vec<Node>,
    /// The join keys of this side, as columns of the group-by input.
    keys: Vec<ExprIR>,
}

/// Rewrite `keys` to read the columns that `exprs` name them after. Every key must
/// be an elementwise expression over columns that `exprs` pass through, renamed or
/// not.
fn keys_through(
    keys: &mut [ExprIR],
    mut sources: PlIndexMap<PlSmallStr, Option<PlSmallStr>>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<()> {
    let names: PlIndexSet<PlSmallStr> = keys
        .iter()
        .flat_map(|key| aexpr_to_leaf_names_iter(key.node(), expr_arena).cloned())
        .collect();
    let mut renames = PlIndexMap::default();
    for name in names {
        let source = sources.swap_remove(&name)??;
        if source != name {
            renames.insert(name, source);
        }
    }
    for key in keys {
        map_column_references(key, expr_arena, &renames);
    }
    Some(())
}

/// The output name and, for a plain column, the column each expression reads.
fn column_sources(
    exprs: &[ExprIR],
    expr_arena: &Arena<AExpr>,
) -> PlIndexMap<PlSmallStr, Option<PlSmallStr>> {
    exprs
        .iter()
        .map(|e| {
            let source = match expr_arena.get(e.node()) {
                AExpr::Column(column) => Some(column.clone()),
                _ => None,
            };
            (e.output_name().clone(), source)
        })
        .collect()
}

/// Match `T(GroupBy(R, K))` at `side`, where `T` is a chain of projections and
/// filters that are elementwise and infallible, and map `join_keys` through `T` and
/// the grouping keys to columns of `R`.
fn grouped_side(
    side: Node,
    join_keys: &[ExprIR],
    ir_arena: &Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<Grouped> {
    // The keys move below the filters in `T`, so they must not raise on the rows
    // those filters drop.
    if !join_keys.iter().all(|key| pushable(key.node(), expr_arena)) {
        return None;
    }
    let mut keys = join_keys.to_vec();
    let mut chain = Vec::new();
    let mut below = side;
    let group_by = loop {
        let ir = ir_arena.get(below);
        match ir {
            IR::SimpleProjection { .. } => {},
            IR::Filter { predicate, .. } if all_conjuncts_pushable(predicate, expr_arena) => {},
            IR::Select { expr, .. } if expr.iter().all(|e| pushable(e.node(), expr_arena)) => {
                keys_through(&mut keys, column_sources(expr, expr_arena), expr_arena)?;
            },
            IR::HStack { input, exprs, .. }
                if exprs.iter().all(|e| pushable(e.node(), expr_arena)) =>
            {
                let mut sources = column_sources(exprs, expr_arena);
                for name in ir_arena.get(*input).schema(ir_arena).iter_names() {
                    sources
                        .entry(name.clone())
                        .or_insert_with(|| Some(name.clone()));
                }
                keys_through(&mut keys, sources, expr_arena)?;
            },
            IR::GroupBy { .. } => break below,
            _ => return None,
        }
        chain.push(below);
        below = ir.get_inputs()[0];
    };
    let IR::GroupBy {
        input,
        keys: group_keys,
        maintain_order,
        options,
        apply,
        ..
    } = ir_arena.get(group_by)
    else {
        unreachable!()
    };
    // The same cases predicate pushdown leaves alone, an order of groups that a
    // semi join below would not keep, and grouping keys that change with the rows
    // it drops.
    if apply.is_some()
        || options.is_rolling()
        || options.is_dynamic()
        || options.slice.is_some()
        || *maintain_order
        || group_keys.is_empty()
        || !group_keys
            .iter()
            .all(|key| elementwise(key.node(), expr_arena))
    {
        return None;
    }
    let input_schema = ir_arena.get(*input).schema(ir_arena);
    let mut sources = column_sources(group_keys, expr_arena);
    for source in sources.values_mut() {
        *source = source.take().filter(|column| input_schema.contains(column));
    }
    keys_through(&mut keys, sources, expr_arena)?;
    Some(Grouped {
        group_by,
        chain,
        keys,
    })
}

/// Rewrite the inner join at `join` if one side is a group-by whose input the other
/// side is estimated to narrow enough.
fn restrict(
    join: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    stats: &mut StatsCache,
) {
    let IR::Join {
        input_left,
        input_right,
        options,
        ..
    } = ir_arena.get(join)
    else {
        return;
    };
    let JoinTypeOptionsIR::Equi { on, .. } = &options.options else {
        return;
    };
    // Validation must see every group, including those the semi join would drop,
    // and both joins must compute the same keys.
    if !matches!(options.args.how, JoinType::Inner)
        || options.args.slice.is_some()
        || options.args.validation.needs_checks()
        || on.is_empty()
        || on.iter().any(|(left, right)| {
            is_inherently_nondeterministic(left.node(), expr_arena)
                || is_inherently_nondeterministic(right.node(), expr_arena)
        })
    {
        return;
    }
    let (input_left, input_right, options) = (*input_left, *input_right, options.clone());
    let (left_keys, right_keys): (Vec<ExprIR>, Vec<ExprIR>) = on.iter().cloned().unzip();

    // The grouped side and the other, with the other side's keys.
    let sides = [
        (true, input_right, input_left, &right_keys, &left_keys),
        (false, input_left, input_right, &left_keys, &right_keys),
    ];
    for (grouped_is_right, grouped, s, grouped_keys, s_keys) in sides {
        let Some(matched) = grouped_side(grouped, grouped_keys, ir_arena, expr_arena) else {
            continue;
        };
        let IR::GroupBy { input: r, .. } = ir_arena.get(matched.group_by) else {
            unreachable!()
        };
        let r = *r;
        // Already restricted by this join, or the other side is built from these
        // groups.
        let restricted = matches!(
            ir_arena.get(r),
            IR::Join { input_right, options, .. }
                if *input_right == s && matches!(options.args.how, JoinType::Semi)
        );
        if restricted || depends_on(s, matched.group_by, ir_arena) {
            continue;
        }
        let semi_ir = |s: Node, pass_through_above: Option<usize>, ir_arena: &Arena<IR>| IR::Join {
            input_left: r,
            input_right: s,
            schema: ir_arena.get(r).schema(ir_arena).into_owned(),
            options: Arc::new(JoinOptionsIR {
                allow_parallel: options.allow_parallel,
                force_parallel: options.force_parallel,
                args: JoinArgs {
                    how: JoinType::Semi,
                    nulls_equal: options.args.nulls_equal,
                    build_side: Some(JoinBuildSide::ForceRight),
                    ..JoinArgs::default()
                },
                options: JoinTypeOptionsIR::Equi {
                    on: matched
                        .keys
                        .iter()
                        .cloned()
                        .zip(s_keys.iter().cloned())
                        .collect(),
                    fused_predicate: None,
                },
                runtime_filters: Vec::new(),
                pass_through_above,
            }),
        };

        let semi = ir_arena.add(semi_ir(s, None, ir_arena));
        let before = node_stats_with_cache(r, ir_arena, expr_arena, stats);
        let after = node_stats_with_cache(semi, ir_arena, expr_arena, stats);
        let s_stats = node_stats_with_cache(s, ir_arena, expr_arena, stats);
        // The candidate was the last node added; its id will be reused.
        ir_arena.pop();
        stats.remove(&semi);
        let (Some(before), Some(after), Some(s_stats)) = (before, after, s_stats) else {
            continue;
        };
        if polars_core::config::verbose() {
            eprintln!(
                "group-by key restriction: estimated {:.0} grouped rows, {:.0} kept, {:.0} key rows",
                before.filtered, after.filtered, s_stats.filtered
            );
        }
        // The semi join must drop most of the grouped rows, and its build side must
        // be smaller than they are. The estimate of the build side is checked again
        // against its actual rows when the join runs.
        if after.filtered > before.filtered * MAX_KEPT_FRACTION
            || s_stats.filtered > before.filtered
        {
            continue;
        }
        let pass_through_above = Some((before.filtered * MAX_BUILD_SHARE) as usize);

        let shared = match ir_arena.get(s) {
            IR::Cache { .. } => s,
            _ => ir_arena.add(IR::Cache {
                input: s,
                id: UniqueId::new(),
            }),
        };
        let semi = ir_arena.add(semi_ir(shared, pass_through_above, ir_arena));
        let mut root = semi;
        for &step in std::iter::once(&matched.group_by).chain(matched.chain.iter().rev()) {
            let mut ir = ir_arena.get(step).clone();
            for slot in ir.inputs_mut() {
                *slot = root;
            }
            root = ir_arena.add(ir);
        }
        let (input_left, input_right) = if grouped_is_right {
            (shared, root)
        } else {
            (root, shared)
        };
        ir_arena.replace_with(join, |ir| {
            let IR::Join { schema, .. } = ir else {
                unreachable!()
            };
            IR::Join {
                input_left,
                input_right,
                schema,
                options,
            }
        });
        return;
    }
}
