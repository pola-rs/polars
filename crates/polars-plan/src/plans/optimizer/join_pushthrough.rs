//! Moving inner joins below the preserved side of a left/semi/anti join, and turning
//! the `LEFT JOIN … WHERE right_key IS NULL` idiom into an anti join.
//!
//! `(A ⟕ B) ⋈ C` equals `(A ⋈ C) ⟕ B` when every key of the inner join comes from
//! `A`: the null-extended rows of the left join are exactly the `A` rows without a
//! match, whichever side of the inner join they meet.
//!
//! The second rewrite is an identity: the filter keeps only the unmatched rows, on
//! which every column from `B` is null.

use std::sync::Arc;

use polars_core::prelude::PlIndexMap;
use polars_core::schema::{Schema, SchemaRef};
use polars_utils::arena::{Arena, Node};
use polars_utils::idx_vec::UnitVec;
use polars_utils::pl_str::PlSmallStr;
use recursive::recursive;

use super::join_utils::{plain_inner_equi_join, unconstrained};
use crate::plans::iterator::ArenaExprIter;
use crate::plans::schema::det_join_schema;
use crate::plans::stats::{StatsCache, node_stats_with_cache};
use crate::plans::{
    AExpr, ExprIR, ExprPushdownGroup, IR, JoinOptionsIR, JoinTypeOptionsIR, MintermIter,
};
#[cfg(feature = "semi_anti_join")]
use crate::plans::{
    IRBooleanFunction, IRFunctionExpr, LiteralValue, OutputName, ProjectionOptions,
};
use crate::prelude::JoinType;
use crate::utils::check_input_node;

/// Rewrite throughout the plan, returning the new root.
pub(super) fn push_through_outer_joins(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> Node {
    // Two passes: the anti rewrite must not run first, or the inner join above sees
    // an `HStack` where it looks for the filter.
    let root = run_pass(root, ir_arena, expr_arena, push_inner_through);
    #[cfg(feature = "semi_anti_join")]
    let root = run_pass(root, ir_arena, expr_arena, filter_to_anti);
    root
}

type Rule = fn(Node, &mut Arena<IR>, &mut Arena<AExpr>) -> Node;

fn run_pass(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    rule: Rule,
) -> Node {
    // A cached subtree can be reached from several parents; rewriting it once per
    // parent would give caches that share an id but not a plan.
    let mut rewritten = PlIndexMap::default();
    rewrite(root, ir_arena, expr_arena, &mut rewritten, rule)
}

/// Bottom-up: rewrite the inputs, then apply `rule` at this node.
#[recursive]
fn rewrite(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    rewritten: &mut PlIndexMap<Node, Node>,
    rule: Rule,
) -> Node {
    if let Some(&done) = rewritten.get(&node) {
        return done;
    }
    // A filter over a join is one pattern; rewrite below the join so the rule sees
    // both together.
    let step = match ir_arena.get(node) {
        IR::Filter { input, .. } if matches!(ir_arena.get(*input), IR::Join { .. }) => *input,
        _ => node,
    };
    let children = ir_arena.get(step).get_inputs();
    let mut new_children = UnitVec::with_capacity(children.len());
    for child in children {
        new_children.push(rewrite(child, ir_arena, expr_arena, rewritten, rule));
    }
    for (slot, new) in ir_arena.get_mut(step).inputs_mut().zip(new_children) {
        *slot = new;
    }
    let new_node = rule(node, ir_arena, expr_arena);
    rewritten.insert(node, new_node);
    new_node
}

/// A left/semi/anti join an inner join may move below.
fn preserving_outer(options: &JoinOptionsIR) -> bool {
    let how = &options.args.how;
    (matches!(how, JoinType::Left) || how.is_semi_anti())
        // An inner join below reorders the preserved side, and `1:m` would check
        // `A ⋈ C` for uniqueness instead of `A`.
        && unconstrained(&options.args)
        && matches!(&options.options, JoinTypeOptionsIR::Equi { fused_predicate: None, .. })
}

/// Elementwise and infallible: the rows an expression sees change, so one that
/// could raise on a row must not move. A user function is opaque, so it is taken to
/// be able to.
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

fn same_columns(a: &Schema, b: &Schema) -> bool {
    a.len() == b.len() && a.iter().all(|(name, dtype)| b.get(name) == Some(dtype))
}

/// `Inner(F?(Outer(A, B)), C)` becomes `Restore(F?(Outer(Inner(A, C), B)))`.
///
/// A filter directly above the inner join comes along onto the innermost inner join
/// it can read from, so predicate fusion still sees it.
fn push_inner_through(node: Node, ir_arena: &mut Arena<IR>, expr_arena: &mut Arena<AExpr>) -> Node {
    let (join, filter) = match ir_arena.get(node) {
        IR::Filter { input, predicate } if matches!(ir_arena.get(*input), IR::Join { .. }) => {
            let movable = all_conjuncts_pushable(predicate, expr_arena);
            (*input, Some((predicate.clone(), movable)))
        },
        IR::Join { .. } => (node, None),
        _ => return node,
    };
    let carried = match &filter {
        Some((predicate, true)) => Some(predicate.clone()),
        _ => None,
    };

    let mut stats = StatsCache::default();
    let Some(pushed) = try_push(join, carried, ir_arena, expr_arena, &mut stats) else {
        return node;
    };
    // A filter that found no inner join to sit on would lose its fusion with this
    // one, which is worth more than the push.
    if pushed.unplaced_filter.is_some() {
        return node;
    }
    // A filter that could not move stays where it was.
    match filter {
        Some((predicate, false)) => ir_arena.add(IR::Filter {
            input: pushed.root,
            predicate,
        }),
        _ => pushed.root,
    }
}

struct PushResult {
    root: Node,
    /// The travelling filter, if no inner join could take it.
    unplaced_filter: Option<ExprIR>,
}

/// The parts of an `Inner(F?(Outer(A, B)), C)` that passed every check but cost.
struct Candidate {
    a: Node,
    b: Node,
    c: Node,
    outer: Node,
    /// The filter between the joins.
    between: Option<ExprIR>,
    inner_options: Arc<JoinOptionsIR>,
    outer_options: Arc<JoinOptionsIR>,
    inner_schema: SchemaRef,
    outer_schema: SchemaRef,
    output_schema: SchemaRef,
}

/// Match the pattern at `join` and check that the rewrite keeps the schema and the
/// error behaviour.
fn candidate(join: Node, ir_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> Option<Candidate> {
    let IR::Join {
        input_left,
        input_right: c,
        schema: output_schema,
        options: inner_options,
    } = ir_arena.get(join)
    else {
        return None;
    };
    if !plain_inner_equi_join(inner_options) {
        return None;
    }
    let c = *c;

    // A projection that only reorders columns (the restoring one from an earlier
    // push, typically) hides nothing.
    let mut below = *input_left;
    while let IR::SimpleProjection { input, columns } = ir_arena.get(below) {
        let input_schema = ir_arena.get(*input).schema(ir_arena);
        if columns.len() != input_schema.len()
            || !columns.iter_names().all(|name| input_schema.contains(name))
        {
            break;
        }
        below = *input;
    }
    // Look through the filter that keeps the outer join's own predicate, if any. It
    // will see fewer rows afterwards, so it must not be one that can raise.
    let (outer, between) = match ir_arena.get(below) {
        IR::Filter { input, predicate } if all_conjuncts_pushable(predicate, expr_arena) => {
            (*input, Some(predicate.clone()))
        },
        IR::Filter { .. } => return None,
        _ => (below, None),
    };
    let IR::Join {
        input_left: a,
        input_right: b,
        options: outer_options,
        ..
    } = ir_arena.get(outer)
    else {
        return None;
    };
    if !preserving_outer(outer_options) {
        return None;
    }
    let (a, b) = (*a, *b);
    // The outer join's own keys are evaluated on fewer rows too.
    if !outer_options
        .options
        .left_on()
        .all(|key| pushable(key.node(), expr_arena))
    {
        return None;
    }

    let a_schema = ir_arena.get(a).schema(ir_arena).into_owned();
    let b_schema = ir_arena.get(b).schema(ir_arena).into_owned();
    let c_schema = ir_arena.get(c).schema(ir_arena).into_owned();

    // The preserved side's columns come first and are never suffixed, so a key name
    // found in `A` is `A`'s column.
    let keys_from_a = inner_options.options.left_on().all(|key| {
        key.plain_column(expr_arena)
            .is_some_and(|name| a_schema.contains(name))
    });
    if !keys_from_a {
        return None;
    }
    // `C` is suffixed against `A ∪ B` before and against `A` after. A key that
    // coalescing folds away is never suffixed.
    let coalesced = |options: &JoinOptionsIR| -> Vec<PlSmallStr> {
        if !options.args.should_coalesce() {
            return Vec::new();
        }
        options
            .options
            .right_on()
            .map(|key| key.output_name().clone())
            .collect()
    };
    let (c_coalesced, b_coalesced) = (coalesced(inner_options), coalesced(outer_options));
    let collides = c_schema
        .iter_names()
        .filter(|name| !c_coalesced.contains(name))
        .any(|name| b_schema.contains(name) && !b_coalesced.contains(name));
    if collides {
        return None;
    }

    // A schema that cannot be built (a suffix collision, say) means no rewrite, not
    // an error.
    let inner_schema = det_join_schema(&a_schema, &c_schema, inner_options, expr_arena).ok()?;
    let outer_schema = det_join_schema(&inner_schema, &b_schema, outer_options, expr_arena).ok()?;
    if !same_columns(&outer_schema, output_schema) {
        return None;
    }

    Some(Candidate {
        a,
        b,
        c,
        outer,
        between,
        inner_options: inner_options.clone(),
        outer_options: outer_options.clone(),
        inner_schema,
        outer_schema,
        output_schema: output_schema.clone(),
    })
}

/// Rewrite the inner join at `join` if it matches and pays.
///
/// `stats` is shared down the recursion: every candidate is priced against subtrees
/// its parent already walked, and nothing already in the arena changes here.
fn try_push(
    join: Node,
    top_filter: Option<ExprIR>,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    stats: &mut StatsCache,
) -> Option<PushResult> {
    let Candidate {
        a,
        b,
        c,
        outer,
        between,
        inner_options,
        outer_options,
        inner_schema,
        outer_schema,
        output_schema,
    } = candidate(join, ir_arena, expr_arena)?;

    let inner = ir_arena.add(IR::Join {
        input_left: a,
        input_right: c,
        schema: inner_schema.clone(),
        options: inner_options,
    });

    // Worth it only when `A ⋈ C` is no larger than the outer join's output. The
    // filter between the joins is not priced: the estimator does not know how many
    // rows go unmatched.
    let before = node_stats_with_cache(outer, ir_arena, expr_arena, stats);
    let after = node_stats_with_cache(inner, ir_arena, expr_arena, stats);
    let worth_it = match (before, after) {
        (Some(before), Some(after)) => after.filtered <= before.filtered,
        _ => false,
    };
    if !worth_it {
        // The candidate was the last node added; its id will be reused.
        ir_arena.pop();
        stats.remove(&inner);
        return None;
    }

    // `A` may itself be an outer join to move below; the top filter goes with it. A
    // filter that can sit on this inner join but not deeper stays here, and the
    // deeper push is forgone: keeping it fused is worth more.
    let placeable_here = |predicate: &ExprIR, expr_arena: &Arena<AExpr>| {
        check_input_node(predicate.node(), &inner_schema, expr_arena)
    };
    let deeper = try_push(inner, top_filter.clone(), ir_arena, expr_arena, stats);
    let (inner, unplaced) = match deeper {
        Some(PushResult {
            root,
            unplaced_filter: None,
        }) => (root, None),
        Some(PushResult {
            root,
            unplaced_filter: Some(predicate),
        }) if !placeable_here(&predicate, expr_arena) => (root, Some(predicate)),
        _ => (inner, top_filter),
    };
    let (inner, unplaced_filter) = match unplaced {
        Some(predicate) if placeable_here(&predicate, expr_arena) => {
            let filtered = ir_arena.add(IR::Filter {
                input: inner,
                predicate,
            });
            (filtered, None)
        },
        unplaced => (inner, unplaced),
    };

    let outer = ir_arena.add(IR::Join {
        input_left: inner,
        input_right: b,
        schema: outer_schema.clone(),
        options: outer_options,
    });
    let mut root = outer;
    if let Some(predicate) = between {
        root = ir_arena.add(IR::Filter {
            input: root,
            predicate,
        });
    }
    if outer_schema != output_schema {
        root = ir_arena.add(IR::SimpleProjection {
            input: root,
            columns: output_schema,
        });
    }
    Some(PushResult {
        root,
        unplaced_filter,
    })
}

/// `Filter(is_null(rk) ∧ …)(Left(A, B))` becomes `Anti(A, B)` with `B`'s columns
/// re-added as nulls. Projection pushdown drops those that nothing reads.
#[cfg(feature = "semi_anti_join")]
fn filter_to_anti(node: Node, ir_arena: &mut Arena<IR>, expr_arena: &mut Arena<AExpr>) -> Node {
    let IR::Filter { input, predicate } = ir_arena.get(node) else {
        return node;
    };
    let IR::Join {
        input_left,
        input_right,
        schema,
        options,
    } = ir_arena.get(*input)
    else {
        return node;
    };
    let args = &options.args;
    let JoinTypeOptionsIR::Equi {
        on,
        fused_predicate: None,
    } = &options.options
    else {
        return node;
    };
    // With coalescing the right key is gone; with `nulls_equal` a matched row can
    // carry a null key.
    if !matches!(args.how, JoinType::Left)
        || !unconstrained(args)
        || args.nulls_equal
        || args.should_coalesce()
        || on.is_empty()
    {
        return node;
    }
    let (a, b) = (*input_left, *input_right);
    let a_schema = ir_arena.get(a).schema(ir_arena).into_owned();

    // A matched row always has a non-null right key, provided the key is the
    // column itself: an expression could map null to a value.
    let right_key_names: Vec<PlSmallStr> = on
        .iter()
        .filter_map(|(_, right_key)| {
            let mut expr = right_key.node();
            while let AExpr::Cast { expr: inner, .. } = expr_arena.get(expr) {
                expr = *inner;
            }
            let AExpr::Column(name) = expr_arena.get(expr) else {
                return None;
            };
            Some(if a_schema.contains(name) {
                polars_utils::format_pl_smallstr!("{}{}", name, args.suffix())
            } else {
                name.clone()
            })
        })
        .collect();
    let is_null_of_right_key = |conjunct: Node| match expr_arena.get(conjunct) {
        AExpr::Function {
            input,
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsNull),
            ..
        } => match expr_arena.get(input[0].node()) {
            AExpr::Column(name) => right_key_names.contains(name),
            _ => false,
        },
        _ => false,
    };
    if !MintermIter::new(predicate.node(), expr_arena).all(is_null_of_right_key) {
        return node;
    }

    let mut anti_options = (**options).clone();
    anti_options.args.how = JoinType::Anti;
    let schema = schema.clone();
    let anti = ir_arena.add(IR::Join {
        input_left: a,
        input_right: b,
        schema: a_schema.clone(),
        options: Arc::new(anti_options),
    });
    let exprs: Vec<ExprIR> = schema
        .iter()
        .skip(a_schema.len())
        .map(|(name, dtype)| {
            let null = expr_arena.add(AExpr::Literal(LiteralValue::untyped_null()));
            let cast = expr_arena.add(AExpr::Cast {
                expr: null,
                dtype: dtype.clone(),
                options: Default::default(),
            });
            ExprIR::new(cast, OutputName::Alias(name.clone()))
        })
        .collect();
    if exprs.is_empty() {
        return anti;
    }
    ir_arena.add(IR::HStack {
        input: anti,
        exprs,
        schema,
        options: ProjectionOptions {
            run_parallel: false,
            duplicate_check: false,
            should_broadcast: true,
            maintain_dataframe_height: true,
        },
    })
}
