//! Moving inner joins below the preserved side of a left/semi/anti join, pushing a
//! semi/anti join down to the side of an inner join that holds its keys, and turning the
//! `LEFT JOIN … WHERE right_key IS NULL` idiom into an anti join.
//!
//! `(A ⟕ B) ⋈ C` equals `(A ⋈ C) ⟕ B` when every key of the inner join comes from
//! `A`: the null-extended rows of the left join are exactly the `A` rows without a
//! match, whichever side of the inner join they meet.
//!
//! `(A ⋈ B) ⋉ S` equals `(A ⋉ S) ⋈ B` when every key of the semi join comes from
//! `A`: a semi/anti join keeps a subset of its left rows by their own values, which
//! the inner join neither changes nor depends on. The two rewrites are inverses;
//! whichever join narrows `A` more goes first.
//!
//! The last rewrite is an identity: the filter keeps only the unmatched rows, on
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
#[cfg(feature = "semi_anti_join")]
use crate::plans::schema::join_right_output_names;
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
    // The anti rewrite must not run first, or the inner join above sees an `HStack`
    // where it looks for the filter.
    let root = run_pass(root, ir_arena, expr_arena, push_inner_through);
    #[cfg(feature = "semi_anti_join")]
    let root = run_pass(root, ir_arena, expr_arena, pushdown_semi_anti);
    #[cfg(feature = "semi_anti_join")]
    let root = run_pass(root, ir_arena, expr_arena, filter_to_anti);
    root
}

pub(super) type Rule = fn(Node, &mut Arena<IR>, &mut Arena<AExpr>) -> Node;

/// Applies `rule` to every node, inputs first. A `Filter` directly over a `Join` is taken
/// as one pattern: `rule` runs on the filter, never on that join itself. The rules in this
/// file rely on this; other passes that use this walk get the same behaviour.
pub(super) fn run_pass(
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

/// Elementwise, fallible or not: a filter another predicate may pass below.
#[cfg(feature = "semi_anti_join")]
fn no_conjunct_barrier(predicate: &ExprIR, expr_arena: &Arena<AExpr>) -> bool {
    MintermIter::new(predicate.node(), expr_arena).all(|node| {
        let mut group = ExprPushdownGroup::Pushable;
        group.update_with_expr_rec(expr_arena.get(node), expr_arena, None);
        !group.blocks_pushdown(false)
    })
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
    let inner_schema = det_join_schema(&a_schema, &c_schema, inner_options).ok()?;
    let outer_schema = det_join_schema(&inner_schema, &b_schema, outer_options).ok()?;
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

/// `Semi(T(Inner(A, B)), S)` becomes `T(Inner(Semi(A, S), B))` (or the mirror into
/// `B`), where `T` is a chain of simple projections and elementwise filters. The pushed
/// join is pushed on through whatever inner joins it lands on.
///
/// The inverse of [`push_inner_through`]. Both price the same two subplans, and only
/// this one insists on a strict win, so they never undo each other.
#[cfg(feature = "semi_anti_join")]
fn pushdown_semi_anti(node: Node, ir_arena: &mut Arena<IR>, expr_arena: &mut Arena<AExpr>) -> Node {
    let mut stats = StatsCache::default();
    pushdown_semi_anti_with(node, ir_arena, expr_arena, &mut stats)
}

#[cfg(feature = "semi_anti_join")]
fn pushdown_semi_anti_with(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    stats: &mut StatsCache,
) -> Node {
    // The driver hands a filter over a join in as one node.
    if let IR::Filter { input, predicate } = ir_arena.get(node) {
        let (input, predicate) = (*input, predicate.clone());
        let pushed = pushdown_semi_anti_with(input, ir_arena, expr_arena, stats);
        if pushed == input {
            return node;
        }
        return ir_arena.add(IR::Filter {
            input: pushed,
            predicate,
        });
    }
    let Some(candidate) = pushdown_candidate(node, ir_arena, expr_arena) else {
        return node;
    };
    let pushed = ir_arena.add(candidate.pushed_join(ir_arena));
    if !pushdown_pays(&candidate, pushed, ir_arena, expr_arena, stats) {
        // The candidate is the last node added, so its id is reused.
        debug_assert_eq!(pushed.0, ir_arena.len() - 1);
        ir_arena.pop();
        stats.remove(&pushed);
        return node;
    }
    let pushed = pushdown_semi_anti_with(pushed, ir_arena, expr_arena, stats);
    candidate.rebuild(pushed, ir_arena)
}

/// The parts of a `Semi(T(Inner(A, B)), S)` that passed every check but cost.
#[cfg(feature = "semi_anti_join")]
struct PushdownCandidate {
    /// The side of the inner join holding the semi join's keys, and the other.
    side: Node,
    other: Node,
    side_is_left: bool,
    s: Node,
    inner: Node,
    inner_schema: SchemaRef,
    inner_options: Arc<JoinOptionsIR>,
    /// The semi join's options, keyed by the side's own column names.
    side_options: Arc<JoinOptionsIR>,
    /// `T`, top down.
    chain: Vec<Node>,
}

/// Match the pattern at the semi/anti join `node`, and find which side of the inner
/// join below it holds every key.
#[cfg(feature = "semi_anti_join")]
fn pushdown_candidate(
    node: Node,
    ir_arena: &Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<PushdownCandidate> {
    let IR::Join {
        input_left,
        input_right: s,
        options,
        ..
    } = ir_arena.get(node)
    else {
        return None;
    };
    let JoinTypeOptionsIR::Equi {
        on,
        fused_predicate: None,
    } = &options.options
    else {
        return None;
    };
    if !options.args.how.is_semi_anti() || !unconstrained(&options.args) || on.is_empty() {
        return None;
    }
    let keys = on
        .iter()
        .map(|(left_key, _)| left_key.plain_column(expr_arena).cloned())
        .collect::<Option<Vec<PlSmallStr>>>()?;
    let (s, options) = (*s, options.clone());

    // The nodes between the semi join and the inner join, top down.
    let mut chain = Vec::new();
    let mut below = *input_left;
    let inner = loop {
        match ir_arena.get(below) {
            IR::SimpleProjection { input, .. } => {
                chain.push(below);
                below = *input;
            },
            IR::Filter { input, predicate } if no_conjunct_barrier(predicate, expr_arena) => {
                chain.push(below);
                below = *input;
            },
            IR::Join {
                options: inner_options,
                ..
            } if plain_inner_equi_join(inner_options) => break below,
            _ => return None,
        }
    };
    let IR::Join {
        input_left: a,
        input_right: b,
        schema: inner_schema,
        options: inner_options,
    } = ir_arena.get(inner)
    else {
        unreachable!()
    };
    let (a, b, inner_schema, inner_options) = (*a, *b, inner_schema.clone(), inner_options.clone());
    let a_schema = ir_arena.get(a).schema(ir_arena);
    let b_schema = ir_arena.get(b).schema(ir_arena);

    // `A`'s columns keep their names through the join; a `B` column may be renamed
    // or coalesced away.
    let (side, other, side_is_left, side_keys) = if keys.iter().all(|name| a_schema.contains(name))
    {
        (a, b, true, keys)
    } else {
        let b_output_names = join_right_output_names(&a_schema, &b_schema, &inner_options).ok()?;
        let b_column = |name: &PlSmallStr| -> Option<PlSmallStr> {
            b_schema
                .iter_names()
                .zip(&b_output_names)
                .find(|(_, output_name)| output_name.as_ref() == Some(name))
                .map(|(column, _)| column.clone())
        };
        let columns = keys.iter().map(b_column).collect::<Option<Vec<_>>>()?;
        (b, a, false, columns)
    };

    let mut side_options = (*options).clone();
    if let JoinTypeOptionsIR::Equi { on, .. } = &mut side_options.options {
        for ((left_key, _), name) in on.iter_mut().zip(side_keys) {
            if left_key.output_name() != &name {
                let column = expr_arena.add(AExpr::Column(name.clone()));
                *left_key = ExprIR::new(column, OutputName::ColumnLhs(name));
            }
        }
    }
    Some(PushdownCandidate {
        side,
        other,
        side_is_left,
        s,
        inner,
        inner_schema,
        inner_options,
        side_options: Arc::new(side_options),
        chain,
    })
}

#[cfg(feature = "semi_anti_join")]
impl PushdownCandidate {
    /// The semi join on its side alone.
    fn pushed_join(&self, ir_arena: &Arena<IR>) -> IR {
        IR::Join {
            input_left: self.side,
            input_right: self.s,
            schema: ir_arena.get(self.side).schema(ir_arena).into_owned(),
            options: self.side_options.clone(),
        }
    }

    /// The inner join over `pushed` in place of its side, and `T` over that.
    fn rebuild(self, pushed: Node, ir_arena: &mut Arena<IR>) -> Node {
        let (input_left, input_right) = if self.side_is_left {
            (pushed, self.other)
        } else {
            (self.other, pushed)
        };
        let mut root = ir_arena.add(IR::Join {
            input_left,
            input_right,
            schema: self.inner_schema,
            options: self.inner_options,
        });
        for &step in self.chain.iter().rev() {
            let mut ir = ir_arena.get(step).clone();
            for slot in ir.inputs_mut() {
                *slot = root;
            }
            root = ir_arena.add(ir);
        }
        root
    }
}

/// Whether the semi join narrows its side to fewer rows than the inner join emits.
///
/// A semi join that probes no more rows pushed than it does now costs nothing extra
/// whatever its right side keeps. Otherwise it is priced at the bound on its right
/// side's rows, and stays put without one: an estimate of that side that is too low
/// would push down a join that keeps most rows. An anti join's estimate errs the
/// other way.
#[cfg(feature = "semi_anti_join")]
fn pushdown_pays(
    candidate: &PushdownCandidate,
    pushed: Node,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    stats: &mut StatsCache,
) -> bool {
    let before = node_stats_with_cache(candidate.inner, ir_arena, expr_arena, stats);
    let after = node_stats_with_cache(pushed, ir_arena, expr_arena, stats);
    let right = node_stats_with_cache(candidate.s, ir_arena, expr_arena, stats);
    let (Some(before), Some(after), Some(right)) = (before, after, right) else {
        return false;
    };
    let mut after = after.filtered;
    if matches!(candidate.side_options.args.how, JoinType::Semi) {
        let side = node_stats_with_cache(candidate.side, ir_arena, expr_arena, stats);
        let semi_input = match candidate.chain.first() {
            Some(&top) => node_stats_with_cache(top, ir_arena, expr_arena, stats),
            None => Some(before.clone()),
        };
        if let (Some(side), Some(semi_input)) = (&side, semi_input)
            && side.filtered <= semi_input.filtered
        {
            return true;
        }
        let Some(bound) = right.max_rows() else {
            return false;
        };
        if bound > right.filtered {
            let side_rows = side.map_or(f64::INFINITY, |side| side.filtered);
            after = (after * bound / right.filtered).min(side_rows);
        }
    }
    after < before.filtered
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
