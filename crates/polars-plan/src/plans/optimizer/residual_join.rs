use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::prelude::Schema;
use polars_ops::frame::JoinArgs;
use polars_utils::aliases::{PlIndexMap, PlIndexSet};
use polars_utils::arena::{Arena, Node};
use polars_utils::idx_vec::UnitVec;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{format_pl_smallstr, unitvec};

use super::join_utils::ExprOrigin;
use super::predicate_pushdown::utils::{combine_by_and, contains_dynamic_pred};
use crate::dsl::Operator;
use crate::plans::iterator::ArenaExprIter;
use crate::plans::options::JoinTypeOptionsIR;
use crate::plans::{
    AExpr, ExprIR, ExprPushdownGroup, IR, JoinOptionsIR, JoinType, MintermIter, OutputName,
    is_row_separable_rec,
};
use crate::utils::{aexpr_to_leaf_names_iter, rename_columns};

/// Fuse `Filter(join)` predicates that span both inputs into the join's match condition.
pub(super) fn fuse_residual_predicates(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<()> {
    let mut stack: UnitVec<Node> = unitvec![root];
    let mut seen = PlIndexSet::default();

    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        // Read the inputs before fusing, which replaces what `node` holds.
        ir_arena.get(node).copy_inputs(&mut stack);
        try_fuse(node, ir_arena, expr_arena)?;
    }

    Ok(())
}

/// A residual is only sound on an inner equi join, and a slice must stay above the filter
/// that feeds it.
fn is_fusable_join(options: &JoinOptionsIR) -> bool {
    matches!(options.args.how, JoinType::Inner)
        && options.args.slice.is_none()
        && matches!(
            &options.options,
            JoinTypeOptionsIR::Equi { on, residual: None } if !on.is_empty()
        )
}

/// Elementwise and infallible, so the join can evaluate it per candidate pair.
fn can_fuse_predicate(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    let mut group = ExprPushdownGroup::Pushable;
    group.update_with_expr_rec(expr_arena.get(node), expr_arena, None);

    matches!(group, ExprPushdownGroup::Pushable)
        && is_row_separable_rec(node, expr_arena)
        && !contains_dynamic_pred(node, expr_arena)
}

fn try_fuse(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<()> {
    let IR::Filter { input, predicate } = ir_arena.get(node) else {
        return Ok(());
    };
    let (input, predicate) = (*input, predicate.node());

    let IR::Join {
        input_left,
        input_right,
        schema,
        options,
    } = ir_arena.get(input)
    else {
        return Ok(());
    };
    if !is_fusable_join(options) {
        return Ok(());
    }
    let (input_left, input_right, schema, mut options) =
        (*input_left, *input_right, schema.clone(), options.clone());

    // A single unfusable term leaves the whole filter alone.
    let minterms = MintermIter::new(predicate, expr_arena).collect::<Vec<_>>();
    if !minterms
        .iter()
        .all(|node| can_fuse_predicate(*node, expr_arena))
    {
        return Ok(());
    }

    let left_schema = ir_arena.get(input_left).schema(ir_arena).into_owned();
    let right_schema = ir_arena.get(input_right).schema(ir_arena).into_owned();
    let suffix = options.args.suffix().clone();

    let right_names = right_output_to_input(&left_schema, &right_schema, &options, expr_arena)?;

    let mut left_key_names: PlIndexSet<PlSmallStr> = options
        .options
        .left_on()
        .map(|key| key.output_name().clone())
        .collect();
    let mut right_key_names: PlIndexSet<PlSmallStr> = options
        .options
        .right_on()
        .map(|key| key.output_name().clone())
        .collect();

    let mut promoted: Vec<(ExprIR, ExprIR)> = Vec::new();
    let mut fused: UnitVec<Node> = unitvec![];
    let mut kept: UnitVec<Node> = unitvec![];
    for minterm in minterms {
        // Coalescing on an inner join keeps the left name, so no right-join callback.
        let origin = ExprOrigin::get_expr_origin(
            minterm,
            expr_arena,
            &left_schema,
            &right_schema,
            &suffix,
            None,
        )?;

        // Anything reading a single input is the existing pushdown's job.
        if !matches!(origin, ExprOrigin::Both) {
            kept.push(minterm);
            continue;
        }

        // An equality is cheaper as a key than as a residual.
        if let Some(pair) = as_key_pair(
            minterm,
            expr_arena,
            &left_schema,
            &right_schema,
            &options.args,
            &right_names,
        )? {
            promoted.push(name_key_pair(
                pair,
                &options.args,
                &left_schema,
                &right_schema,
                &mut left_key_names,
                &mut right_key_names,
            ));
            continue;
        }

        fused.push(minterm);
    }

    if fused.is_empty() && promoted.is_empty() {
        return Ok(());
    }

    let fused = fused
        .into_iter()
        .reduce(|left, right| combine_by_and(left, right, expr_arena));
    let kept = kept
        .into_iter()
        .reduce(|left, right| combine_by_and(left, right, expr_arena));

    let options_mut = Arc::make_mut(&mut options);
    for (left, right) in promoted {
        options_mut.options.push_key_pair(left, right);
    }
    if let Some(fused) = fused {
        options_mut
            .options
            .set_residual(ExprIR::from_node(fused, expr_arena));
    }
    let join = IR::Join {
        input_left,
        input_right,
        schema,
        options,
    };

    // The fused join replaces the filter rather than the original join node, which other
    // branches may still reference with a different filter above it.
    let fused_node = match kept {
        None => join,
        Some(kept) => {
            let join = ir_arena.add(join);
            IR::Filter {
                input: join,
                predicate: ExprIR::from_node(kept, expr_arena),
            }
        },
    };
    ir_arena.replace(node, fused_node);

    Ok(())
}

/// Whether an expression is safe to evaluate on every input row.
///
/// A promoted key runs on all rows of its input, not only on the candidate pairs the
/// filter would have seen, so it must not fail or draw randomly on rows the query
/// excludes. Fallibility is tracked per known function rather than proven, so this
/// admits only operations that cannot raise at all.
fn can_promote_key(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    expr_arena.iter(node).all(|(_, ae)| match ae {
        AExpr::Column(_) | AExpr::Literal(_) => true,
        AExpr::Cast { options, .. } => !options.is_strict(),
        _ => false,
    })
}

/// Read a both-sided equality minterm as a join key pair, if it is one.
///
/// Keys come back in their input namespaces, carrying whatever name their expression
/// gives them; [`name_key_pair`] settles the collisions that leaves.
fn as_key_pair(
    minterm: Node,
    expr_arena: &mut Arena<AExpr>,
    left_schema: &Schema,
    right_schema: &Schema,
    args: &JoinArgs,
    right_names: &PlIndexMap<PlSmallStr, PlSmallStr>,
) -> PolarsResult<Option<(ExprIR, ExprIR)>> {
    // Validation counts the rows per key, which a promoted equality would change.
    if args.validation.needs_checks() {
        return Ok(None);
    }

    let AExpr::BinaryExpr { left, op, right } = expr_arena.get(minterm) else {
        return Ok(None);
    };
    let (left, right) = (*left, *right);

    // A key pair matches nulls only when `nulls_equal` is set; `==` never does and
    // `eq_missing` always does.
    let matches_nulls = match op {
        Operator::Eq => false,
        Operator::EqValidity => true,
        _ => return Ok(None),
    };
    if matches_nulls != args.nulls_equal {
        return Ok(None);
    }

    let suffix = args.suffix();
    let left_origin =
        ExprOrigin::get_expr_origin(left, expr_arena, left_schema, right_schema, suffix, None)?;
    let right_origin =
        ExprOrigin::get_expr_origin(right, expr_arena, left_schema, right_schema, suffix, None)?;
    let (left_node, right_node) = match (left_origin, right_origin) {
        (ExprOrigin::Left, ExprOrigin::Right) => (left, right),
        (ExprOrigin::Right, ExprOrigin::Left) => (right, left),
        _ => return Ok(None),
    };

    if !can_promote_key(left_node, expr_arena) || !can_promote_key(right_node, expr_arena) {
        return Ok(None);
    }

    // The right side is in the join's output namespace, where a name can belong to a
    // different input column than the one it shares a name with.
    if !aexpr_to_leaf_names_iter(right_node, expr_arena)
        .all(|name| right_names.contains_key(name.as_str()))
    {
        return Ok(None);
    }
    let right_node = rename_columns(right_node, expr_arena, right_names);

    let left_key = ExprIR::from_node(left_node, expr_arena);
    let right_key = ExprIR::from_node(right_node, expr_arena);

    // Key pairs are matched without coercion.
    if left_key.field(left_schema, expr_arena)?.dtype
        != right_key.field(right_schema, expr_arena)?.dtype
    {
        return Ok(None);
    }

    Ok(Some((left_key, right_key)))
}

/// Aliases a promoted key pair away from the names already spoken for.
///
/// The taken sets grow as pairs are named, so two promotions cannot land on one name.
fn name_key_pair(
    (mut left_key, mut right_key): (ExprIR, ExprIR),
    args: &JoinArgs,
    left_schema: &Schema,
    right_schema: &Schema,
    left_key_names: &mut PlIndexSet<PlSmallStr>,
    right_key_names: &mut PlIndexSet<PlSmallStr>,
) -> (ExprIR, ExprIR) {
    if !left_key_names.insert(left_key.output_name().clone()) {
        let name = unique_key_name(left_key.output_name(), left_key_names, left_schema);
        left_key = ExprIR::new(left_key.node(), OutputName::Alias(name.clone()));
        left_key_names.insert(name);
    }

    // Coalescing drops the right payload column that shares a name with a right key.
    let right_name = right_key.output_name().clone();
    if args.should_coalesce() || !right_key_names.insert(right_name.clone()) {
        let name = unique_key_name(&right_name, right_key_names, right_schema);
        right_key = ExprIR::new(right_key.node(), OutputName::Alias(name.clone()));
        right_key_names.insert(name);
    }

    (left_key, right_key)
}

fn unique_key_name(base: &str, taken: &PlIndexSet<PlSmallStr>, schema: &Schema) -> PlSmallStr {
    (0..)
        .map(|i| format_pl_smallstr!("__POLARS_JOIN_KEY_{i}_{base}"))
        .find(|name| !taken.contains(name) && !schema.contains(name))
        .unwrap()
}

/// Maps each right column reachable in the join's output back to its input name.
///
/// A coalescing join drops the right key columns, and the rest take the join suffix where
/// they collide with a left column, so an output name need not be the input name.
fn right_output_to_input(
    left_schema: &Schema,
    right_schema: &Schema,
    options: &JoinOptionsIR,
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<PlIndexMap<PlSmallStr, PlSmallStr>> {
    let mut coalesced: PlIndexSet<PlSmallStr> = PlIndexSet::default();
    if options.args.should_coalesce() {
        for key in options.options.right_on() {
            coalesced.insert(key.field(right_schema, expr_arena)?.name);
        }
    }

    let suffix = options.args.suffix();
    let mut out = PlIndexMap::default();
    for name in right_schema.iter_names() {
        if coalesced.contains(name) {
            continue;
        }
        let output_name = if left_schema.contains(name) {
            format_pl_smallstr!("{}{}", name, suffix)
        } else {
            name.clone()
        };
        out.insert(output_name, name.clone());
    }
    Ok(out)
}
