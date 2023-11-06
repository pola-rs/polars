use polars_core::datatypes::PlHashMap;
use polars_core::prelude::*;

use super::keys::*;
use crate::logical_plan::Context;
use crate::prelude::*;
use crate::utils::{aexpr_to_leaf_names, check_input_node, has_aexpr, rename_aexpr_leaf_names};

trait Dsl {
    fn and(self, right: Node, arena: &mut Arena<AExpr>) -> Node;
}

impl Dsl for Node {
    fn and(self, right: Node, arena: &mut Arena<AExpr>) -> Node {
        arena.add(AExpr::BinaryExpr {
            left: self,
            op: Operator::And,
            right,
        })
    }
}

/// Don't overwrite predicates but combine them.
pub(super) fn insert_and_combine_predicate(
    acc_predicates: &mut PlHashMap<Arc<str>, Node>,
    predicate: Node,
    arena: &mut Arena<AExpr>,
) {
    let name = predicate_to_key(predicate, arena);

    acc_predicates
        .entry(name)
        .and_modify(|existing_predicate| {
            let node = arena.add(AExpr::BinaryExpr {
                left: predicate,
                op: Operator::And,
                right: *existing_predicate,
            });
            *existing_predicate = node
        })
        .or_insert_with(|| predicate);
}

pub(super) fn combine_predicates<I>(iter: I, arena: &mut Arena<AExpr>) -> Node
where
    I: Iterator<Item = Node>,
{
    let mut single_pred = None;
    for node in iter {
        single_pred = match single_pred {
            None => Some(node),
            Some(left) => Some(arena.add(AExpr::BinaryExpr {
                left,
                op: Operator::And,
                right: node,
            })),
        };
    }
    single_pred.expect("an empty iterator was passed")
}

pub(super) fn predicate_at_scan(
    acc_predicates: PlHashMap<Arc<str>, Node>,
    predicate: Option<Node>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<Node> {
    if !acc_predicates.is_empty() {
        let mut new_predicate =
            combine_predicates(acc_predicates.into_iter().map(|t| t.1), expr_arena);
        if let Some(pred) = predicate {
            new_predicate = new_predicate.and(pred, expr_arena)
        }
        Some(new_predicate)
    } else {
        None
    }
}

fn shifts_elements(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    let matches = |e: &AExpr| {
        matches!(
            e,
            AExpr::Function {
                function: FunctionExpr::Shift | FunctionExpr::ShiftAndFill,
                ..
            }
        )
    };
    has_aexpr(node, expr_arena, matches)
}

pub(super) fn predicate_is_sort_boundary(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    let matches = |e: &AExpr| match e {
        AExpr::Window { function, .. } => shifts_elements(*function, expr_arena),
        AExpr::Function { options, .. } | AExpr::AnonymousFunction { options, .. } => {
            // this check for functions that are
            // group sensitive and doesn't auto-explode (e.g. is a reduction/aggregation
            // like sum, min, etc).
            // function that match this are `cumsum`, `shift`, `sort`, etc.
            options.is_groups_sensitive() && !options.returns_scalar
        },
        _ => false,
    };
    has_aexpr(node, expr_arena, matches)
}

/// Predicates can be renamed during pushdown to support being pushed past
/// aliases. For example, given the projection `col(A).alias(B)`, and the
/// predicate `col(B) == 1`, the predicate can be pushed past the alias by being
/// re-written as col(A) == 1. However, this must not be done for aliases that
/// are preceded by operations on the column.
/// For instance, `predicate = col(A) == col(B).`
/// and `col().some_func().alias(B)` is projected.
/// then the projection can not pass, as column `B` maybe
/// changed by `some_func`
pub(super) fn aexpr_blocks_aliased_predicate_pushdown(
    node: Node,
    expr_arena: &Arena<AExpr>,
) -> bool {
    let matches = |e: &AExpr| {
        use AExpr::*;
        // anything that changes output values modifies the predicate result
        // and is not captured by `aexpr_blocks_predicate_pushdown`.

        // explicit match is more readable in this case
        #[allow(clippy::match_like_matches_macro)]
        match e {
            AnonymousFunction { .. }
            | Function { .. }
            | BinaryExpr { .. }
            | Ternary { .. }
            | Cast { .. } => true,
            _ => false,
        }
    };
    has_aexpr(node, expr_arena, matches)
}

enum LoopBehavior {
    Continue,
    Nothing,
}

fn rename_predicate_columns_due_to_aliased_projection(
    expr_arena: &mut Arena<AExpr>,
    acc_predicates: &mut PlHashMap<Arc<str>, Node>,
    projection_node: Node,
    projection_maybe_boundary: bool,
    local_predicates: &mut Vec<Node>,
) -> LoopBehavior {
    let projection_aexpr = expr_arena.get(projection_node);
    if let AExpr::Alias(_, alias_name) = projection_aexpr {
        let alias_name = alias_name.clone();
        let projection_leaves = aexpr_to_leaf_names(projection_node, expr_arena);

        // this means the leaf is a literal
        if projection_leaves.is_empty() {
            return LoopBehavior::Nothing;
        }

        // if this alias refers to one of the predicates in the upper nodes
        // we rename the column of the predicate before we push it downwards.
        if let Some(predicate) = acc_predicates.remove(&alias_name) {
            if projection_maybe_boundary {
                local_predicates.push(predicate);
                remove_predicate_refers_to_alias(acc_predicates, local_predicates, &alias_name);
                return LoopBehavior::Continue;
            }
            if projection_leaves.len() == 1 {
                // we were able to rename the alias column with the root column name
                // before pushing down the predicate
                let predicate =
                    rename_aexpr_leaf_names(predicate, expr_arena, projection_leaves[0].clone());

                insert_and_combine_predicate(acc_predicates, predicate, expr_arena);
            } else {
                // this may be a complex binary function. The predicate may only be valid
                // on this projected column so we do filter locally.
                local_predicates.push(predicate)
            }
        }

        remove_predicate_refers_to_alias(acc_predicates, local_predicates, &alias_name);
    }
    LoopBehavior::Nothing
}

/// we could not find the alias name
/// that could still mean that a predicate that is a complicated binary expression
/// refers to the aliased name. If we find it, we remove it for now
/// TODO! rename the expression.
fn remove_predicate_refers_to_alias(
    acc_predicates: &mut PlHashMap<Arc<str>, Node>,
    local_predicates: &mut Vec<Node>,
    alias_name: &str,
) {
    let mut remove_names = vec![];
    for (composed_name, _) in acc_predicates.iter() {
        if key_has_name(composed_name, alias_name) {
            remove_names.push(composed_name.clone());
            break;
        }
    }

    for composed_name in remove_names {
        let predicate = acc_predicates.remove(&composed_name).unwrap();
        local_predicates.push(predicate)
    }
}

/// Implementation for both Hstack and Projection
/// Safety: `projections` and `acc_predicates` do not contain blocking exprs.
pub(super) fn rewrite_projection_node(
    expr_arena: &mut Arena<AExpr>,
    lp_arena: &Arena<ALogicalPlan>,
    acc_predicates: &mut PlHashMap<Arc<str>, Node>,
    projections: Vec<Node>,
    input: Node,
) -> (Vec<Node>, Vec<Node>)
where
{
    let mut local_predicates = Vec::with_capacity(acc_predicates.len());

    // maybe update predicate name if a projection is an alias
    // aliases change the column names and because we push the predicates downwards
    // this may be problematic as the aliased column may not yet exist.
    for projection_node in &projections {
        // only if a predicate refers to this projection's output column.
        let projection_maybe_boundary =
            aexpr_blocks_aliased_predicate_pushdown(*projection_node, expr_arena);

        {
            // if this alias refers to one of the predicates in the upper nodes
            // we rename the column of the predicate before we push it downwards.
            match rename_predicate_columns_due_to_aliased_projection(
                expr_arena,
                acc_predicates,
                *projection_node,
                projection_maybe_boundary,
                &mut local_predicates,
            ) {
                LoopBehavior::Continue => continue,
                LoopBehavior::Nothing => {},
            }
        }
        let input_schema = lp_arena.get(input).schema(lp_arena);
        let projection_expr = expr_arena.get(*projection_node);
        let output_field = projection_expr
            .to_field(&input_schema, Context::Default, expr_arena)
            .unwrap();

        // should have been handled earlier by `pushdown_and_continue`.
        assert_aexpr_allows_predicate_pushdown(*projection_node, expr_arena);

        // remove predicates that cannot be done on the input above
        let to_local = acc_predicates
            .iter()
            .filter_map(|(name, predicate)| {
                // there are some conditions we need to check for every predicate we try to push down
                // 1. does the column exist on the node above
                // 2. if the projection is a computation/transformation and the predicate is based on that column
                //    we must block because the predicate would be incorrect.

                // checks 1.
                if check_input_node(*predicate, &input_schema, expr_arena)
                // checks 2.
                && !(key_has_name(name, output_field.name()) && projection_maybe_boundary)
                {
                    None
                } else {
                    Some(name.clone())
                }
            })
            .collect::<Vec<_>>();

        for name in to_local {
            let local = acc_predicates.remove(&name).unwrap();
            local_predicates.push(local);
        }

        // remove predicates that are based on column modifications
        no_pushdown_preds(
            *projection_node,
            expr_arena,
            |e| matches!(e, AExpr::Explode(_)) || matches!(e, AExpr::Ternary { .. }),
            &mut local_predicates,
            acc_predicates,
        );
    }
    (local_predicates, projections)
}

pub(super) fn no_pushdown_preds<F>(
    // node that is projected | hstacked
    node: Node,
    arena: &Arena<AExpr>,
    matches: F,
    // predicates that will be filtered at this node in the LP
    local_predicates: &mut Vec<Node>,
    acc_predicates: &mut PlHashMap<Arc<str>, Node>,
) where
    F: Fn(&AExpr) -> bool,
{
    // matching expr are typically explode, shift, etc. expressions that mess up predicates when pushed down
    if has_aexpr(node, arena, matches) {
        // columns that are projected. We check if we can push down the predicates past this projection
        let columns = aexpr_to_leaf_names(node, arena);

        let condition = |name: Arc<str>| columns.contains(&name);
        local_predicates.extend(transfer_to_local_by_name(arena, acc_predicates, condition));
    }
}

/// Transfer a predicate from `acc_predicates` that will be pushed down
/// to a local_predicates vec based on a condition.
pub(super) fn transfer_to_local_by_name<F>(
    expr_arena: &Arena<AExpr>,
    acc_predicates: &mut PlHashMap<Arc<str>, Node>,
    mut condition: F,
) -> Vec<Node>
where
    F: FnMut(Arc<str>) -> bool,
{
    let mut remove_keys = Vec::with_capacity(acc_predicates.len());

    for (key, predicate) in &*acc_predicates {
        let root_names = aexpr_to_leaf_names(*predicate, expr_arena);
        for name in root_names {
            if condition(name) {
                remove_keys.push(key.clone());
                continue;
            }
        }
    }
    let mut local_predicates = Vec::with_capacity(remove_keys.len());
    for key in remove_keys {
        if let Some(pred) = acc_predicates.remove(&*key) {
            local_predicates.push(pred)
        }
    }
    local_predicates
}

/// An expression blocks predicates from being pushed past it if its results for
/// the subset where the predicate evaluates as true becomes different compared
/// to if it was performed before the predicate was applied. This is in general
/// any expression that produces outputs based on groups of values
/// (i.e. groups-wise) rather than individual values (i.e. element-wise).
///
/// Examples of expressions whose results would change, and thus block push-down:
/// - any aggregation - sum, mean, first, last, min, max etc.
/// - sorting - as the sort keys would change between filters
///
/// Examples of expressions whose results do not change, and thus allow push-down:
/// - column addition, equality, string concatenation
pub(super) fn aexpr_blocks_predicate_pushdown(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    let mut stack = Vec::<Node>::with_capacity(4);
    stack.push(node);

    while !stack.is_empty() {
        let node = stack.pop().unwrap();
        let ae = expr_arena.get(node);

        let should_block = match ae {
            AExpr::Function {
                function: FunctionExpr::Boolean(BooleanFunction::IsIn),
                input,
                ..
            } => {
                // Handles a special case where the expr contains a series, but it is being
                // used as part of an `is_in`, so it can be pushed down.
                let values = input.get(1).unwrap();
                let ae = expr_arena.get(*values);
                if matches!(ae, AExpr::Literal { .. }) {
                    // Still need to check the input expr (LHS) of the is_in.
                    let node = *input.get(0).unwrap();
                    stack.push(node);
                    expr_arena.get(node).nodes(&mut stack);
                    false
                } else {
                    ae.nodes(&mut stack);
                    ae.groups_sensitive()
                }
            },
            // Already checked we are not inside an `is_in`, meaning these literals are
            // either projecting as columns or predicates, both of which rely on the
            // current height of the dataframe and thus need to block pushdown.
            AExpr::Literal(LiteralValue::Range { .. }) => true,
            AExpr::Literal(LiteralValue::Series(s)) => s.len() > 1,
            ae => {
                ae.nodes(&mut stack);
                ae.groups_sensitive()
            },
        };

        if should_block {
            return should_block;
        };
    }

    false
}

/// Used in places that previously handled blocking exprs before refactoring.
/// Can probably be eventually removed if it isn't catching anything.
#[inline(always)]
pub(super) fn assert_aexpr_allows_predicate_pushdown(node: Node, expr_arena: &Arena<AExpr>) {
    assert!(
        !aexpr_blocks_predicate_pushdown(node, expr_arena),
        "Predicate pushdown: Did not expect blocking exprs at this point, please open an issue."
    );
}
