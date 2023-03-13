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

// this checks if a predicate from a node upstream can pass
// the predicate in this filter
// Cases where this cannot be the case:
//
// .filter(a > 1)           # filter 2
///.filter(a == min(a))     # filter 1
///
/// the min(a) is influenced by filter 2 so min(a) should not pass
pub(super) fn predicate_is_pushdown_boundary(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    let matches = |e: &AExpr| {
        matches!(
            e,
            AExpr::Sort { .. } | AExpr::SortBy { .. }
            | AExpr::Agg(_) // an aggregation needs all rows
            // Apply groups can be something like shift, sort, or an aggregation like skew
            // both need all values
            | AExpr::AnonymousFunction {options: FunctionOptions { collect_groups: ApplyOptions::ApplyGroups, .. }, ..}
            | AExpr::Function {options: FunctionOptions { collect_groups: ApplyOptions::ApplyGroups, .. }, ..}
            | AExpr::Explode {..}
            // A groupby needs all rows for aggregation
            | AExpr::Window {..}
        )
    };
    has_aexpr(node, expr_arena, matches)
}

/// Some predicates should not pass a projection if they would influence results of other columns.
/// For instance shifts | sorts results are influenced by a filter so we do all predicates before the shift | sort
/// The rule of thumb is any operation that changes the order of a column w/r/t other columns should be a
/// predicate pushdown blocker.
///
/// This checks the boundary of other columns
pub(super) fn projection_is_definite_pushdown_boundary(
    node: Node,
    expr_arena: &Arena<AExpr>,
) -> bool {
    let matches = |e: &AExpr| {
        use AExpr::*;
        // any result that will change due to rows filtered before the projection

        // explicit match is more readable in this case
        #[allow(clippy::match_like_matches_macro)]
        match e {
             Agg(_) // an aggregation needs all rows
            // Apply groups can be something like shift, sort, or an aggregation like skew
            // both need all values
            | AnonymousFunction {options: FunctionOptions { collect_groups: ApplyOptions::ApplyGroups, .. }, ..}
            | Function {options: FunctionOptions { collect_groups: ApplyOptions::ApplyGroups, .. }, ..}
            // still need to investigate this one
            | Explode {..}
            | Count
             | Nth(_)
             | Slice {..}
             | Take {..}
            // A groupby needs all rows for aggregation
            | Window {..}
            | Literal(LiteralValue::Range {..}) => true,
            // The series might be used in a comparison with exactly the right length
            Literal(LiteralValue::Series(s)) => s.len() > 1,
            _ => false
        }
    };
    has_aexpr(node, expr_arena, matches)
}

/// This is only a boundary if a predicate refers to the projection output name.
/// This checks the boundary of same columns.
/// So that means columns that are referred in the predicate
/// for instance `predicate = col(A) == col(B).`
/// and `col().some_func().alias(B)` is projected.
/// then the projection can not pass, as column `B` maybe
/// changed by `some_func`
pub(super) fn projection_is_optional_pushdown_boundary(
    node: Node,
    expr_arena: &Arena<AExpr>,
) -> bool {
    let matches = |e: &AExpr| {
        use AExpr::*;
        // anything that changes output values modifies the predicate result
        // and is not captured by function above: `projection_is_definite_pushdown_boundary`

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
        let alias_name = alias_name.as_ref();
        let projection_roots = aexpr_to_leaf_names(projection_node, expr_arena);
        // if this alias refers to one of the predicates in the upper nodes
        // we rename the column of the predicate before we push it downwards.
        if let Some(predicate) = acc_predicates.remove(alias_name) {
            if projection_maybe_boundary {
                local_predicates.push(predicate);
                return LoopBehavior::Continue;
            }
            if projection_roots.len() == 1 {
                // we were able to rename the alias column with the root column name
                // before pushing down the predicate
                let predicate =
                    rename_aexpr_leaf_names(predicate, expr_arena, projection_roots[0].clone());

                insert_and_combine_predicate(acc_predicates, predicate, expr_arena);
            } else {
                // this may be a complex binary function. The predicate may only be valid
                // on this projected column so we do filter locally.
                local_predicates.push(predicate)
            }
        } else {
            // we could not find the alias name
            // that could still mean that a predicate that is a complicated binary expression
            // refers to the aliased name. If we find it, we remove it for now
            // TODO! rename the expression.
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
    }
    LoopBehavior::Nothing
}

/// Implementation for both Hstack and Projection
pub(super) fn rewrite_projection_node(
    expr_arena: &mut Arena<AExpr>,
    lp_arena: &mut Arena<ALogicalPlan>,
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
            projection_is_optional_pushdown_boundary(*projection_node, expr_arena);

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
                LoopBehavior::Nothing => {}
            }
        }
        let input_schema = lp_arena.get(input).schema(lp_arena);
        let projection_expr = expr_arena.get(*projection_node);
        let output_field = projection_expr
            .to_field(&input_schema, Context::Default, expr_arena)
            .unwrap();

        // we check if predicates can be done on the input above
        // this can only be done if the current projection is not a projection boundary
        let is_boundary = projection_is_definite_pushdown_boundary(*projection_node, expr_arena);

        // remove predicates that cannot be done on the input above
        let to_local = acc_predicates
            .iter()
            .filter_map(|(name, predicate)| {
                // there are some conditions we need to check for every predicate we try to push down
                // 1. does the column exist on the node above
                // 2. if the projection is a computation/transformation and the predicate is based on that column
                //    we must block because the predicate would be incorrect.
                // 3. if applying the predicate earlier does not influence the result of this projection
                //    this is the case for instance with a sum operation (filtering out rows influences the result)

                // checks 1.
                if check_input_node(*predicate, &input_schema, expr_arena)
                // checks 2.
                && !(key_has_name(name, output_field.name()) && projection_maybe_boundary)
                // checks 3.
                && !is_boundary
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

/// Transfer a predicate from `acc_predicates` that will be pushed down
/// to a local_predicates vec based on a condition.
pub(super) fn transfer_to_local_by_node<F>(
    acc_predicates: &mut PlHashMap<Arc<str>, Node>,
    mut condition: F,
) -> Vec<Node>
where
    F: FnMut(Node) -> bool,
{
    let mut remove_keys = Vec::with_capacity(acc_predicates.len());

    for (key, predicate) in &*acc_predicates {
        if condition(*predicate) {
            remove_keys.push(key.clone());
            continue;
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

/// predicates that need the full context should not be pushed down to the scans
/// example: min(..) == null_count
pub(super) fn partition_by_full_context(
    acc_predicates: &mut PlHashMap<Arc<str>, Node>,
    expr_arena: &Arena<AExpr>,
) -> Vec<Node> {
    transfer_to_local_by_node(acc_predicates, |node| {
        has_aexpr(node, expr_arena, |ae| match ae {
            AExpr::BinaryExpr { left, right, .. } => {
                expr_arena.get(*left).groups_sensitive()
                    || expr_arena.get(*right).groups_sensitive()
            }
            ae => ae.groups_sensitive(),
        })
    })
}
