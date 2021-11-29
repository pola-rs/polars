use crate::logical_plan::iterator::ArenaExprIter;
use crate::logical_plan::Context;
use crate::prelude::*;
use crate::utils::{
    aexpr_to_root_column_name, aexpr_to_root_names, check_input_node, has_aexpr,
    rename_aexpr_root_name,
};
use polars_core::datatypes::PlHashMap;
use polars_core::prelude::*;

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
    name: Arc<str>,
    predicate: Node,
    arena: &mut Arena<AExpr>,
) {
    let existing_predicate = acc_predicates
        .entry(name)
        .or_insert_with(|| arena.add(AExpr::Literal(LiteralValue::Boolean(true))));

    let node = arena.add(AExpr::BinaryExpr {
        left: predicate,
        op: Operator::And,
        right: *existing_predicate,
    });

    *existing_predicate = node;
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

/// Determine the hashmap key by combining all the root column names of a predicate
pub(super) fn roots_to_key(roots: &[Arc<str>]) -> Arc<str> {
    if roots.len() == 1 {
        roots[0].clone()
    } else {
        let mut new = String::with_capacity(32 * roots.len());
        for name in roots {
            new.push_str(name);
        }
        Arc::from(new)
    }
}

pub(super) fn get_insertion_name(
    expr_arena: &Arena<AExpr>,
    predicate: Node,
    schema: &Schema,
) -> Arc<str> {
    Arc::from(
        expr_arena
            .get(predicate)
            .to_field(schema, Context::Default, expr_arena)
            .unwrap()
            .name()
            .as_ref(),
    )
}

/// Some predicates should not pass a projection if they would influence results of other columns.
/// For instance shifts | sorts results are influenced by a filter so we do all predicates before the shift | sort
/// The rule of thumb is any operation that changes the order of a column w/r/t other columns should be a
/// predicate pushdown blocker.
pub(super) fn is_pushdown_boundary(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    let matches = |e: &AExpr| {
        matches!(
            e,
            AExpr::Shift { .. } | AExpr::Sort { .. } | AExpr::SortBy { .. }
            | AExpr::Agg(_) // an aggregation needs all rows
            | AExpr::Reverse(_)
            // everything that works on groups likely changes to order of elements w/r/t the other columns
            | AExpr::Function {options: FunctionOptions { collect_groups: ApplyOptions::ApplyGroups, .. }, ..}
            | AExpr::Function {options: FunctionOptions { collect_groups: ApplyOptions::ApplyList, .. }, ..}
            // Could be fine, could be not, for now let's be conservative on this one
            | AExpr::BinaryFunction {..}
            // still need to investigate this one
            | AExpr::Explode {..}
            // A groupby needs all rows for aggregation
            | AExpr::Window {..}
            | AExpr::Literal(LiteralValue::Range {..})
        ) ||
            // a series that is not a singleton would also have a different result
            // if filter is applied earlier
            matches!(e, AExpr::Literal(LiteralValue::Series(s)) if s.len() > 1
        )
    };
    has_aexpr(node, expr_arena, matches)
}

/// Implementation for both Hstack and Projection
pub(super) fn rewrite_projection_node(
    expr_arena: &mut Arena<AExpr>,
    lp_arena: &mut Arena<ALogicalPlan>,
    acc_predicates: &mut PlHashMap<Arc<str>, Node>,
    expr: Vec<Node>,
    input: Node,
) -> (Vec<Node>, Vec<Node>)
where
{
    let mut local_predicates = Vec::with_capacity(acc_predicates.len());

    // maybe update predicate name if a projection is an alias
    // aliases change the column names and because we push the predicates downwards
    // this may be problematic as the aliased column may not yet exist.
    for node in &expr {
        {
            let e = expr_arena.get(*node);
            if let AExpr::Alias(e, name) = e {
                // if this alias refers to one of the predicates in the upper nodes
                // we rename the column of the predicate before we push it downwards.
                if let Some(predicate) = acc_predicates.remove(&*name) {
                    match aexpr_to_root_column_name(*e, &*expr_arena) {
                        // we were able to rename the alias column with the root column name
                        // before pushing down the predicate
                        Ok(new_name) => {
                            rename_aexpr_root_name(predicate, expr_arena, new_name.clone())
                                .unwrap();

                            insert_and_combine_predicate(
                                acc_predicates,
                                new_name,
                                predicate,
                                expr_arena,
                            );
                        }
                        // this may be a complex binary function. The predicate may only be valid
                        // on this projected column so we do filter locally.
                        Err(_) => local_predicates.push(predicate),
                    }
                }
            }
        }

        let e = expr_arena.get(*node);
        let input_schema = lp_arena.get(input).schema(lp_arena);

        // we check if predicates can be done on the input above
        // with the following conditions:

        // 1. predicate based on current column may only pushed down if simple projection, e.g. col() / col().alias()
        let expr_depth = (&*expr_arena).iter(*node).count();
        let is_computation = if let AExpr::Alias(_, _) = e {
            expr_depth > 2
        } else {
            expr_depth > 1
        };

        // remove predicates that cannot be done on the input above
        let to_local = acc_predicates
            .iter()
            .filter_map(|kv| {
                // if they can be executed on input node above its ok
                if check_input_node(*kv.1, input_schema, expr_arena)
                    // if this predicate not equals a column that is a computation
                    // it is ok
                    && !is_computation
                {
                    None
                } else {
                    Some(kv.0.clone())
                }
            })
            .collect::<Vec<_>>();

        for name in to_local {
            let local = acc_predicates.remove(&name).unwrap();
            local_predicates.push(local);
        }

        // remove predicates that are based on column modifications
        no_pushdown_preds(
            *node,
            expr_arena,
            |e| matches!(e, AExpr::Explode(_)) || matches!(e, AExpr::Ternary { .. }),
            &mut local_predicates,
            acc_predicates,
        );
    }
    (local_predicates, expr)
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
        let columns = aexpr_to_root_names(node, arena);

        let condition = |name: Arc<str>| columns.contains(&name);
        local_predicates.extend(transfer_to_local(arena, acc_predicates, condition));
    }
}

/// Transfer a predicate from `acc_predicates` that will be pushed down
/// to a local_predicates vec based on a condition.
pub(super) fn transfer_to_local<F>(
    expr_arena: &Arena<AExpr>,
    acc_predicates: &mut PlHashMap<Arc<str>, Node>,
    mut condition: F,
) -> Vec<Node>
where
    F: FnMut(Arc<str>) -> bool,
{
    let mut remove_keys = Vec::with_capacity(acc_predicates.len());

    for (key, predicate) in &*acc_predicates {
        let root_names = aexpr_to_root_names(*predicate, expr_arena);
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
