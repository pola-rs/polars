use polars_core::datatypes::PlHashMap;
use polars_core::prelude::*;

use super::keys::*;
use crate::logical_plan::Context;
use crate::prelude::*;
use crate::utils::{aexpr_to_leaf_names, has_aexpr};

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
            // function that match this are `cum_sum`, `shift`, `sort`, etc.
            options.is_groups_sensitive() && !options.returns_scalar
        },
        _ => false,
    };
    has_aexpr(node, expr_arena, matches)
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

fn check_and_extend_predicate_pd_nodes(
    stack: &mut Vec<Node>,
    ae: &AExpr,
    expr_arena: &Arena<AExpr>,
) -> bool {
    if match ae {
        // These literals do not come from the RHS of an is_in, meaning that
        // they are projected as either columns or predicates, both of which
        // rely on the height of the dataframe at this level and thus need
        // to block pushdown.
        AExpr::Literal(lit) => !lit.projects_as_scalar(),
        ae => ae.groups_sensitive(),
    } {
        false
    } else {
        match ae {
            #[cfg(feature = "is_in")]
            AExpr::Function {
                function: FunctionExpr::Boolean(BooleanFunction::IsIn),
                input,
                ..
            } => {
                // Handles a special case where the expr contains a series, but it is being
                // used as part the RHS of an `is_in`, so it can be pushed down as it is not
                // being projected.
                let mut transferred_local_nodes = false;
                if let Some(rhs) = input.get(1) {
                    if matches!(expr_arena.get(*rhs), AExpr::Literal { .. }) {
                        let mut local_nodes = Vec::<Node>::with_capacity(4);
                        ae.nodes(&mut local_nodes);

                        stack.extend(local_nodes.into_iter().filter(|node| node != rhs));
                        transferred_local_nodes = true;
                    }
                };
                if !transferred_local_nodes {
                    ae.nodes(stack);
                }
            },
            ae => {
                ae.nodes(stack);
            },
        };
        true
    }
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
pub(super) fn aexpr_blocks_predicate_pushdown(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    let mut stack = Vec::<Node>::with_capacity(4);
    stack.push(node);

    // Cannot use `has_aexpr` because we need to ignore any literals in the RHS
    // of an `is_in` operation.
    while let Some(node) = stack.pop() {
        let ae = expr_arena.get(node);

        if !check_and_extend_predicate_pd_nodes(&mut stack, ae, expr_arena) {
            return true;
        }
    }
    false
}

/// * `col(A).alias(B).alias(C) => (C, A)`
/// * `col(A)                   => (A, A)`
/// * `col(A).sum().alias(B)    => None`
fn get_maybe_aliased_projection_to_input_name_map(
    node: Node,
    expr_arena: &Arena<AExpr>,
) -> Option<(Arc<str>, Arc<str>)> {
    let mut curr_node = node;
    let mut curr_alias: Option<Arc<str>> = None;

    loop {
        match expr_arena.get(curr_node) {
            AExpr::Alias(node, alias) => {
                if curr_alias.is_none() {
                    curr_alias = Some(alias.clone());
                }

                curr_node = *node;
            },
            AExpr::Column(name) => {
                return if let Some(alias) = curr_alias {
                    Some((alias, name.clone()))
                } else {
                    Some((name.clone(), name.clone()))
                }
            },
            _ => break,
        }
    }

    None
}

/// This function returns None if predicates cannot be pushed. Otherwise, it
/// returns:
/// * A function to determine if a column used by a predicate can be used
///   in the upper node.
/// * A mapping from aliased names to the column names in the upper schema.
#[allow(clippy::type_complexity)]
pub fn get_column_allowed_checker_and_rename_map(
    input_schema: Arc<Schema>,
    projection_nodes: &Vec<Node>,
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<
    Option<(
        Box<dyn Fn(&Arc<str>) -> bool>,
        PlHashMap<Arc<str>, Arc<str>>,
    )>,
> {
    let mut ae_nodes_stack = Vec::<Node>::with_capacity(4);
    let mut pushdown_rename_map =
        optimizer::init_hashmap::<Arc<str>, Arc<str>>(Some(projection_nodes.len()));

    let mut modified_projection_columns =
        PlHashSet::<Arc<str>>::with_capacity(projection_nodes.len());
    let mut common_window_inputs: Option<PlHashSet<Arc<str>>> = None;

    for projection_node in projection_nodes.iter() {
        if let Some((alias, column_name)) =
            get_maybe_aliased_projection_to_input_name_map(*projection_node, expr_arena)
        {
            if alias != column_name {
                pushdown_rename_map.insert(alias, column_name);
            }
            continue;
        }

        modified_projection_columns.insert(Arc::<str>::from(
            expr_arena
                .get(*projection_node)
                .to_field(&input_schema, Context::Default, expr_arena)?
                .name()
                .as_str(),
        ));

        ae_nodes_stack.push(*projection_node);

        while let Some(node) = ae_nodes_stack.pop() {
            let ae = expr_arena.get(node);

            match ae {
                AExpr::Window {
                    partition_by,
                    #[cfg(feature = "dynamic_group_by")]
                    options,
                    ..
                } => {
                    #[cfg(feature = "dynamic_group_by")]
                    if matches!(options, WindowType::Rolling(..)) {
                        return Ok(None);
                    };

                    let mut partition_by_names =
                        PlHashSet::<Arc<str>>::with_capacity(partition_by.len());

                    for node in partition_by.iter() {
                        // Only accept col() or col().alias()
                        if let Some((_, name)) =
                            get_maybe_aliased_projection_to_input_name_map(*node, expr_arena)
                        {
                            partition_by_names.insert(name.clone());
                        } else {
                            // This needs to be checked for groups-sensitivity.
                            // e.g.:
                            // * sum().over(col(A).sum().over(..))
                            if aexpr_blocks_predicate_pushdown(*node, expr_arena) {
                                return Ok(None);
                            }
                        }
                    }

                    // Cannot push into disjoint windows:
                    // e.g.:
                    // * sum().over(A)
                    // * sum().over(B)
                    if let Some(ref mut inputs) = common_window_inputs {
                        inputs.retain(|k| partition_by_names.contains(k))
                    } else {
                        common_window_inputs = Some(partition_by_names);
                    }

                    if common_window_inputs.as_ref().unwrap().is_empty() {
                        return Ok(None);
                    }
                },
                _ => {
                    if !check_and_extend_predicate_pd_nodes(&mut ae_nodes_stack, ae, expr_arena) {
                        return Ok(None);
                    }
                },
            }
        }
    }

    if let Some(common_window_inputs) = common_window_inputs {
        // Rename column names in column window inputs to any potential aliases.
        let column_name_to_alias_map = pushdown_rename_map
            .iter()
            .map(|(k, v)| (v.clone(), k.clone()))
            .collect::<PlHashMap<Arc<str>, Arc<str>>>();

        let common_window_inputs = common_window_inputs
            .into_iter()
            .flat_map(|key| {
                let mut out = Vec::<Arc<str>>::with_capacity(2);

                if let Some(aliased) = column_name_to_alias_map.get(&key) {
                    out.push(aliased.clone())
                }

                // Ensure predicate does not refer to a different column that
                // got aliased to the same name as the window column. E.g.:
                // .with_columns(col(A).alias(C), sum=sum().over(C))
                // .filter(col(C) == ..)
                if !pushdown_rename_map.contains_key(&key) {
                    out.push(key)
                };

                out
            })
            .collect::<PlHashSet<Arc<str>>>();

        if common_window_inputs.is_empty() {
            Ok(None)
        } else {
            Ok(Some((
                Box::new(move |name| common_window_inputs.contains(name)),
                pushdown_rename_map,
            )))
        }
    } else {
        Ok(Some((
            Box::new(move |name| !modified_projection_columns.contains(name)),
            pushdown_rename_map,
        )))
    }
}

/// Used in places that previously handled blocking exprs before refactoring.
/// Can probably be eventually removed if it isn't catching anything.
pub(super) fn debug_assert_aexpr_allows_predicate_pushdown(node: Node, expr_arena: &Arena<AExpr>) {
    debug_assert!(
        !aexpr_blocks_predicate_pushdown(node, expr_arena),
        "Predicate pushdown: Did not expect blocking exprs at this point, please open an issue."
    );
}
