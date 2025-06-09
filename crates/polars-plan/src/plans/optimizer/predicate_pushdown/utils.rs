use polars_core::prelude::*;
use polars_utils::idx_vec::UnitVec;

use super::keys::*;
use crate::plans::visitor::{AexprNode, RewriteRecursion, RewritingVisitor, TreeWalker};
use crate::prelude::*;
fn combine_by_and(left: Node, right: Node, arena: &mut Arena<AExpr>) -> Node {
    arena.add(AExpr::BinaryExpr {
        left,
        op: Operator::And,
        right,
    })
}

/// Don't overwrite predicates but combine them.
pub(super) fn insert_and_combine_predicate(
    acc_predicates: &mut PlHashMap<PlSmallStr, ExprIR>,
    predicate: &ExprIR,
    arena: &mut Arena<AExpr>,
) {
    let name = predicate_to_key(predicate.node(), arena);

    acc_predicates
        .entry(name)
        .and_modify(|existing_predicate| {
            let node = combine_by_and(predicate.node(), existing_predicate.node(), arena);
            existing_predicate.set_node(node)
        })
        .or_insert_with(|| predicate.clone());
}

pub(super) fn temporary_unique_key(acc_predicates: &PlHashMap<PlSmallStr, ExprIR>) -> PlSmallStr {
    // TODO: Don't heap allocate during construction.
    let mut out_key = '\u{1D17A}'.to_string();
    let mut existing_keys = acc_predicates.keys();

    while acc_predicates.contains_key(&*out_key) {
        out_key.push_str(existing_keys.next().unwrap());
    }

    PlSmallStr::from_string(out_key)
}

pub(super) fn combine_predicates<I>(iter: I, arena: &mut Arena<AExpr>) -> ExprIR
where
    I: Iterator<Item = ExprIR>,
{
    let mut single_pred = None;
    for e in iter {
        single_pred = match single_pred {
            None => Some(e.node()),
            Some(left) => Some(arena.add(AExpr::BinaryExpr {
                left,
                op: Operator::And,
                right: e.node(),
            })),
        };
    }
    single_pred
        .map(|node| ExprIR::from_node(node, arena))
        .expect("an empty iterator was passed")
}

pub(super) fn predicate_at_scan(
    acc_predicates: PlHashMap<PlSmallStr, ExprIR>,
    predicate: Option<ExprIR>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<ExprIR> {
    if !acc_predicates.is_empty() {
        let mut new_predicate = combine_predicates(acc_predicates.into_values(), expr_arena);
        if let Some(pred) = predicate {
            new_predicate.set_node(combine_by_and(
                new_predicate.node(),
                pred.node(),
                expr_arena,
            ));
        }
        Some(new_predicate)
    } else {
        None
    }
}

/// Evaluates a condition on the column name inputs of every predicate, where if
/// the condition evaluates to true on any column name the predicate is
/// transferred to local.
pub(super) fn transfer_to_local_by_expr_ir<F>(
    expr_arena: &Arena<AExpr>,
    acc_predicates: &mut PlHashMap<PlSmallStr, ExprIR>,
    mut condition: F,
) -> Vec<ExprIR>
where
    F: FnMut(&ExprIR) -> bool,
{
    let mut remove_keys = Vec::with_capacity(acc_predicates.len());

    for predicate in acc_predicates.values() {
        if condition(predicate) {
            if let Some(name) = aexpr_to_leaf_names_iter(predicate.node(), expr_arena).next() {
                remove_keys.push(name);
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

/// Evaluates a condition on the column name inputs of every predicate, where if
/// the condition evaluates to true on any column name the predicate is
/// transferred to local.
pub(super) fn transfer_to_local_by_name<F>(
    expr_arena: &Arena<AExpr>,
    acc_predicates: &mut PlHashMap<PlSmallStr, ExprIR>,
    mut condition: F,
) -> Vec<ExprIR>
where
    F: FnMut(&PlSmallStr) -> bool,
{
    let mut remove_keys = Vec::with_capacity(acc_predicates.len());

    for (key, predicate) in &*acc_predicates {
        let root_names = aexpr_to_leaf_names_iter(predicate.node(), expr_arena);
        for name in root_names {
            if condition(&name) {
                remove_keys.push(key.clone());
                break;
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

/// * `col(A).alias(B).alias(C) => (C, A)`
/// * `col(A)                   => (A, A)`
/// * `col(A).sum().alias(B)    => None`
fn get_maybe_aliased_projection_to_input_name_map(
    e: &ExprIR,
    expr_arena: &Arena<AExpr>,
) -> Option<(PlSmallStr, PlSmallStr)> {
    let ae = expr_arena.get(e.node());
    match e.get_alias() {
        Some(alias) => match ae {
            AExpr::Column(c_name) => Some((alias.clone(), c_name.clone())),
            _ => None,
        },
        _ => match ae {
            AExpr::Column(c_name) => Some((c_name.clone(), c_name.clone())),
            _ => None,
        },
    }
}

#[derive(Debug)]
pub enum PushdownEligibility {
    Full,
    // Partial can happen when there are window exprs.
    Partial { to_local: Vec<PlSmallStr> },
    NoPushdown,
}

#[allow(clippy::type_complexity)]
pub fn pushdown_eligibility(
    projection_nodes: &[ExprIR],
    // Predicates that need to be checked (key, expr_ir)
    new_predicates: &[(&PlSmallStr, ExprIR)],
    // Note: These predicates have already passed checks.
    acc_predicates: &PlHashMap<PlSmallStr, ExprIR>,
    expr_arena: &mut Arena<AExpr>,
    scratch: &mut UnitVec<Node>,
    maintain_errors: bool,
    input_ir: &IR,
) -> PolarsResult<(PushdownEligibility, PlHashMap<PlSmallStr, PlSmallStr>)> {
    scratch.clear();
    let ae_nodes_stack = scratch;

    let mut alias_to_col_map =
        optimizer::init_hashmap::<PlSmallStr, PlSmallStr>(Some(projection_nodes.len()));
    let mut col_to_alias_map = alias_to_col_map.clone();

    let mut modified_projection_columns =
        PlHashSet::<PlSmallStr>::with_capacity(projection_nodes.len());
    let mut has_window = false;
    let mut common_window_inputs = PlHashSet::<PlSmallStr>::new();

    // Important: Names inserted into any data structure by this function are
    // all non-aliased.
    // This function returns false if pushdown cannot be performed.
    let process_projection_or_predicate = |ae_nodes_stack: &mut UnitVec<Node>,
                                           has_window: &mut bool,
                                           common_window_inputs: &mut PlHashSet<PlSmallStr>|
     -> ExprPushdownGroup {
        debug_assert_eq!(ae_nodes_stack.len(), 1);

        let mut partition_by_names = PlHashSet::<PlSmallStr>::new();
        let mut expr_pushdown_eligibility = ExprPushdownGroup::Pushable;

        while let Some(node) = ae_nodes_stack.pop() {
            let ae = expr_arena.get(node);

            match ae {
                AExpr::Window {
                    partition_by,
                    #[cfg(feature = "dynamic_group_by")]
                    options,
                    // The function is not checked for groups-sensitivity because
                    // it is applied over the windows.
                    ..
                } => {
                    #[cfg(feature = "dynamic_group_by")]
                    if matches!(options, WindowType::Rolling(..)) {
                        return ExprPushdownGroup::Barrier;
                    };

                    partition_by_names.clear();
                    partition_by_names.reserve(partition_by.len());

                    for node in partition_by.iter() {
                        // Only accept col()
                        if let AExpr::Column(name) = expr_arena.get(*node) {
                            partition_by_names.insert(name.clone());
                        } else {
                            // Nested windows can also qualify for push down.
                            // e.g.:
                            // * expr1 = min().over(A)
                            // * expr2 = sum().over(A, expr1)
                            // Both exprs window over A, so predicates referring
                            // to A can still be pushed.
                            ae_nodes_stack.push(*node);
                        }
                    }

                    if !*has_window {
                        for name in partition_by_names.drain() {
                            common_window_inputs.insert(name);
                        }

                        *has_window = true;
                    } else {
                        common_window_inputs.retain(|k| partition_by_names.contains(k))
                    }

                    // Cannot push into disjoint windows:
                    // e.g.:
                    // * sum().over(A)
                    // * sum().over(B)
                    if common_window_inputs.is_empty() {
                        return ExprPushdownGroup::Barrier;
                    }
                },
                _ => {
                    if let ExprPushdownGroup::Barrier =
                        expr_pushdown_eligibility.update_with_expr(ae_nodes_stack, ae, expr_arena)
                    {
                        return ExprPushdownGroup::Barrier;
                    }
                },
            }
        }

        expr_pushdown_eligibility
    };

    for e in projection_nodes.iter() {
        if let Some((alias, column_name)) =
            get_maybe_aliased_projection_to_input_name_map(e, expr_arena)
        {
            if alias != column_name {
                alias_to_col_map.insert(alias.clone(), column_name.clone());
                col_to_alias_map.insert(column_name, alias);
            }
            continue;
        }

        if !does_not_modify_rec(e.node(), expr_arena) {
            modified_projection_columns.insert(e.output_name().clone());
        }

        debug_assert!(ae_nodes_stack.is_empty());
        ae_nodes_stack.push(e.node());

        if process_projection_or_predicate(
            ae_nodes_stack,
            &mut has_window,
            &mut common_window_inputs,
        )
        .blocks_pushdown(maintain_errors)
        {
            return Ok((PushdownEligibility::NoPushdown, alias_to_col_map));
        }
    }

    if has_window && !col_to_alias_map.is_empty() {
        // Rename to aliased names.
        let mut new = PlHashSet::<PlSmallStr>::with_capacity(2 * common_window_inputs.len());

        for key in common_window_inputs.into_iter() {
            if let Some(aliased) = col_to_alias_map.get(&key) {
                new.insert(aliased.clone());
            }
            // Ensure predicate does not refer to a different column that
            // got aliased to the same name as the window column. E.g.:
            // .with_columns(col(A).alias(C), sum=sum().over(C))
            // .filter(col(C) == ..)
            if !alias_to_col_map.contains_key(&key) {
                new.insert(key);
            }
        }

        if new.is_empty() {
            return Ok((PushdownEligibility::NoPushdown, alias_to_col_map));
        }

        common_window_inputs = new;
    }

    for (_, e) in new_predicates.iter() {
        debug_assert!(ae_nodes_stack.is_empty());
        ae_nodes_stack.push(e.node());

        let pd_group = process_projection_or_predicate(
            ae_nodes_stack,
            &mut has_window,
            &mut common_window_inputs,
        );

        if pd_group.blocks_pushdown(maintain_errors) {
            return Ok((PushdownEligibility::NoPushdown, alias_to_col_map));
        }
    }

    // Should have returned early.
    debug_assert!(!common_window_inputs.is_empty() || !has_window);

    // Note: has_window is constant.
    let can_use_column = |col: &str| {
        if has_window {
            common_window_inputs.contains(col)
        } else {
            !modified_projection_columns.contains(col)
        }
    };

    // For an allocation-free dyn iterator
    let mut check_predicates_all: Option<_> = None;
    let mut check_predicates_only_new: Option<_> = None;

    // We only need to check the new predicates if no columns were renamed and there are no window
    // aggregations.
    if !has_window
        && modified_projection_columns.is_empty()
        && !(
            // If there is only a single predicate, it may be fallible
            acc_predicates.len() == 1 && ir_removes_rows(input_ir)
        )
    {
        check_predicates_only_new = Some(new_predicates.iter().map(|(key, expr)| (*key, expr)))
    } else {
        check_predicates_all = Some(acc_predicates.iter())
    }

    let to_check_iter: &mut dyn Iterator<Item = (&PlSmallStr, &ExprIR)> = check_predicates_all
        .as_mut()
        .map(|x| x as _)
        .unwrap_or_else(|| check_predicates_only_new.as_mut().map(|x| x as _).unwrap());

    let mut allow_single_fallible = !ir_removes_rows(input_ir);
    ae_nodes_stack.clear();

    let to_local = to_check_iter
        .filter_map(|(key, e)| {
            debug_assert!(ae_nodes_stack.is_empty());

            ae_nodes_stack.push(e.node());

            let mut uses_blocked_name = false;
            let mut pd_group = ExprPushdownGroup::Pushable;

            while let Some(node) = ae_nodes_stack.pop() {
                let ae = expr_arena.get(node);

                if let AExpr::Column(name) = ae {
                    uses_blocked_name |= !can_use_column(name);
                } else {
                    pd_group.update_with_expr(ae_nodes_stack, ae, expr_arena);
                };

                if uses_blocked_name {
                    break;
                };
            }

            ae_nodes_stack.clear();

            if uses_blocked_name || matches!(pd_group, ExprPushdownGroup::Barrier) {
                allow_single_fallible = false;
            }

            if uses_blocked_name
                || matches!(
                    // Note: We do not use `blocks_pushdown()`, this fallible indicates that the
                    // predicate we are checking to push is fallible.
                    pd_group,
                    ExprPushdownGroup::Fallible | ExprPushdownGroup::Barrier
                )
            {
                Some(key.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    Ok(match to_local.len() {
        0 => (PushdownEligibility::Full, alias_to_col_map),
        len if len == acc_predicates.len() => {
            if len == 1 && allow_single_fallible {
                (PushdownEligibility::Full, alias_to_col_map)
            } else {
                (PushdownEligibility::NoPushdown, alias_to_col_map)
            }
        },
        _ => (PushdownEligibility::Partial { to_local }, alias_to_col_map),
    })
}

/// Note: This may give false positives as it is a conservative function.
pub(crate) fn ir_removes_rows(ir: &IR) -> bool {
    use IR::*;

    // NOTE
    // At time of writing predicate pushdown runs before slice pushdown, so
    // some of the below checks for slice may never be hit.

    match ir {
        DataFrameScan { .. }
        | SimpleProjection { .. }
        | Select { .. }
        | Cache { .. }
        | HStack { .. }
        | HConcat { .. } => false,

        GroupBy { options, .. } => options.slice.is_some(),

        Sort { slice, .. } => slice.is_some(),

        #[cfg(feature = "merge_sorted")]
        MergeSorted { .. } => false,

        #[cfg(feature = "python")]
        PythonScan { options } => options.n_rows.is_some(),

        // Scan currently may evaluate the predicate on the statistics of the
        // entire files list.
        Scan {
            unified_scan_args, ..
        } => unified_scan_args.pre_slice.is_some(),

        _ => true,
    }
}

/// Maps column references within an expression. Used to handle column renaming when pushing
/// predicates.
pub(super) fn map_column_references(
    expr: &mut ExprIR,
    expr_arena: &mut Arena<AExpr>,
    rename_map: &PlHashMap<PlSmallStr, PlSmallStr>,
) {
    if rename_map.is_empty() {
        return;
    }

    let node = AexprNode::new(expr.node())
        .rewrite(
            &mut MapColumnReferences {
                rename_map,
                column_nodes: PlHashMap::with_capacity(rename_map.len()),
            },
            expr_arena,
        )
        .unwrap()
        .node();

    expr.set_node(node);

    struct MapColumnReferences<'a> {
        rename_map: &'a PlHashMap<PlSmallStr, PlSmallStr>,
        column_nodes: PlHashMap<&'a str, Node>,
    }

    impl RewritingVisitor for MapColumnReferences<'_> {
        type Node = AexprNode;
        type Arena = Arena<AExpr>;

        fn pre_visit(
            &mut self,
            node: &Self::Node,
            arena: &mut Self::Arena,
        ) -> polars_core::prelude::PolarsResult<crate::prelude::visitor::RewriteRecursion> {
            let AExpr::Column(colname) = arena.get(node.node()) else {
                return Ok(RewriteRecursion::NoMutateAndContinue);
            };

            if !self.rename_map.contains_key(colname) {
                return Ok(RewriteRecursion::NoMutateAndContinue);
            }

            Ok(RewriteRecursion::MutateAndContinue)
        }

        fn mutate(
            &mut self,
            node: Self::Node,
            arena: &mut Self::Arena,
        ) -> polars_core::prelude::PolarsResult<Self::Node> {
            let AExpr::Column(colname) = arena.get(node.node()) else {
                unreachable!();
            };

            let new_colname = self.rename_map.get(colname).unwrap();

            if !self.column_nodes.contains_key(new_colname.as_str()) {
                self.column_nodes.insert(
                    new_colname.as_str(),
                    arena.add(AExpr::Column(new_colname.clone())),
                );
            }

            // Safety: Checked in pre_visit()
            Ok(AexprNode::new(
                *self.column_nodes.get(new_colname.as_str()).unwrap(),
            ))
        }
    }
}
