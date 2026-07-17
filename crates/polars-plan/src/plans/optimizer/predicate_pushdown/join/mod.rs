use polars_utils::format_pl_smallstr;

use super::*;
mod hive;
mod predicate_pruning;
use hive::rewrite_hive;
use predicate_pruning::*;

use crate::plans::optimizer::join_utils::remove_suffix;

const IEJOIN_MAX_PREDICATES: usize = 2;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_join(
    opt: &mut PredicatePushDown,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    mut input_left: Node,
    mut input_right: Node,
    mut left_on: Vec<ExprIR>,
    mut right_on: Vec<ExprIR>,
    mut schema: SchemaRef,
    mut options: Arc<JoinOptionsIR>,
    mut acc_predicates: PlIndexMap<PlSmallStr, ExprIR>,
    streaming: bool,
) -> PolarsResult<IR> {
    if options.args.slice.is_some() {
        let ir = rewrite_hive(
            input_left,
            input_right,
            left_on,
            right_on,
            schema,
            options,
            opt,
            lp_arena,
            expr_arena,
        )?;
        // Ensures we don't trigger the hive rewrite again.
        if matches!(ir, IR::Union { .. }) {
            opt.hive_rewrite_active = true;
        }
        let result = opt.no_pushdown_restart_opt(ir, acc_predicates, lp_arena, expr_arena);
        return result;
    }

    let schema_left = lp_arena.get(input_left).schema(lp_arena).into_owned();
    let schema_right = lp_arena.get(input_right).schema(lp_arena).into_owned();

    let mut opt_join_key_reduction_select = try_reduce_redundant_join_keys(
        opt,
        lp_arena,
        expr_arena,
        &mut input_left,
        &mut input_right,
        &schema_left,
        &schema_right,
        &mut schema,
        &options,
        &mut left_on,
        &mut right_on,
    )?;

    let opt_post_select = try_rewrite_join_type(
        &schema_left,
        &schema_right,
        &mut schema,
        &mut options,
        &mut left_on,
        &mut right_on,
        &mut acc_predicates,
        expr_arena,
        streaming,
    )?;

    if match &options.args.how {
        // Full-join with no coalesce. We can only push filters if they do not remove NULLs, but
        // we don't have a reliable way to guarantee this.
        JoinType::Full => !options.args.should_coalesce(),

        _ => false,
    } || acc_predicates.is_empty()
    {
        let lp = rewrite_hive(
            input_left,
            input_right,
            left_on,
            right_on,
            schema,
            options,
            opt,
            lp_arena,
            expr_arena,
        )?;

        // See the comment on the analogous guard above (including why this must be a
        // save/restore, not an unconditional reset): `rewrite_hive`'s branch joins may end
        // up nested a level deeper here (under the optional post-select), but they're still
        // reachable from `no_pushdown_restart_opt`'s re-descent, so the guard still needs to
        // span that call.
        let rewrote_to_union = matches!(lp, IR::Union { .. });
        let lp =
            apply_join_key_reduction_select(lp, opt_join_key_reduction_select.take(), lp_arena);

        if rewrote_to_union {
            opt.hive_rewrite_active = true;
        }
        let result = opt.no_pushdown_restart_opt(lp, acc_predicates, lp_arena, expr_arena);
        return result;
    }

    let should_coalesce = options.args.should_coalesce();

    // AsOf has the equality join keys under `asof_options.left/right_by`. This code builds an
    // iterator to address these generically without creating a `Box<dyn Iterator>`.
    let get_lhs_column_keys_iter = || {
        let len = match &options.args.how {
            #[cfg(feature = "asof_join")]
            JoinType::AsOf(asof_options) => {
                asof_options.left_by.as_deref().unwrap_or_default().len()
            },
            _ => left_on.len(),
        };

        (0..len).map(|i| match &options.args.how {
            #[cfg(feature = "asof_join")]
            JoinType::AsOf(asof_options) => Some(
                asof_options
                    .left_by
                    .as_deref()
                    .unwrap_or_default()
                    .get(i)
                    .unwrap(),
            ),
            _ => {
                let expr = left_on.get(i).unwrap();

                // For non full-joins coalesce can still insert casts into the key exprs.
                let node = match expr_arena.get(expr.node()) {
                    AExpr::Cast {
                        expr,
                        dtype: _,
                        options: _,
                    } if should_coalesce => *expr,

                    _ => expr.node(),
                };

                if let AExpr::Column(name) = expr_arena.get(node) {
                    Some(name)
                } else {
                    None
                }
            },
        })
    };

    let get_rhs_column_keys_iter = || {
        let len = match &options.args.how {
            #[cfg(feature = "asof_join")]
            JoinType::AsOf(asof_options) => {
                asof_options.right_by.as_deref().unwrap_or_default().len()
            },
            _ => right_on.len(),
        };

        (0..len).map(|i| match &options.args.how {
            #[cfg(feature = "asof_join")]
            JoinType::AsOf(asof_options) => Some(
                asof_options
                    .right_by
                    .as_deref()
                    .unwrap_or_default()
                    .get(i)
                    .unwrap(),
            ),
            _ => {
                let expr = right_on.get(i).unwrap();

                // For non full-joins coalesce can still insert casts into the key exprs.
                let node = match expr_arena.get(expr.node()) {
                    AExpr::Cast {
                        expr,
                        dtype: _,
                        options: _,
                    } if should_coalesce => *expr,

                    _ => expr.node(),
                };

                if let AExpr::Column(name) = expr_arena.get(node) {
                    Some(name)
                } else {
                    None
                }
            },
        })
    };

    if cfg!(debug_assertions) && options.args.should_coalesce() {
        match &options.args.how {
            #[cfg(feature = "asof_join")]
            JoinType::AsOf(_) => {},

            _ => {
                assert!(get_lhs_column_keys_iter().len() > 0);
                assert!(get_rhs_column_keys_iter().len() > 0);
            },
        }

        assert!(get_lhs_column_keys_iter().all(|x| x.is_some()));
        assert!(get_rhs_column_keys_iter().all(|x| x.is_some()));
    }

    // Key columns of the left table that are coalesced into an output column of the right table.
    let coalesced_to_right: PlIndexSet<PlSmallStr> =
        if matches!(&options.args.how, JoinType::Right) && options.args.should_coalesce() {
            get_lhs_column_keys_iter()
                .map(|x| x.unwrap().clone())
                .collect()
        } else {
            Default::default()
        };

    let mut output_key_to_left_input_map: PlIndexMap<PlSmallStr, PlSmallStr> =
        PlIndexMap::with_capacity(get_lhs_column_keys_iter().len());
    let mut output_key_to_right_input_map: PlIndexMap<PlSmallStr, PlSmallStr> =
        PlIndexMap::with_capacity(get_rhs_column_keys_iter().len());

    for (lhs_input_key, rhs_input_key) in get_lhs_column_keys_iter().zip(get_rhs_column_keys_iter())
    {
        let (Some(lhs_input_key), Some(rhs_input_key)) = (lhs_input_key, rhs_input_key) else {
            continue;
        };

        // lhs_input_key: Column name within the left table.
        use JoinType::*;
        // Map output name of an LHS join key output to an input key column of the right table.
        // This will cause predicates referring to LHS join keys to also be pushed to the RHS table.
        if match &options.args.how {
            Left | Inner | Full => true,

            #[cfg(feature = "asof_join")]
            AsOf(_) => true,
            #[cfg(feature = "semi_anti_join")]
            Semi | Anti => true,

            // NOTE: Right-join is excluded.
            Right => false,

            #[cfg(feature = "iejoin")]
            IEJoin | Range => false,

            Cross => unreachable!(), // Cross left/right_on should be empty
        } {
            // Note: `lhs_input_key` maintains its name in the output column for all cases except
            // for a coalescing right-join.
            output_key_to_right_input_map.insert(lhs_input_key.clone(), rhs_input_key.clone());
        }

        // Map output name of an RHS join key output to a key column of the left table.
        // This will cause predicates referring to RHS join keys to also be pushed to the LHS table.
        if match &options.args.how {
            JoinType::Right => true,
            // Non-coalesced output columns of an inner join are equivalent between LHS and RHS.
            JoinType::Inner => !options.args.should_coalesce(),
            _ => false,
        } {
            let rhs_output_key: PlSmallStr = if schema_left.contains(rhs_input_key.as_str())
                && !coalesced_to_right.contains(rhs_input_key.as_str())
            {
                format_pl_smallstr!("{}{}", rhs_input_key, options.args.suffix())
            } else {
                rhs_input_key.clone()
            };

            assert!(schema.contains(&rhs_output_key));

            output_key_to_left_input_map.insert(rhs_output_key.clone(), lhs_input_key.clone());
        }
    }

    let mut pushdown_left: PlIndexMap<PlSmallStr, ExprIR> =
        init_indexmap(Some(acc_predicates.len()));
    let mut pushdown_right: PlIndexMap<PlSmallStr, ExprIR> =
        init_indexmap(Some(acc_predicates.len()));
    let mut local_predicates = Vec::with_capacity(acc_predicates.len());

    for (_, predicate) in acc_predicates {
        let mut push_left = true;
        let mut push_right = true;

        for col_name in aexpr_to_leaf_names_iter(predicate.node(), expr_arena) {
            let origin: ExprOrigin = ExprOrigin::get_column_origin(
                col_name.as_str(),
                &schema_left,
                &schema_right,
                options.args.suffix(),
                Some(&|name| coalesced_to_right.contains(name)),
            )
            .unwrap();

            push_left &= matches!(origin, ExprOrigin::Left | ExprOrigin::None)
                || output_key_to_left_input_map.contains_key(col_name);

            push_right &= matches!(origin, ExprOrigin::Right | ExprOrigin::None)
                || output_key_to_right_input_map.contains_key(col_name);
        }

        // Note: If `push_left` and `push_right` are both `true`, it means the predicate refers only
        // to the join key columns, or the predicate does not refer any columns.

        let has_residual = match &options.args.how {
            // Pushing to a single side is enough to observe the full effect of the filter.
            JoinType::Inner => !(push_left || push_right),

            // Left-join: Pushing filters to the left table is enough to observe the effect of the
            // filter. Pushing filters to the right is optional, but can only be done if the
            // filter is also pushed to the left (if this is the case it means the filter only
            // references join key columns).
            JoinType::Left => {
                push_right &= push_left;
                !push_left
            },

            // Same as left-join, just flipped around.
            JoinType::Right => {
                push_left &= push_right;
                !push_right
            },

            // Full-join: Filters must strictly apply only to coalesced output key columns.
            JoinType::Full => {
                assert!(options.args.should_coalesce());

                let push = push_left && push_right;
                push_left = push;
                push_right = push;

                !push
            },

            JoinType::Cross => {
                // Predicate should only refer to a single side.
                assert!(output_key_to_left_input_map.is_empty());
                assert!(output_key_to_right_input_map.is_empty());
                !(push_left || push_right)
            },

            // Behaves similarly to left-join on "by" columns (takes a single match instead of
            // all matches according to asof strategy).
            #[cfg(feature = "asof_join")]
            JoinType::AsOf(_) => {
                push_right &= push_left;
                !push_left
            },

            // Same as inner-join.
            #[cfg(feature = "semi_anti_join")]
            JoinType::Semi => !(push_left || push_right),

            // Anti-join is an exclusion of key tuples that exist in the right table, meaning that
            // filters can only be pushed to the right table if they are also pushed to the left.
            #[cfg(feature = "semi_anti_join")]
            JoinType::Anti => {
                push_right &= push_left;
                !push_left
            },

            // Same as inner-join.
            #[cfg(feature = "iejoin")]
            JoinType::IEJoin | JoinType::Range => !(push_left || push_right),
        };

        if has_residual {
            local_predicates.push(predicate.clone())
        }

        if push_left {
            let mut predicate = predicate.clone();
            map_column_references(&mut predicate, expr_arena, &output_key_to_left_input_map);
            insert_predicate_dedup(&mut pushdown_left, &predicate, expr_arena);
        }

        if push_right {
            let mut predicate = predicate;
            map_column_references(&mut predicate, expr_arena, &output_key_to_right_input_map);
            remove_suffix(
                &mut predicate,
                expr_arena,
                &schema_right,
                options.args.suffix(),
            );
            insert_predicate_dedup(&mut pushdown_right, &predicate, expr_arena);
        }
    }

    opt.pushdown_and_assign(input_left, pushdown_left, lp_arena, expr_arena)?;
    opt.pushdown_and_assign(input_right, pushdown_right, lp_arena, expr_arena)?;

    let lp = rewrite_hive(
        input_left,
        input_right,
        left_on,
        right_on,
        schema,
        options,
        opt,
        lp_arena,
        expr_arena,
    )?;

    let lp = opt.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena);

    let lp = if let Some((projections, schema)) = opt_post_select {
        IR::Select {
            input: lp_arena.add(lp),
            expr: projections,
            schema,
            options: ProjectionOptions {
                run_parallel: false,
                duplicate_check: false,
                should_broadcast: false,
                maintain_dataframe_height: false,
            },
        }
    } else {
        lp
    };

    let lp = apply_join_key_reduction_select(lp, opt_join_key_reduction_select, lp_arena);

    Ok(lp)
}

fn apply_join_key_reduction_select(
    lp: IR,
    opt_select: Option<(Vec<ExprIR>, SchemaRef)>,
    lp_arena: &mut Arena<IR>,
) -> IR {
    if let Some((projections, schema)) = opt_select {
        IR::Select {
            input: lp_arena.add(lp),
            expr: projections,
            schema,
            options: ProjectionOptions {
                run_parallel: false,
                duplicate_check: false,
                should_broadcast: false,
                maintain_dataframe_height: false,
            },
        }
    } else {
        lp
    }
}

#[expect(clippy::too_many_arguments)]
fn try_reduce_redundant_join_keys(
    opt: &mut PredicatePushDown,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    input_left: &mut Node,
    input_right: &mut Node,
    schema_left: &SchemaRef,
    schema_right: &SchemaRef,
    output_schema: &mut SchemaRef,
    options: &Arc<JoinOptionsIR>,
    left_on: &mut Vec<ExprIR>,
    right_on: &mut Vec<ExprIR>,
) -> PolarsResult<Option<(Vec<ExprIR>, SchemaRef)>> {
    if left_on.len() <= 1 {
        return Ok(None);
    }

    // Only filter a side when removed rows from that side cannot contribute to the join output.
    let reduce_left = match options.args.how {
        JoinType::Inner | JoinType::Right => true,
        #[cfg(feature = "semi_anti_join")]
        JoinType::Semi => true,
        _ => false,
    };
    let reduce_right = match options.args.how {
        JoinType::Inner | JoinType::Left => true,
        #[cfg(feature = "semi_anti_join")]
        JoinType::Semi => true,
        _ => false,
    };

    if !(reduce_left || reduce_right) {
        return Ok(None);
    }

    let mut remove_key = vec![false; left_on.len()];
    let mut pushdown_left = init_indexmap(None);
    let mut pushdown_right = init_indexmap(None);

    if reduce_left {
        collect_redundant_join_key_filters(
            left_on,
            right_on,
            options.args.nulls_equal,
            &mut remove_key,
            &mut pushdown_left,
            expr_arena,
        );
    }

    if reduce_right {
        collect_redundant_join_key_filters(
            right_on,
            left_on,
            options.args.nulls_equal,
            &mut remove_key,
            &mut pushdown_right,
            expr_arena,
        );
    }

    if !remove_key.iter().any(|remove| *remove) {
        return Ok(None);
    }

    debug_assert!(remove_key.iter().any(|remove| !*remove));

    if !pushdown_left.is_empty() {
        opt.pushdown_and_assign(*input_left, pushdown_left, lp_arena, expr_arena)?;
    }
    if !pushdown_right.is_empty() {
        opt.pushdown_and_assign(*input_right, pushdown_right, lp_arena, expr_arena)?;
    }

    let original_schema = output_schema.clone();

    let mut new_left_on = Vec::with_capacity(left_on.len());
    let mut new_right_on = Vec::with_capacity(right_on.len());
    for (i, (l, r)) in left_on.iter().zip(right_on.iter()).enumerate() {
        if !remove_key[i] {
            new_left_on.push(l.clone());
            new_right_on.push(r.clone());
        }
    }
    *left_on = new_left_on;
    *right_on = new_right_on;

    *output_schema = det_join_schema(
        schema_left,
        schema_right,
        left_on,
        right_on,
        options,
        expr_arena,
    )?;

    let original_names = original_schema.iter_names().collect::<Vec<_>>();
    let new_names = output_schema.iter_names().collect::<Vec<_>>();
    if original_names == new_names {
        return Ok(None);
    }

    let projections = original_schema
        .iter_names()
        .map(|name| {
            let node = expr_arena.add(AExpr::Column(name.clone()));
            ExprIR::from_node(node, expr_arena)
        })
        .collect();

    Ok(Some((projections, original_schema)))
}

fn collect_redundant_join_key_filters(
    reduced_side_on: &[ExprIR],
    other_side_on: &[ExprIR],
    nulls_equal: bool,
    remove_key: &mut [bool],
    pushdown: &mut PlIndexMap<PlSmallStr, ExprIR>,
    expr_arena: &mut Arena<AExpr>,
) {
    let op = if nulls_equal {
        Operator::EqValidity
    } else {
        Operator::Eq
    };

    let mut l_stack = Vec::new();
    let mut r_stack = Vec::new();

    for group_start in 0..other_side_on.len() {
        let group_node = other_side_on[group_start].node();

        for i in group_start + 1..other_side_on.len() {
            if remove_key[i] {
                continue;
            }

            let other_node = other_side_on[i].node();
            if !expr_arena.get(group_node).is_expr_equal_to_amortized(
                expr_arena.get(other_node),
                expr_arena,
                &mut l_stack,
                &mut r_stack,
            ) {
                continue;
            }

            remove_key[i] = true;

            let lhs = reduced_side_on[group_start].node();
            let rhs = reduced_side_on[i].node();

            if expr_arena.get(lhs).is_expr_equal_to_amortized(
                expr_arena.get(rhs),
                expr_arena,
                &mut l_stack,
                &mut r_stack,
            ) {
                continue;
            }

            let predicate = expr_arena.add(AExpr::BinaryExpr {
                left: lhs,
                op,
                right: rhs,
            });
            let predicate = ExprIR::from_node(predicate, expr_arena);
            insert_predicate_dedup(pushdown, &predicate, expr_arena);
        }
    }
}

#[cfg(feature = "iejoin")]
#[derive(Debug, Clone)]
struct IEJoinCompatiblePredicate {
    input_lhs: Node,
    input_rhs: Node,
    ie_op: InequalityOperator,
    /// Original input node.
    source_node: Node,
}
