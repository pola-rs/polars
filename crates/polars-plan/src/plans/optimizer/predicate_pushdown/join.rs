use polars_utils::format_pl_smallstr;

use super::*;
use crate::plans::optimizer::join_utils::remove_suffix;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_join(
    opt: &mut PredicatePushDown,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    input_left: Node,
    input_right: Node,
    left_on: Vec<ExprIR>,
    right_on: Vec<ExprIR>,
    schema: SchemaRef,
    options: Arc<JoinOptionsIR>,
    acc_predicates: PlHashMap<PlSmallStr, ExprIR>,
) -> PolarsResult<IR> {
    let schema_left = lp_arena.get(input_left).schema(lp_arena);
    let schema_right = lp_arena.get(input_right).schema(lp_arena);

    // This is always lowered to cross join + filter.
    // The translation to IEJoin happens in collapse_joins, which runs after this function.
    #[cfg(feature = "iejoin")]
    assert!(!matches!(&options.args.how, JoinType::IEJoin));

    if acc_predicates.is_empty()
        || match &options.args.how {
            // Full-join with no coalesce. We can only push filters if they do not remove NULLs, but
            // we don't have a reliable way to guarantee this.
            JoinType::Full => !options.args.should_coalesce(),

            _ => false,
        }
    {
        let lp = IR::Join {
            input_left,
            input_right,
            left_on,
            right_on,
            schema,
            options,
        };

        return opt.no_pushdown_restart_opt(lp, acc_predicates, lp_arena, expr_arena);
    }

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

                if let AExpr::Column(name) = expr_arena.get(expr.node()) {
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

                if let AExpr::Column(name) = expr_arena.get(expr.node()) {
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
    let coalesced_to_right: PlHashSet<PlSmallStr> =
        if matches!(&options.args.how, JoinType::Right) && options.args.should_coalesce() {
            get_lhs_column_keys_iter()
                .map(|x| x.unwrap().clone())
                .collect()
        } else {
            Default::default()
        };

    let mut output_key_to_left_input_map: PlHashMap<PlSmallStr, PlSmallStr> =
        PlHashMap::with_capacity(get_lhs_column_keys_iter().len());
    let mut output_key_to_right_input_map: PlHashMap<PlSmallStr, PlSmallStr> =
        PlHashMap::with_capacity(get_rhs_column_keys_iter().len());

    for (lhs_input_key, rhs_input_key) in get_lhs_column_keys_iter().zip(get_rhs_column_keys_iter())
    {
        let (Some(lhs_input_key), Some(rhs_input_key)) = (lhs_input_key, rhs_input_key) else {
            continue;
        };

        // lhs_input_key: Column name within the left table.
        use JoinType::*;
        // Map output name of an LHS join key output to an input key column of the right table.
        if match &options.args.how {
            Left | Inner | Full => true,

            #[cfg(feature = "asof_join")]
            AsOf(_) => true,
            #[cfg(feature = "semi_anti_join")]
            Semi | Anti => true,

            // NOTE: Right-join is excluded.
            Right => false,

            #[cfg(feature = "iejoin")]
            IEJoin => unreachable!(),
            Cross => unreachable!(),
        } {
            // Note: `lhs_input_key` maintains its name in the output column for all cases except
            // for a coalescing right-join.
            output_key_to_right_input_map.insert(lhs_input_key.clone(), rhs_input_key.clone());
        }

        // Map output name of an RHS join key output to a key column of the left table.
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

    let mut pushdown_left: PlHashMap<PlSmallStr, ExprIR> = init_hashmap(Some(acc_predicates.len()));
    let mut pushdown_right: PlHashMap<PlSmallStr, ExprIR> =
        init_hashmap(Some(acc_predicates.len()));
    let mut local_predicates = Vec::with_capacity(acc_predicates.len());

    for (predicate_key, predicate) in acc_predicates {
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

            push_left &= matches!(origin, ExprOrigin::Left)
                || output_key_to_left_input_map.contains_key(&col_name);

            push_right &= matches!(origin, ExprOrigin::Right)
                || output_key_to_right_input_map.contains_key(&col_name);
        }

        // Note: If `push_left` and `push_right` are both `true`, it means the predicate refers only
        // to the join key columns.

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

            // Same as inner-join
            #[cfg(feature = "semi_anti_join")]
            JoinType::Semi => !(push_left || push_right),

            // Anti-join is an exclusion of key tuples that exist in the right table, meaning that
            // filters can only be pushed to the right table if they are also pushed to the left.
            #[cfg(feature = "semi_anti_join")]
            JoinType::Anti => {
                push_right &= push_left;
                !push_left
            },

            #[cfg(feature = "iejoin")]
            JoinType::IEJoin => unreachable!(),
        };

        if has_residual {
            local_predicates.push(predicate.clone())
        }

        if push_left {
            let mut predicate = predicate.clone();
            map_column_references(&mut predicate, expr_arena, &output_key_to_left_input_map);
            pushdown_left.insert(predicate_key.clone(), predicate);
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
            pushdown_right.insert(predicate_key, predicate);
        }
    }

    opt.pushdown_and_assign(input_left, pushdown_left, lp_arena, expr_arena)?;
    opt.pushdown_and_assign(input_right, pushdown_right, lp_arena, expr_arena)?;

    let lp = IR::Join {
        input_left,
        input_right,
        left_on,
        right_on,
        schema,
        options,
    };

    Ok(opt.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
}
