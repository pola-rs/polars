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
    mut schema: SchemaRef,
    mut options: Arc<JoinOptionsIR>,
    mut acc_predicates: PlHashMap<PlSmallStr, ExprIR>,
) -> PolarsResult<IR> {
    let schema_left = lp_arena.get(input_left).schema(lp_arena).into_owned();
    let schema_right = lp_arena.get(input_right).schema(lp_arena).into_owned();

    let opt_post_select = try_rewrite_join_type(
        &schema_left,
        &schema_right,
        &mut schema,
        &mut options,
        &left_on,
        &right_on,
        &mut acc_predicates,
        expr_arena,
    );

    if match &options.args.how {
        // Full-join with no coalesce. We can only push filters if they do not remove NULLs, but
        // we don't have a reliable way to guarantee this.
        JoinType::Full => !options.args.should_coalesce(),

        #[cfg(feature = "iejoin")]
        JoinType::IEJoin => {
            // TODO: Optimize this - https://github.com/pola-rs/polars/issues/23489
            true
        },

        _ => false,
    } || acc_predicates.is_empty()
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

    let lp = IR::Join {
        input_left,
        input_right,
        left_on,
        right_on,
        schema,
        options,
    };

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
            },
        }
    } else {
        lp
    };

    Ok(lp)
}

/// Attempts to rewrite the join-type based on NULL-removing filters.
///
/// Changing between some join types may cause the output column order to change. If this is the
/// case, a Vec of column selectors will be returned that restore the original column order.
#[expect(clippy::too_many_arguments)]
fn try_rewrite_join_type(
    schema_left: &SchemaRef,
    schema_right: &SchemaRef,
    output_schema: &mut SchemaRef,
    options: &mut Arc<JoinOptionsIR>,
    left_on: &[ExprIR],
    right_on: &[ExprIR],
    acc_predicates: &mut PlHashMap<PlSmallStr, ExprIR>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<(Vec<ExprIR>, SchemaRef)> {
    if acc_predicates.is_empty()
        || !matches!(
            &options.args.how,
            JoinType::Full | JoinType::Left | JoinType::Right
        )
    {
        return None;
    }

    let should_coalesce = options.args.should_coalesce();

    /// Note: This may panic if `args.should_coalesce()` is false.
    macro_rules! lhs_input_column_keys_iter {
        () => {{
            left_on.iter().map(|expr| {
                let node = match expr_arena.get(expr.node()) {
                    AExpr::Cast {
                        expr,
                        dtype: _,
                        options: _,
                    } if should_coalesce => *expr,

                    _ => expr.node(),
                };

                let AExpr::Column(name) = expr_arena.get(node) else {
                    // All keys should be columns when coalesce=True
                    unreachable!()
                };

                name.clone()
            })
        }};
    }

    let mut coalesced_to_right: PlHashSet<PlSmallStr> = Default::default();
    // Removing NULLs on these columns do not allow for join downgrading.
    // We only need to track these for full-join - e.g. for left-join, removing NULLs from any left
    // column does not cause any join rewrites.
    let mut coalesced_full_join_key_outputs: PlHashSet<PlSmallStr> = Default::default();

    if options.args.should_coalesce() {
        match &options.args.how {
            JoinType::Full => {
                coalesced_full_join_key_outputs = lhs_input_column_keys_iter!().collect()
            },
            JoinType::Right => coalesced_to_right = lhs_input_column_keys_iter!().collect(),
            _ => {},
        }
    }

    let mut non_null_side = ExprOrigin::None;

    for predicate in acc_predicates.values() {
        for node in MintermIter::new(predicate.node(), expr_arena) {
            predicate_non_null_column_outputs(node, expr_arena, &mut |non_null_column| {
                if coalesced_full_join_key_outputs.contains(non_null_column) {
                    return;
                }

                non_null_side |= ExprOrigin::get_column_origin(
                    non_null_column.as_str(),
                    schema_left,
                    schema_right,
                    options.args.suffix(),
                    Some(&|x| coalesced_to_right.contains(x)),
                )
                .unwrap();
            });
        }
    }

    #[expect(clippy::question_mark)]
    let Some(new_join_type) = (match non_null_side {
        ExprOrigin::Both => Some(JoinType::Inner),

        ExprOrigin::Left => match &options.args.how {
            JoinType::Full => Some(JoinType::Left),
            JoinType::Right => Some(JoinType::Inner),
            _ => None,
        },

        ExprOrigin::Right => match &options.args.how {
            JoinType::Full => Some(JoinType::Right),
            JoinType::Left => Some(JoinType::Inner),
            _ => None,
        },

        ExprOrigin::None => None,
    }) else {
        return None;
    };

    let options = Arc::make_mut(options);
    // Ensure JoinSpecific is materialized to a specific config option, as we change the join type.
    options.args.coalesce = if options.args.should_coalesce() {
        JoinCoalesce::CoalesceColumns
    } else {
        JoinCoalesce::KeepColumns
    };
    let original_join_type = std::mem::replace(&mut options.args.how, new_join_type.clone());
    let original_output_schema = match (&original_join_type, &new_join_type) {
        (JoinType::Right, _) | (_, JoinType::Right) => std::mem::replace(
            output_schema,
            det_join_schema(
                schema_left,
                schema_right,
                left_on,
                right_on,
                options,
                expr_arena,
            )
            .unwrap(),
        ),
        _ => {
            debug_assert_eq!(
                output_schema,
                &det_join_schema(
                    schema_left,
                    schema_right,
                    left_on,
                    right_on,
                    options,
                    expr_arena,
                )
                .unwrap()
            );
            output_schema.clone()
        },
    };

    // Maps the original join output names to the new join output names (used for mapping column
    // references of the predicates).
    let mut original_to_new_names_map: PlHashMap<PlSmallStr, PlSmallStr> = Default::default();
    // Projects the new join output table back into the original join output table.
    let mut project_to_original: Option<Vec<ExprIR>> = None;

    if options.args.should_coalesce() {
        // If we changed join types between a coalescing right-join, we need to do a select() to restore the column
        // order of the original join type. The column references in the predicates may also need to be changed.
        match (&original_join_type, &new_join_type) {
            (JoinType::Right, JoinType::Right) => unreachable!(),

            // Right-join rewritten to inner-join.
            //
            // E.g.
            // Left:  | a | b | c |
            // Right: | a | b | c |
            //
            // right_join(left_on='a', right_on='b'): |  b | c | a | *b_right | c_right |
            // inner_join(left_on='a', right_on='b'): | *a | b | c |  a_right | c_right |
            // note: '*' means coalesced key output column
            //
            // project_to_original: | col(b) | col(c) | col(a_right).alias(a) | col(a).alias(b_right) | col(c_right) |
            // original_to_new_names_map: {'a': 'a_right', 'b_right': 'a'}
            //
            (JoinType::Right, JoinType::Inner) => {
                let mut join_output_key_selectors = PlHashMap::with_capacity(right_on.len());

                for (l, r) in left_on.iter().zip(right_on) {
                    let (AExpr::Column(lhs_input_key), AExpr::Column(rhs_input_key)) =
                        (expr_arena.get(l.node()), expr_arena.get(r.node()))
                    else {
                        // `should_coalesce() == true` should guarantee all are columns.
                        unreachable!()
                    };

                    let original_key_output_name: PlSmallStr = if schema_left
                        .contains(rhs_input_key.as_str())
                        && !coalesced_to_right.contains(rhs_input_key.as_str())
                    {
                        format_pl_smallstr!("{}{}", rhs_input_key, options.args.suffix())
                    } else {
                        rhs_input_key.clone()
                    };

                    let new_key_output_name = lhs_input_key.clone();
                    let rhs_input_key = rhs_input_key.clone();

                    let node = expr_arena.add(AExpr::Column(lhs_input_key.clone()));
                    let mut ae = ExprIR::from_node(node, expr_arena);

                    if original_key_output_name != new_key_output_name {
                        // E.g. left_on=col(a), right_on=col(b)
                        // rhs_output_key = 'b', lhs_input_key = 'a', the original right-join is supposed to output 'b'.
                        original_to_new_names_map.insert(
                            original_key_output_name.clone(),
                            new_key_output_name.clone(),
                        );
                        ae.set_alias(original_key_output_name)
                    }

                    join_output_key_selectors.insert(rhs_input_key, ae);
                }

                let mut column_selectors: Vec<ExprIR> = Vec::with_capacity(output_schema.len());

                for lhs_input_col in schema_left.iter_names() {
                    if coalesced_to_right.contains(lhs_input_col) {
                        continue;
                    }

                    let node = expr_arena.add(AExpr::Column(lhs_input_col.clone()));
                    column_selectors.push(ExprIR::from_node(node, expr_arena));
                }

                for rhs_input_col in schema_right.iter_names() {
                    let expr = if let Some(expr) = join_output_key_selectors.get(rhs_input_col) {
                        expr.clone()
                    } else if schema_left.contains(rhs_input_col) {
                        let new_join_output_name =
                            format_pl_smallstr!("{}{}", rhs_input_col, options.args.suffix());

                        let node = expr_arena.add(AExpr::Column(new_join_output_name.clone()));
                        let mut expr = ExprIR::from_node(node, expr_arena);

                        // The column with the same name from the LHS is not projected in the original
                        // right-join, so we alias to remove the suffix that was added from the inner-join.
                        if coalesced_to_right.contains(rhs_input_col.as_str()) {
                            original_to_new_names_map
                                .insert(rhs_input_col.clone(), new_join_output_name);
                            expr.set_alias(rhs_input_col.clone());
                        }

                        expr
                    } else {
                        let node = expr_arena.add(AExpr::Column(rhs_input_col.clone()));
                        ExprIR::from_node(node, expr_arena)
                    };

                    column_selectors.push(expr)
                }

                assert_eq!(column_selectors.len(), output_schema.len());
                assert_eq!(column_selectors.len(), original_output_schema.len());

                if cfg!(debug_assertions) {
                    assert!(
                        column_selectors
                            .iter()
                            .zip(original_output_schema.iter_names())
                            .all(|(l, r)| l.output_name() == r)
                    )
                }

                project_to_original = Some(column_selectors)
            },

            // Full-join rewritten to right-join
            //
            // E.g.
            // Left:  | a | b | c |
            // Right: | a | b | c |
            //
            // full_join(left_on='a', right_on='b'):  | *a | b | c |  a_right | c_right |
            // right_join(left_on='a', right_on='b'): |  b | c | a | *b_right | c_right |
            // note: '*' means coalesced key output column
            //
            // project_to_original: | col(b_right).alias(a) | col(b) | col(c) | col(a).alias(a_right) | col(c_right) |
            // original_to_new_names_map: {'a': 'b_right', 'a_right': 'a'}
            //
            (JoinType::Full, JoinType::Right) => {
                let mut join_output_key_selectors = PlHashMap::with_capacity(left_on.len());

                // The existing one is empty because the original join type was not a right-join.
                assert!(coalesced_to_right.is_empty());
                // LHS input key columns that are coalesced (i.e. not projected) for the right-join.
                let coalesced_to_right: PlHashSet<PlSmallStr> =
                    lhs_input_column_keys_iter!().collect();
                // RHS input key columns that are coalesced (i.e. not projected) for the full-join.
                let mut coalesced_to_left: PlHashSet<PlSmallStr> =
                    PlHashSet::with_capacity(right_on.len());

                for (l, r) in left_on.iter().zip(right_on) {
                    let (AExpr::Column(lhs_input_key), AExpr::Column(rhs_input_key)) =
                        (expr_arena.get(l.node()), expr_arena.get(r.node()))
                    else {
                        // `should_coalesce() == true` should guarantee all columns.
                        unreachable!()
                    };

                    let new_key_output_name: PlSmallStr = if schema_left
                        .contains(rhs_input_key.as_str())
                        && !coalesced_to_right.contains(rhs_input_key.as_str())
                    {
                        format_pl_smallstr!("{}{}", rhs_input_key, options.args.suffix())
                    } else {
                        rhs_input_key.clone()
                    };

                    let lhs_input_key = lhs_input_key.clone();
                    let rhs_input_key = rhs_input_key.clone();
                    let original_key_output_name = &lhs_input_key;

                    coalesced_to_left.insert(rhs_input_key);

                    let node = expr_arena.add(AExpr::Column(new_key_output_name.clone()));

                    let mut ae = ExprIR::from_node(node, expr_arena);

                    // E.g. left_on=col(a), right_on=col(b)
                    // rhs_output_key = 'b', lhs_input_key = 'a'
                    if new_key_output_name != original_key_output_name {
                        original_to_new_names_map.insert(
                            original_key_output_name.clone(),
                            new_key_output_name.clone(),
                        );
                        ae.set_alias(original_key_output_name.clone())
                    }

                    join_output_key_selectors.insert(lhs_input_key.clone(), ae);
                }

                let mut column_selectors = Vec::with_capacity(output_schema.len());

                for lhs_input_col in schema_left.iter_names() {
                    let expr = if let Some(expr) = join_output_key_selectors.get(lhs_input_col) {
                        expr.clone()
                    } else {
                        let node = expr_arena.add(AExpr::Column(lhs_input_col.clone()));
                        ExprIR::from_node(node, expr_arena)
                    };

                    column_selectors.push(expr)
                }

                for rhs_input_col in schema_right.iter_names() {
                    if coalesced_to_left.contains(rhs_input_col) {
                        continue;
                    }

                    let mut original_output_name: Option<PlSmallStr> = None;

                    let new_join_output_name = if schema_left.contains(rhs_input_col) {
                        let suffixed =
                            format_pl_smallstr!("{}{}", rhs_input_col, options.args.suffix());

                        if coalesced_to_right.contains(rhs_input_col) {
                            original_output_name = Some(suffixed);
                            rhs_input_col.clone()
                        } else {
                            suffixed
                        }
                    } else {
                        rhs_input_col.clone()
                    };

                    let node = expr_arena.add(AExpr::Column(new_join_output_name));

                    let mut expr = ExprIR::from_node(node, expr_arena);

                    if let Some(original_output_name) = original_output_name {
                        original_to_new_names_map
                            .insert(original_output_name.clone(), rhs_input_col.clone());
                        expr.set_alias(original_output_name);
                    }

                    column_selectors.push(expr);
                }

                assert_eq!(column_selectors.len(), output_schema.len());
                assert_eq!(column_selectors.len(), original_output_schema.len());

                if cfg!(debug_assertions) {
                    assert!(
                        column_selectors
                            .iter()
                            .zip(original_output_schema.iter_names())
                            .all(|(l, r)| l.output_name() == r)
                    )
                }

                project_to_original = Some(column_selectors)
            },

            (JoinType::Right, _) | (_, JoinType::Right) => unreachable!(),

            _ => {},
        }
    }

    if !original_to_new_names_map.is_empty() {
        assert!(project_to_original.is_some());

        for (_, predicate_expr) in acc_predicates.iter_mut() {
            map_column_references(predicate_expr, expr_arena, &original_to_new_names_map);
        }
    }

    project_to_original.map(|p| (p, original_output_schema))
}
