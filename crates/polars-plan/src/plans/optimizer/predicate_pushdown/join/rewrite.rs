use super::*;

/// Attempts to rewrite the join-type based on NULL-removing filters.
///
/// Changing between some join types may cause the output column order to change. If this is the
/// case, a Vec of column selectors will be returned that restore the original column order.
#[expect(clippy::too_many_arguments)]
pub fn try_rewrite_join_type(
    schema_left: &SchemaRef,
    schema_right: &SchemaRef,
    output_schema: &mut SchemaRef,
    options: &mut Arc<JoinOptionsIR>,
    left_on: &mut Vec<ExprIR>,
    right_on: &mut Vec<ExprIR>,
    acc_predicates: &mut PlIndexMap<PlSmallStr, ExprIR>,
    expr_arena: &mut Arena<AExpr>,
    streaming: bool,
) -> PolarsResult<Option<(Vec<ExprIR>, SchemaRef)>> {
    if acc_predicates.is_empty() {
        return Ok(None);
    }

    let suffix = options.args.suffix().clone();

    // * Cross -> Inner | RangeJoin | IEJoin
    // * IEJoin -> Inner
    //
    // Note: The join rewrites here all maintain output column ordering, hence this does not need
    // to return any post-select (inserted inner joins will use JoinCoalesce::KeepColumns).
    (|| {
        match &options.args.how {
            #[cfg(feature = "iejoin")]
            JoinType::IEJoin | JoinType::Range => {},
            JoinType::Cross => {},

            _ => return PolarsResult::Ok(()),
        }

        match &options.options {
            Some(JoinTypeOptionsIR::CrossAndFilter { .. }) => {
                let Some(JoinTypeOptionsIR::CrossAndFilter { predicate }) =
                    Arc::make_mut(options).options.take()
                else {
                    unreachable!()
                };

                insert_predicate_dedup(acc_predicates, &predicate, expr_arena);
            },

            #[cfg(feature = "iejoin")]
            Some(JoinTypeOptionsIR::IEJoin(_)) => {},
            None => {},
        }

        // Try converting to inner join
        let equality_conditions = take_inner_join_compatible_filters(
            acc_predicates,
            expr_arena,
            schema_left,
            schema_right,
            &suffix,
        )?;

        for InnerJoinKeys {
            input_lhs,
            input_rhs,
        } in equality_conditions
        {
            let join_options = Arc::make_mut(options);
            join_options.args.how = JoinType::Inner;
            join_options.args.coalesce = JoinCoalesce::KeepColumns;

            left_on.push(ExprIR::from_node(input_lhs, expr_arena));
            let mut rexpr = ExprIR::from_node(input_rhs, expr_arena);
            remove_suffix(&mut rexpr, expr_arena, schema_right, &suffix);
            right_on.push(rexpr);
        }

        if options.args.how == JoinType::Inner {
            return Ok(());
        }

        // Try converting cross join to double-bounded RangeJoin
        #[cfg(feature = "iejoin")]
        if streaming
            && matches!(options.args.maintain_order, MaintainOrderJoin::None)
            && left_on.is_empty()
        {
            let range_predicate = take_double_bounded_range_join_filter(
                acc_predicates,
                expr_arena,
                schema_left,
                schema_right,
                output_schema,
                &suffix,
            )?;
            if let Some((bound_lower, bound_upper, left_is_point)) = range_predicate {
                let join_options = Arc::make_mut(options);
                join_options.args.how = JoinType::Range;
                let JoinTypeOptionsIR::IEJoin(ie_options) = join_options
                    .options
                    .get_or_insert(JoinTypeOptionsIR::IEJoin(IEJoinOptions::default()))
                else {
                    unreachable!()
                };

                left_on.push(ExprIR::from_node(bound_lower.input_lhs, expr_arena));
                let mut rexpr_lower = ExprIR::from_node(bound_lower.input_rhs, expr_arena);
                remove_suffix(&mut rexpr_lower, expr_arena, schema_right, &suffix);
                right_on.push(rexpr_lower);
                let expr_eq = |e1, e2| {
                    AExpr::is_expr_equal_to(expr_arena.get(e1), expr_arena.get(e2), expr_arena)
                };
                if left_is_point {
                    debug_assert!(expr_eq(bound_lower.input_lhs, bound_upper.input_lhs));
                    let mut rexpr_upper = ExprIR::from_node(bound_upper.input_rhs, expr_arena);
                    remove_suffix(&mut rexpr_upper, expr_arena, schema_right, &suffix);
                    right_on.push(rexpr_upper);
                } else {
                    debug_assert!(expr_eq(bound_lower.input_rhs, bound_upper.input_rhs));
                    left_on.push(ExprIR::from_node(bound_upper.input_lhs, expr_arena));
                }
                ie_options.operator1 = bound_lower.ie_op;
                ie_options.operator2 = Some(bound_upper.ie_op);
                return Ok(());
            }
        }

        // Try converting cross join to IEJoin
        #[cfg(feature = "iejoin")]
        if matches!(options.args.maintain_order, MaintainOrderJoin::None)
            && left_on.len() < IEJOIN_MAX_PREDICATES
        {
            use polars_utils::itertools::Itertools;

            let ie_conditions = take_iejoin_compatible_filters(
                acc_predicates,
                expr_arena,
                schema_left,
                schema_right,
                output_schema,
                &suffix,
            )?
            .collect_vec();

            // If there is only one predicate, prefer lowering to a single-bounded range-join
            if ie_conditions.len() == 1 && streaming {
                let join_options = Arc::make_mut(options);
                join_options.args.how = JoinType::Range;
                let JoinTypeOptionsIR::IEJoin(ie_options) = join_options
                    .options
                    .get_or_insert(JoinTypeOptionsIR::IEJoin(IEJoinOptions::default()))
                else {
                    unreachable!()
                };
                let pred = ie_conditions.into_iter().next().unwrap();
                left_on.push(ExprIR::from_node(pred.input_lhs, expr_arena));
                let mut rexpr = ExprIR::from_node(pred.input_rhs, expr_arena);
                remove_suffix(&mut rexpr, expr_arena, schema_right, &suffix);
                right_on.push(rexpr);
                ie_options.operator1 = pred.ie_op;
                return Ok(());
            }

            for IEJoinCompatiblePredicate {
                input_lhs,
                input_rhs,
                ie_op,
                source_node,
            } in ie_conditions
            {
                let join_options = Arc::make_mut(options);
                join_options.args.how = JoinType::IEJoin;

                if left_on.len() >= IEJOIN_MAX_PREDICATES {
                    // Important: Place these back into acc_predicates.
                    insert_predicate_dedup(
                        acc_predicates,
                        &ExprIR::from_node(source_node, expr_arena),
                        expr_arena,
                    );
                } else {
                    left_on.push(ExprIR::from_node(input_lhs, expr_arena));
                    let mut rexpr = ExprIR::from_node(input_rhs, expr_arena);
                    remove_suffix(&mut rexpr, expr_arena, schema_right, &suffix);
                    right_on.push(rexpr);

                    let JoinTypeOptionsIR::IEJoin(ie_options) = join_options
                        .options
                        .get_or_insert(JoinTypeOptionsIR::IEJoin(IEJoinOptions::default()))
                    else {
                        unreachable!()
                    };

                    match left_on.len() {
                        1 => ie_options.operator1 = ie_op,
                        2 => ie_options.operator2 = Some(ie_op),
                        _ => unreachable!("{}", IEJOIN_MAX_PREDICATES),
                    };
                }
            }

            if options.args.how == JoinType::IEJoin {
                return Ok(());
            }
        }

        debug_assert_eq!(options.args.how, JoinType::Cross);

        if options.args.how != JoinType::Cross {
            return Ok(());
        }

        if streaming {
            return Ok(());
        }

        let Some(nested_loop_predicates) = take_nested_loop_join_compatible_filters(
            acc_predicates,
            expr_arena,
            schema_left,
            schema_right,
            &suffix,
        )?
        .reduce(|left, right| {
            expr_arena.add(AExpr::BinaryExpr {
                left,
                op: Operator::And,
                right,
            })
        }) else {
            return Ok(());
        };

        let existing = Arc::make_mut(options)
            .options
            .replace(JoinTypeOptionsIR::CrossAndFilter {
                predicate: ExprIR::from_node(nested_loop_predicates, expr_arena),
            });
        assert!(existing.is_none()); // Important

        Ok(())
    })()?;

    if !matches!(
        &options.args.how,
        JoinType::Full | JoinType::Left | JoinType::Right
    ) {
        return Ok(None);
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

    let mut coalesced_to_right: PlIndexSet<PlSmallStr> = Default::default();
    // Removing NULLs on these columns do not allow for join downgrading.
    // We only need to track these for full-join - e.g. for left-join, removing NULLs from any left
    // column does not cause any join rewrites.
    let mut coalesced_full_join_key_outputs: PlIndexSet<PlSmallStr> = Default::default();

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
        return Ok(None);
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
    let mut original_to_new_names_map: PlIndexMap<PlSmallStr, PlSmallStr> = Default::default();
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
                let mut join_output_key_selectors = PlIndexMap::with_capacity(right_on.len());

                for (l, r) in left_on.iter().zip(right_on) {
                    // Unwrap any Cast expressions that may have been inserted for type coercion.
                    // For non full-joins coalesce can still insert casts into the key exprs.
                    let l_node = match expr_arena.get(l.node()) {
                        AExpr::Cast {
                            expr,
                            dtype: _,
                            options: _,
                        } if should_coalesce => *expr,
                        _ => l.node(),
                    };
                    let r_node = match expr_arena.get(r.node()) {
                        AExpr::Cast {
                            expr,
                            dtype: _,
                            options: _,
                        } if should_coalesce => *expr,
                        _ => r.node(),
                    };

                    let (AExpr::Column(lhs_input_key), AExpr::Column(rhs_input_key)) =
                        (expr_arena.get(l_node), expr_arena.get(r_node))
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
                let mut join_output_key_selectors = PlIndexMap::with_capacity(left_on.len());

                // The existing one is empty because the original join type was not a right-join.
                assert!(coalesced_to_right.is_empty());
                // LHS input key columns that are coalesced (i.e. not projected) for the right-join.
                let coalesced_to_right: PlIndexSet<PlSmallStr> =
                    lhs_input_column_keys_iter!().collect();
                // RHS input key columns that are coalesced (i.e. not projected) for the full-join.
                let mut coalesced_to_left: PlIndexSet<PlSmallStr> =
                    PlIndexSet::with_capacity(right_on.len());

                for (l, r) in left_on.iter().zip(right_on) {
                    // Unwrap any Cast expressions that may have been inserted for type coercion.
                    // For non full-joins coalesce can still insert casts into the key exprs.
                    let l_node = match expr_arena.get(l.node()) {
                        AExpr::Cast {
                            expr,
                            dtype: _,
                            options: _,
                        } if should_coalesce => *expr,
                        _ => l.node(),
                    };
                    let r_node = match expr_arena.get(r.node()) {
                        AExpr::Cast {
                            expr,
                            dtype: _,
                            options: _,
                        } if should_coalesce => *expr,
                        _ => r.node(),
                    };

                    let (AExpr::Column(lhs_input_key), AExpr::Column(rhs_input_key)) =
                        (expr_arena.get(l_node), expr_arena.get(r_node))
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

    Ok(project_to_original.map(|p| (p, original_output_schema)))
}

struct InnerJoinKeys {
    input_lhs: Node,
    input_rhs: Node,
}

/// Removes all equality predicates that can be used as inner-join conditions from `acc_predicates`.
fn take_inner_join_compatible_filters(
    acc_predicates: &mut PlIndexMap<PlSmallStr, ExprIR>,
    expr_arena: &mut Arena<AExpr>,
    schema_left: &Schema,
    schema_right: &Schema,
    suffix: &str,
) -> PolarsResult<indexmap::map::IntoValues<Node, InnerJoinKeys>> {
    take_predicates_mut(acc_predicates, expr_arena, |ae, _ae_node, expr_arena| {
        Ok(match ae {
            AExpr::BinaryExpr {
                left,
                op: Operator::Eq,
                right,
            } => {
                let left_origin = ExprOrigin::get_expr_origin(
                    *left,
                    expr_arena,
                    schema_left,
                    schema_right,
                    suffix,
                    None, // is_coalesced_to_right
                )?;
                let right_origin = ExprOrigin::get_expr_origin(
                    *right,
                    expr_arena,
                    schema_left,
                    schema_right,
                    suffix,
                    None,
                )?;

                match (left_origin, right_origin) {
                    (ExprOrigin::Left, ExprOrigin::Right) => Some(InnerJoinKeys {
                        input_lhs: *left,
                        input_rhs: *right,
                    }),
                    (ExprOrigin::Right, ExprOrigin::Left) => Some(InnerJoinKeys {
                        input_lhs: *right,
                        input_rhs: *left,
                    }),
                    _ => None,
                }
            },
            _ => None,
        })
    })
}
