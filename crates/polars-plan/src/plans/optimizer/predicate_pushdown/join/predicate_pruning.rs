use super::*;
use crate::plans::aexpr::{ExprPushdownGroup, is_inherently_nondeterministic};

fn commutes_with_filter(predicate: &ExprIR, expr_arena: &Arena<AExpr>) -> bool {
    matches!(
        ExprPushdownGroup::Pushable.update_with_expr_rec(
            expr_arena.get(predicate.node()),
            expr_arena,
            None,
        ),
        ExprPushdownGroup::Pushable
    ) && !is_inherently_nondeterministic(predicate.node(), expr_arena)
}

/// ON conditions may filter the non-preserved input of an outer join. The entire
/// condition must commute with filtering, including the expressions used as keys.
#[allow(clippy::too_many_arguments)]
pub(super) fn push_down_join_condition(
    input_left: &mut Node,
    input_right: &mut Node,
    schema_left: &Schema,
    schema_right: &Schema,
    options: &mut Arc<JoinOptionsIR>,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<()> {
    let JoinTypeOptionsIR::CrossAndFilter { predicate } = &options.options else {
        return Ok(());
    };
    if !matches!(options.args.how, JoinType::Left | JoinType::Right)
        || !commutes_with_filter(predicate, expr_arena)
    {
        return Ok(());
    }

    let suffix = options.args.suffix();
    let mut local = Vec::new();
    let mut pushed = Vec::new();
    for node in MintermIter::new(predicate.node(), expr_arena) {
        let origin =
            ExprOrigin::get_expr_origin(node, expr_arena, schema_left, schema_right, suffix, None)?;
        let pushable = matches!(
            (&options.args.how, origin),
            (JoinType::Left, ExprOrigin::Right) | (JoinType::Right, ExprOrigin::Left)
        );
        let predicate = ExprIR::from_node(node, expr_arena);
        if pushable {
            pushed.push(predicate);
        } else {
            local.push(predicate);
        }
    }
    if pushed.is_empty() {
        return Ok(());
    }

    let input = if options.args.how == JoinType::Left {
        for predicate in &mut pushed {
            remove_suffix(predicate, expr_arena, schema_right, suffix);
        }
        input_right
    } else {
        input_left
    };
    *input = lp_arena.add(IR::Filter {
        input: *input,
        predicate: combine_predicates(pushed, expr_arena).unwrap(),
    });
    Arc::make_mut(options).options = if let Some(predicate) = combine_predicates(local, expr_arena)
    {
        JoinTypeOptionsIR::CrossAndFilter { predicate }
    } else {
        let node = expr_arena.add(AExpr::Literal(Scalar::from(true).into()));
        let key = ExprIR::from_node(node, expr_arena);
        JoinTypeOptionsIR::Equi {
            on: vec![(key.clone(), key)],
            fused_predicate: None,
        }
    };
    Ok(())
}

#[cfg(feature = "iejoin")]
/// Removes all inequality filters that can be used as iejoin conditions from `acc_predicates`.
pub fn take_iejoin_compatible_filters(
    acc_predicates: &mut PlIndexMap<PlSmallStr, ExprIR>,
    expr_arena: &mut Arena<AExpr>,
    schema_left: &Schema,
    schema_right: &Schema,
    output_schema: &Schema,
    suffix: &str,
) -> PolarsResult<indexmap::map::IntoValues<Node, IEJoinCompatiblePredicate>> {
    return take_predicates_mut(acc_predicates, expr_arena, |ae, ae_node, expr_arena| {
        Ok(match ae {
            AExpr::BinaryExpr { left, op, right } => {
                if to_inequality_operator(op).is_none() {
                    return Ok(None);
                }

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

                let is_supported_type =
                    |node: Node| -> PolarsResult<bool> {
                        let field = expr_arena
                            .get(node)
                            .to_field(&ToFieldContext::new(expr_arena, output_schema))?;
                        let dtype = field.dtype();
                        let phys = dtype.to_physical();
                        Ok(!dtype.is_nested()
                            && phys.is_primitive_numeric()
                            && !dtype.is_categorical())
                    };

                // IEJoin only supports physical representations whose ordering matches the
                // logical dtype. Categorical codes are in first-appearance order, whereas
                // Categorical comparisons are lexical.
                if !is_supported_type(*left)? || !is_supported_type(*right)? {
                    return Ok(None);
                }

                match (left_origin, right_origin) {
                    (ExprOrigin::Left, ExprOrigin::Right) => Some(IEJoinCompatiblePredicate {
                        input_lhs: *left,
                        input_rhs: *right,
                        ie_op: to_inequality_operator(op).unwrap(),
                        source_node: ae_node,
                    }),
                    (ExprOrigin::Right, ExprOrigin::Left) => {
                        let op = op.swap_operands().unwrap();

                        Some(IEJoinCompatiblePredicate {
                            input_lhs: *right,
                            input_rhs: *left,
                            ie_op: to_inequality_operator(&op).unwrap(),
                            source_node: ae_node,
                        })
                    },
                    _ => None,
                }
            },
            _ => None,
        })
    });

    fn to_inequality_operator(op: &Operator) -> Option<InequalityOperator> {
        match op {
            Operator::Lt => Some(InequalityOperator::Lt),
            Operator::LtEq => Some(InequalityOperator::LtEq),
            Operator::Gt => Some(InequalityOperator::Gt),
            Operator::GtEq => Some(InequalityOperator::GtEq),
            _ => None,
        }
    }
}

#[cfg(feature = "iejoin")]
pub fn take_double_bounded_range_join_filter(
    acc_predicates: &mut PlIndexMap<PlSmallStr, ExprIR>,
    expr_arena: &mut Arena<AExpr>,
    schema_left: &Schema,
    schema_right: &Schema,
    output_schema: &Schema,
    suffix: &str,
    dedup: &mut PredicateDedupState,
) -> PolarsResult<Option<(IEJoinCompatiblePredicate, IEJoinCompatiblePredicate, bool)>> {
    use InequalityOperator::*;
    use polars_utils::itertools::Itertools;

    let ie_join_filters = take_iejoin_compatible_filters(
        acc_predicates,
        expr_arena,
        schema_left,
        schema_right,
        output_schema,
        suffix,
    )?
    .collect_vec();

    let (lower_idx, upper_idx, left_is_bounded_side) = 'bound_preds: {
        let mut l_stack = Vec::new();
        let mut r_stack = Vec::new();
        let mut exprs_eq = |e1, e2| {
            AExpr::is_expr_equal_to_amortized(e1, e2, expr_arena, &mut l_stack, &mut r_stack)
        };
        for (idx1, pred1) in ie_join_filters.iter().enumerate() {
            for (idx2, pred2) in ie_join_filters
                .iter()
                .enumerate()
                .take_while(|(idx2, _)| *idx2 < idx1)
            {
                let lhs_expr1 = expr_arena.get(pred1.input_lhs);
                let lhs_expr2 = expr_arena.get(pred2.input_lhs);
                let rhs_expr1 = expr_arena.get(pred1.input_rhs);
                let rhs_expr2 = expr_arena.get(pred2.input_rhs);
                let op1_is_less = matches!(pred1.ie_op, LtEq | Lt);
                let op2_is_less = matches!(pred2.ie_op, LtEq | Lt);
                let lhs_exprs_eq = exprs_eq(lhs_expr1, lhs_expr2);
                let rhs_exprs_eq = exprs_eq(rhs_expr1, rhs_expr2);
                if lhs_exprs_eq && !op1_is_less && op2_is_less {
                    break 'bound_preds (idx1, idx2, true);
                } else if lhs_exprs_eq && op1_is_less && !op2_is_less {
                    break 'bound_preds (idx2, idx1, true);
                } else if rhs_exprs_eq && op1_is_less && !op2_is_less {
                    break 'bound_preds (idx1, idx2, false);
                } else if rhs_exprs_eq && !op1_is_less && op2_is_less {
                    break 'bound_preds (idx2, idx1, false);
                }
            }
        }
        // No compatible filters found
        for pred in ie_join_filters.into_iter() {
            insert_predicate_dedup(
                acc_predicates,
                &ExprIR::from_node(pred.source_node, expr_arena),
                expr_arena,
                dedup,
            );
        }
        return Ok(None);
    };

    let mut bound_lower = None;
    let mut bound_upper = None;
    for (idx, pred) in ie_join_filters.into_iter().enumerate() {
        if idx == lower_idx {
            bound_lower = Some(pred);
        } else if idx == upper_idx {
            bound_upper = Some(pred);
        } else {
            insert_predicate_dedup(
                acc_predicates,
                &ExprIR::from_node(pred.source_node, expr_arena),
                expr_arena,
                dedup,
            );
        }
    }
    Ok(Some((
        bound_lower.unwrap(),
        bound_upper.unwrap(),
        left_is_bounded_side,
    )))
}

/// Removes all filters that can be used as nested loop join conditions from `acc_predicates`.
///
/// Note that filters that refer only to a single side are not removed so that they can be pushed
/// into the LHS/RHS tables.
pub fn take_nested_loop_join_compatible_filters(
    acc_predicates: &mut PlIndexMap<PlSmallStr, ExprIR>,
    expr_arena: &mut Arena<AExpr>,
    schema_left: &Schema,
    schema_right: &Schema,
    suffix: &str,
) -> PolarsResult<indexmap::map::IntoValues<Node, Node>> {
    take_predicates_mut(acc_predicates, expr_arena, |_ae, ae_node, expr_arena| {
        Ok(
            match ExprOrigin::get_expr_origin(
                ae_node,
                expr_arena,
                schema_left,
                schema_right,
                suffix,
                None,
            )? {
                // Leave single-origin exprs as they get pushed to the left/right tables individually.
                ExprOrigin::Left | ExprOrigin::Right | ExprOrigin::None => None,
                _ => Some(ae_node),
            },
        )
    })
}

/// Removes predicates from the map according to a function.
pub fn take_predicates_mut<F, T>(
    acc_predicates: &mut PlIndexMap<PlSmallStr, ExprIR>,
    expr_arena: &mut Arena<AExpr>,
    take_predicate: F,
) -> PolarsResult<indexmap::map::IntoValues<Node, T>>
where
    F: Fn(&AExpr, Node, &Arena<AExpr>) -> PolarsResult<Option<T>>,
{
    let mut selected_predicates: PlIndexMap<Node, T> = init_indexmap(None);

    for predicate in acc_predicates.values() {
        for node in MintermIter::new(predicate.node(), expr_arena) {
            let ae = expr_arena.get(node);

            if let Some(t) = take_predicate(ae, node, expr_arena)? {
                selected_predicates.insert(node, t);
            }
        }
    }

    if !selected_predicates.is_empty() {
        remove_min_terms(acc_predicates, expr_arena, &|node| {
            selected_predicates.contains_key(node)
        });
    }

    return Ok(selected_predicates.into_values());

    #[inline(never)]
    fn remove_min_terms(
        acc_predicates: &mut PlIndexMap<PlSmallStr, ExprIR>,
        expr_arena: &mut Arena<AExpr>,
        should_remove: &dyn Fn(&Node) -> bool,
    ) {
        let mut remove_keys = PlIndexSet::new();
        let mut nodes_scratch = vec![];

        for (k, predicate) in acc_predicates.iter_mut() {
            let mut has_removed = false;

            nodes_scratch.clear();
            nodes_scratch.extend(
                MintermIter::new(predicate.node(), expr_arena).filter(|node| {
                    let remove = should_remove(node);
                    has_removed |= remove;
                    !remove
                }),
            );

            if nodes_scratch.is_empty() {
                remove_keys.insert(k.clone());
                continue;
            };

            if has_removed {
                let new_predicate_node = nodes_scratch
                    .drain(..)
                    .reduce(|left, right| {
                        expr_arena.add(AExpr::BinaryExpr {
                            left,
                            op: Operator::And,
                            right,
                        })
                    })
                    .unwrap();

                *predicate = ExprIR::from_node(new_predicate_node, expr_arena);
            }
        }

        for k in remove_keys {
            let v = acc_predicates.swap_remove(&k);
            assert!(v.is_some());
        }
    }
}

/// The side of the join whose rows are dropped when nulls in `name`, an output column, are
/// dropped. A coalesced key of a full join is null for unmatched rows of either side, so it
/// gives `None`.
pub(super) fn non_null_side_for_column(
    name: &str,
    schema_left: &Schema,
    schema_right: &Schema,
    options: &JoinOptionsIR,
) -> ExprOrigin {
    let coalesced_key = |name: &str| {
        options.args.should_coalesce() && options.options.left_on().any(|e| e.output_name() == name)
    };
    match options.args.how {
        JoinType::Full if coalesced_key(name) => ExprOrigin::None,
        JoinType::Right => ExprOrigin::get_column_origin(
            name,
            schema_left,
            schema_right,
            options.args.suffix(),
            Some(&coalesced_key),
        )
        .unwrap(),
        _ => ExprOrigin::get_column_origin(
            name,
            schema_left,
            schema_right,
            options.args.suffix(),
            None,
        )
        .unwrap(),
    }
}

/// The stricter join an outer join becomes when a filter above it drops rows that are null
/// on `non_null_side`.
pub(super) fn downgraded_join_type(how: &JoinType, non_null_side: ExprOrigin) -> Option<JoinType> {
    match non_null_side {
        ExprOrigin::Both => Some(JoinType::Inner),

        ExprOrigin::Left => match how {
            JoinType::Full => Some(JoinType::Left),
            JoinType::Right => Some(JoinType::Inner),
            _ => None,
        },

        ExprOrigin::Right => match how {
            JoinType::Full => Some(JoinType::Right),
            JoinType::Left => Some(JoinType::Inner),
            _ => None,
        },

        ExprOrigin::None => None,
    }
}

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
    dedup: &mut PredicateDedupState,
) -> PolarsResult<Option<(Vec<ExprIR>, SchemaRef)>> {
    // A non-equi condition is attached directly to the join, so this must still run even
    // with nothing pushed down from above.
    let has_attached_predicate =
        matches!(&options.options, JoinTypeOptionsIR::CrossAndFilter { .. }) || {
            #[cfg(feature = "iejoin")]
            {
                matches!(
                    &options.options,
                    JoinTypeOptionsIR::IEJoin { .. } | JoinTypeOptionsIR::Range { .. }
                )
            }
            #[cfg(not(feature = "iejoin"))]
            {
                false
            }
        };
    if acc_predicates.is_empty() && !has_attached_predicate {
        return Ok(None);
    }

    let suffix = options.args.suffix().clone();

    // * Cross -> Inner | RangeJoin | IEJoin
    // * IEJoin -> Inner
    //
    // Note: The join rewrites here all maintain output column ordering, hence this does not need
    // to return any post-select (inserted inner joins will use JoinCoalesce::KeepColumns).
    (|| {
        // Outer ON conditions must be fully captured by the selected join algorithm.
        let is_outer_non_equi =
            matches!(options.args.how, JoinType::Left | JoinType::Right) && options.is_non_equi();

        if is_outer_non_equi {
            if !try_rewrite_outer_equi_join(
                schema_left,
                schema_right,
                options,
                left_on,
                right_on,
                expr_arena,
            )? {
                #[cfg(feature = "iejoin")]
                try_rewrite_outer_iejoin(
                    schema_left,
                    schema_right,
                    output_schema,
                    options,
                    left_on,
                    right_on,
                    expr_arena,
                )?;
            }
            return PolarsResult::Ok(());
        }

        let is_rewrite_candidate = match &options.options {
            // non-equi joins
            JoinTypeOptionsIR::CrossAndFilter { .. } => true,
            // Range joins
            #[cfg(feature = "iejoin")]
            JoinTypeOptionsIR::IEJoin { .. } | JoinTypeOptionsIR::Range { .. } => true,
            _ => options.args.how.is_cross(),
        };
        if !is_rewrite_candidate {
            return PolarsResult::Ok(());
        }

        if matches!(&options.options, JoinTypeOptionsIR::CrossAndFilter { .. }) {
            let JoinTypeOptionsIR::CrossAndFilter { predicate } =
                std::mem::take(&mut Arc::make_mut(options).options)
            else {
                unreachable!()
            };

            insert_predicate_dedup(acc_predicates, &predicate, expr_arena, dedup);
        }

        // We are in a cross join + filter
        // Try converting to inner join
        assert!(matches!(options.args.how, JoinType::Cross));
        let equality_conditions = take_equi_join_keys(
            acc_predicates,
            expr_arena,
            schema_left,
            schema_right,
            &suffix,
        )?;

        let equality_conditions: Vec<_> = equality_conditions.collect();
        if !equality_conditions.is_empty() {
            let join_options = Arc::make_mut(options);
            join_options.args.how = JoinType::Inner;
            join_options.args.coalesce = JoinCoalesce::KeepColumns;
            push_equi_join_keys(
                equality_conditions,
                expr_arena,
                schema_right,
                &suffix,
                left_on,
                right_on,
            );
            return Ok(());
        }

        // Try converting cross join to double-bounded RangeJoin.
        #[cfg(feature = "iejoin")]
        if streaming && matches!(options.args.maintain_order, MaintainOrderJoin::None) {
            assert!(left_on.is_empty());
            let range_predicate = take_double_bounded_range_join_filter(
                acc_predicates,
                expr_arena,
                schema_left,
                schema_right,
                output_schema,
                &suffix,
                dedup,
            )?;
            if let Some((bound_lower, bound_upper, left_is_point)) = range_predicate {
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

                let join_options = Arc::make_mut(options);
                join_options.args.how = JoinType::Range;
                join_options.options = JoinTypeOptionsIR::Range {
                    ie_options: IEJoinOptions {
                        operator1: bound_lower.ie_op,
                        operator2: Some(bound_upper.ie_op),
                    },
                    left_on: left_on.clone(),
                    right_on: right_on.clone(),
                };
                return Ok(());
            }
        }

        // Try converting cross join to IEJoin.
        #[cfg(feature = "iejoin")]
        if matches!(options.args.maintain_order, MaintainOrderJoin::None) {
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
                let pred = ie_conditions.into_iter().next().unwrap();
                left_on.push(ExprIR::from_node(pred.input_lhs, expr_arena));
                let mut rexpr = ExprIR::from_node(pred.input_rhs, expr_arena);
                remove_suffix(&mut rexpr, expr_arena, schema_right, &suffix);
                right_on.push(rexpr);

                let join_options = Arc::make_mut(options);
                join_options.args.how = JoinType::Range;
                join_options.options = JoinTypeOptionsIR::Range {
                    ie_options: IEJoinOptions {
                        operator1: pred.ie_op,
                        operator2: None,
                    },
                    left_on: left_on.clone(),
                    right_on: right_on.clone(),
                };
                return Ok(());
            }

            let mut ie_options = IEJoinOptions::default();

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
                        dedup,
                    );
                } else {
                    left_on.push(ExprIR::from_node(input_lhs, expr_arena));
                    let mut rexpr = ExprIR::from_node(input_rhs, expr_arena);
                    remove_suffix(&mut rexpr, expr_arena, schema_right, &suffix);
                    right_on.push(rexpr);

                    match left_on.len() {
                        1 => ie_options.operator1 = ie_op,
                        2 => ie_options.operator2 = Some(ie_op),
                        _ => unreachable!("{}", IEJOIN_MAX_PREDICATES),
                    };
                }
            }

            if options.args.how == JoinType::IEJoin {
                let join_options = Arc::make_mut(options);
                join_options.options = JoinTypeOptionsIR::IEJoin {
                    ie_options,
                    left_on: left_on.clone(),
                    right_on: right_on.clone(),
                };
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

        let existing = std::mem::replace(
            &mut Arc::make_mut(options).options,
            JoinTypeOptionsIR::CrossAndFilter {
                predicate: ExprIR::from_node(nested_loop_predicates, expr_arena),
            },
        );
        // Important
        assert!(
            matches!(existing, JoinTypeOptionsIR::Equi { ref on, fused_predicate: None } if on.is_empty())
        );

        Ok(())
    })()?;

    // The rewrites below reason about `left_on`/`right_on` as equality keys, which does
    // not hold when the match condition has a non-equality component.
    if options.is_non_equi() {
        return Ok(None);
    }

    // Only equality keys reach here; the non-equi paths install their own condition above.
    Arc::make_mut(options)
        .options
        .set_keys(left_on.clone(), right_on.clone());

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
    if options.args.should_coalesce() && matches!(options.args.how, JoinType::Right) {
        coalesced_to_right = lhs_input_column_keys_iter!().collect();
    }

    let mut non_null_side = ExprOrigin::None;

    for predicate in acc_predicates.values() {
        for node in MintermIter::new(predicate.node(), expr_arena) {
            predicate_non_null_column_outputs(node, expr_arena, &mut |non_null_column| {
                non_null_side |=
                    non_null_side_for_column(non_null_column, schema_left, schema_right, options);
            });
        }
    }

    let Some(new_join_type) = downgraded_join_type(&options.args.how, non_null_side) else {
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
            det_join_schema(schema_left, schema_right, options).unwrap(),
        ),
        _ => {
            debug_assert_eq!(
                output_schema,
                &det_join_schema(schema_left, schema_right, options).unwrap()
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

                for (l, r) in left_on.iter().zip(right_on.iter()) {
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

                for (l, r) in left_on.iter().zip(right_on.iter()) {
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

    // An outer join whose ON condition was fully pushed down keeps a `true == true` key.
    // Once inner, that key only costs a hash join over a single bucket.
    if new_join_type == JoinType::Inner
        && !options.args.should_coalesce()
        && !options.args.validation.needs_checks()
        && matches!(&options.options, JoinTypeOptionsIR::Equi { on, fused_predicate: None } if !on.is_empty())
        && left_on.iter().zip(right_on.iter()).all(|(l, r)| {
            matches!(
                (expr_arena.get(l.node()), expr_arena.get(r.node())),
                (AExpr::Literal(l), AExpr::Literal(r)) if l.bool() == Some(true) && r.bool() == Some(true)
            )
        })
    {
        options.args.how = JoinType::Cross;
        options.options = JoinTypeOptionsIR::Equi {
            on: Vec::new(),
            fused_predicate: None,
        };
        left_on.clear();
        right_on.clear();
    }

    Ok(project_to_original.map(|p| (p, original_output_schema)))
}

/// Recognize an equi join independently of inequality algorithm restrictions.
fn try_rewrite_outer_equi_join(
    schema_left: &SchemaRef,
    schema_right: &SchemaRef,
    options: &mut Arc<JoinOptionsIR>,
    left_on: &mut Vec<ExprIR>,
    right_on: &mut Vec<ExprIR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<bool> {
    let JoinTypeOptionsIR::CrossAndFilter { predicate } = &options.options else {
        return Ok(false);
    };
    if !commutes_with_filter(predicate, expr_arena) {
        return Ok(false);
    }
    let suffix = options.args.suffix().clone();
    let mut remaining = init_indexmap(None);
    let mut dedup = PredicateDedupState::default();
    insert_predicate_dedup(&mut remaining, predicate, expr_arena, &mut dedup);
    let keys: Vec<_> = take_equi_join_keys(
        &mut remaining,
        expr_arena,
        schema_left,
        schema_right,
        &suffix,
    )?
    .collect();
    if !remaining.is_empty() || keys.is_empty() {
        return Ok(false);
    }
    let on = push_equi_join_keys(keys, expr_arena, schema_right, &suffix, left_on, right_on);
    Arc::make_mut(options).options = JoinTypeOptionsIR::Equi {
        on,
        fused_predicate: None,
    };
    Ok(true)
}

/// Convert an outer ON condition to IEJoin when it captures the whole predicate.
#[cfg(feature = "iejoin")]
fn try_rewrite_outer_iejoin(
    schema_left: &SchemaRef,
    schema_right: &SchemaRef,
    output_schema: &Schema,
    options: &mut Arc<JoinOptionsIR>,
    left_on: &mut Vec<ExprIR>,
    right_on: &mut Vec<ExprIR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<()> {
    use polars_utils::itertools::Itertools;

    let JoinTypeOptionsIR::CrossAndFilter { predicate } = &options.options else {
        return Ok(());
    };
    let predicate = predicate.clone();

    let suffix = options.args.suffix().clone();

    if matches!(options.args.maintain_order, MaintainOrderJoin::None) {
        let mut on_local: PlIndexMap<PlSmallStr, ExprIR> = init_indexmap(None);
        let mut local_dedup = PredicateDedupState::default();
        insert_predicate_dedup(&mut on_local, &predicate, expr_arena, &mut local_dedup);

        let ie_conditions = take_iejoin_compatible_filters(
            &mut on_local,
            expr_arena,
            schema_left,
            schema_right,
            output_schema,
            &suffix,
        )?
        .collect_vec();

        if on_local.is_empty()
            && !ie_conditions.is_empty()
            && ie_conditions.len() <= IEJOIN_MAX_PREDICATES
        {
            let mut ie_options = IEJoinOptions::default();
            for (
                i,
                IEJoinCompatiblePredicate {
                    input_lhs,
                    input_rhs,
                    ie_op,
                    ..
                },
            ) in ie_conditions.into_iter().enumerate()
            {
                left_on.push(ExprIR::from_node(input_lhs, expr_arena));
                let mut rexpr = ExprIR::from_node(input_rhs, expr_arena);
                remove_suffix(&mut rexpr, expr_arena, schema_right, &suffix);
                right_on.push(rexpr);

                match i {
                    0 => ie_options.operator1 = ie_op,
                    1 => ie_options.operator2 = Some(ie_op),
                    _ => unreachable!("{}", IEJOIN_MAX_PREDICATES),
                }
            }

            Arc::make_mut(options).options = JoinTypeOptionsIR::IEJoin {
                ie_options,
                left_on: left_on.clone(),
                right_on: right_on.clone(),
            };
            return Ok(());
        }
    }

    Ok(())
}

struct EquiJoinKeys {
    input_lhs: Node,
    input_rhs: Node,
}

/// Append `keys` to `left_on`/`right_on` and return them as pairs.
fn push_equi_join_keys(
    keys: impl IntoIterator<Item = EquiJoinKeys>,
    expr_arena: &mut Arena<AExpr>,
    schema_right: &Schema,
    suffix: &str,
    left_on: &mut Vec<ExprIR>,
    right_on: &mut Vec<ExprIR>,
) -> Vec<(ExprIR, ExprIR)> {
    keys.into_iter()
        .map(
            |EquiJoinKeys {
                 input_lhs,
                 input_rhs,
             }| {
                let left = ExprIR::from_node(input_lhs, expr_arena);
                let mut right = ExprIR::from_node(input_rhs, expr_arena);
                remove_suffix(&mut right, expr_arena, schema_right, suffix);
                left_on.push(left.clone());
                right_on.push(right.clone());
                (left, right)
            },
        )
        .collect()
}

fn take_equi_join_keys(
    acc_predicates: &mut PlIndexMap<PlSmallStr, ExprIR>,
    expr_arena: &mut Arena<AExpr>,
    schema_left: &Schema,
    schema_right: &Schema,
    suffix: &str,
) -> PolarsResult<indexmap::map::IntoValues<Node, EquiJoinKeys>> {
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
                    (ExprOrigin::Left, ExprOrigin::Right) => Some(EquiJoinKeys {
                        input_lhs: *left,
                        input_rhs: *right,
                    }),
                    (ExprOrigin::Right, ExprOrigin::Left) => Some(EquiJoinKeys {
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
