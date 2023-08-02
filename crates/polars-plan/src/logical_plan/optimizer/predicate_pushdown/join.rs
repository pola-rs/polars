use super::*;

fn should_block_join_specific(ae: &AExpr, how: &JoinType) -> bool {
    use AExpr::*;
    match ae {
        // joins can produce null values
        Function {
            function:
                FunctionExpr::Boolean(BooleanFunction::IsNotNull)
                | FunctionExpr::Boolean(BooleanFunction::IsNull)
                | FunctionExpr::FillNull { .. },
            ..
        } => join_produces_null(how),
        #[cfg(feature = "is_in")]
        Function {
            function: FunctionExpr::Boolean(BooleanFunction::IsIn),
            ..
        } => join_produces_null(how),
        // joins can produce duplicates
        #[cfg(feature = "is_unique")]
        Function {
            function:
                FunctionExpr::Boolean(BooleanFunction::IsUnique)
                | FunctionExpr::Boolean(BooleanFunction::IsDuplicated),
            ..
        } => true,
        #[cfg(feature = "is_first")]
        Function {
            function: FunctionExpr::Boolean(BooleanFunction::IsFirst),
            ..
        } => true,
        // any operation that checks for equality or ordering can be wrong because
        // the join can produce null values
        // TODO! check if we can be less conservative here
        BinaryExpr { op, .. } => !matches!(op, Operator::NotEq) && join_produces_null(how),
        _ => false,
    }
}

fn join_produces_null(how: &JoinType) -> bool {
    #[cfg(feature = "asof_join")]
    {
        matches!(
            how,
            JoinType::Left | JoinType::Outer | JoinType::Cross | JoinType::AsOf(_)
        )
    }
    #[cfg(not(feature = "asof_join"))]
    {
        matches!(how, JoinType::Left | JoinType::Outer | JoinType::Cross)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_join(
    opt: &PredicatePushDown,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
    input_left: Node,
    input_right: Node,
    left_on: Vec<Node>,
    right_on: Vec<Node>,
    schema: SchemaRef,
    options: Arc<JoinOptions>,
    acc_predicates: PlHashMap<Arc<str>, Node>,
) -> PolarsResult<ALogicalPlan> {
    use ALogicalPlan::*;
    let schema_left = lp_arena.get(input_left).schema(lp_arena);
    let schema_right = lp_arena.get(input_right).schema(lp_arena);

    let mut pushdown_left = init_hashmap(Some(acc_predicates.len()));
    let mut pushdown_right = init_hashmap(Some(acc_predicates.len()));
    let mut local_predicates = Vec::with_capacity(acc_predicates.len());

    for (_, predicate) in acc_predicates {
        // check if predicate can pass the joins node
        if has_aexpr(predicate, expr_arena, |ae| {
            should_block_join_specific(ae, &options.args.how)
        }) {
            local_predicates.push(predicate);
            continue;
        }
        // these indicate to which tables we are going to push down the predicate
        let mut filter_left = false;
        let mut filter_right = false;

        // predicate should not have an aggregation or window function as that would
        // be influenced by join
        #[allow(clippy::suspicious_else_formatting)]
        if !predicate_is_pushdown_boundary(predicate, expr_arena) {
            if check_input_node(predicate, &schema_left, expr_arena) {
                insert_and_combine_predicate(&mut pushdown_left, predicate, expr_arena);
                filter_left = true;
            }
            // this is `else if` because if the predicate is in the left hand side
            // the right hand side should be renamed with the suffix.
            // in that case we should not push down as the user wants to filter on `x`
            // not on `x_rhs`.
            else if check_input_node(predicate, &schema_right, expr_arena) {
                insert_and_combine_predicate(&mut pushdown_right, predicate, expr_arena);
                filter_right = true;
            }
        }
        match (filter_left, filter_right, &options.args.how) {
            // if not pushed down on one of the tables we have to do it locally.
            (false, false, _) |
            // if left join and predicate only available in right table,
            // 'we should not filter right, because that would lead to
            // invalid results.
            // see: #2057
            (false, true, JoinType::Left)
            => {
                local_predicates.push(predicate);
                continue;
            },
            // business as usual
            _ => {}
        }
    }

    opt.pushdown_and_assign(input_left, pushdown_left, lp_arena, expr_arena)?;
    opt.pushdown_and_assign(input_right, pushdown_right, lp_arena, expr_arena)?;

    let lp = Join {
        input_left,
        input_right,
        left_on,
        right_on,
        schema,
        options,
    };
    Ok(opt.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
}
