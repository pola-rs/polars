use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_group_by(
    opt: &PredicatePushDown,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    input: Node,
    keys: Vec<ExprIR>,
    aggs: Vec<ExprIR>,
    schema: SchemaRef,
    maintain_order: bool,
    apply: Option<Arc<dyn DataFrameUdf>>,
    options: Arc<GroupbyOptions>,
    acc_predicates: PlHashMap<Arc<str>, ExprIR>,
) -> PolarsResult<IR> {
    use IR::*;

    #[cfg(feature = "dynamic_group_by")]
    let no_push = { options.rolling.is_some() || options.dynamic.is_some() };

    #[cfg(not(feature = "dynamic_group_by"))]
    let no_push = false;

    // Don't pushdown predicates on these cases.
    if apply.is_some() || no_push || options.slice.is_some() {
        let lp = GroupBy {
            input,
            keys,
            aggs,
            schema,
            apply,
            maintain_order,
            options,
        };
        return opt.no_pushdown_restart_opt(lp, acc_predicates, lp_arena, expr_arena);
    }

    // If the predicate only resolves to the keys we can push it down.
    // When it filters the aggregations, the predicate should be done after aggregation.
    let mut local_predicates = Vec::with_capacity(acc_predicates.len());
    let key_schema = aexprs_to_schema(
        &keys,
        lp_arena.get(input).schema(lp_arena).as_ref(),
        Context::Default,
        expr_arena,
    );

    let mut new_acc_predicates = PlHashMap::with_capacity(acc_predicates.len());

    for (pred_name, predicate) in acc_predicates {
        // Counts change due to groupby's
        // TODO! handle aliases, so that the predicate that is pushed down refers to the column before alias.
        let mut push_down = !has_aexpr(predicate.node(), expr_arena, |ae| {
            matches!(ae, AExpr::Len | AExpr::Alias(_, _))
        });

        for name in aexpr_to_leaf_names_iter(predicate.node(), expr_arena) {
            push_down &= key_schema.contains(name.as_ref());

            if !push_down {
                break;
            }
        }
        if !push_down {
            local_predicates.push(predicate)
        } else {
            new_acc_predicates.insert(pred_name.clone(), predicate.clone());
        }
    }

    opt.pushdown_and_assign(input, new_acc_predicates, lp_arena, expr_arena)?;

    let lp = GroupBy {
        input,
        keys,
        aggs,
        schema,
        apply,
        maintain_order,
        options,
    };
    Ok(opt.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
}
