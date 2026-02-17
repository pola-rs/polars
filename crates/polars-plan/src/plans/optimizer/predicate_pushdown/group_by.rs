use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_group_by(
    opt: &mut PredicatePushDown,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    input: Node,
    keys: Vec<ExprIR>,
    aggs: Vec<ExprIR>,
    schema: SchemaRef,
    maintain_order: bool,
    apply: Option<PlanCallback<DataFrame, DataFrame>>,
    options: Arc<GroupbyOptions>,
    acc_predicates: PlHashMap<PlSmallStr, ExprIR>,
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

    // If the predicate only resolves to the keys we can push it down, on the condition
    // that the key values are not modified from their original values.
    // When it filters the aggregations, the predicate should be done after aggregation.
    //
    // For aliased column keys (e.g. col("A").alias("key")), we can still push down by
    // rewriting the predicate to reference the original column name.
    let mut local_predicates = Vec::with_capacity(acc_predicates.len());
    let input_schema = lp_arena.get(input).schema(lp_arena);
    let mut alias_rename_map: PlHashMap<PlSmallStr, PlSmallStr> = PlHashMap::new();
    let mut key_schema = Schema::with_capacity(keys.len());
    for key in &keys {
        if let AExpr::Column(c) = expr_arena.get(key.node()) {
            let output = key.output_name();
            if let Some(dtype) = input_schema.get(c) {
                key_schema.insert(output.clone(), dtype.clone());
                if c != output {
                    alias_rename_map.insert(output.clone(), c.clone());
                }
            }
        }
    }

    let mut new_acc_predicates = PlHashMap::with_capacity(acc_predicates.len());

    for (pred_name, predicate) in acc_predicates {
        // Counts change due to groupby's
        let mut push_down = !has_aexpr(predicate.node(), expr_arena, |ae| matches!(ae, AExpr::Len));

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

    // Rewrite aliased column references in predicates to original column names.
    if !alias_rename_map.is_empty() {
        for (_, predicate) in new_acc_predicates.iter_mut() {
            map_column_references(predicate, expr_arena, &alias_rename_map);
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
