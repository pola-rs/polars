use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_groupby(
    proj_pd: &mut ProjectionPushDown,
    input: Node,
    keys: Vec<Node>,
    aggs: Vec<Node>,
    apply: Option<Arc<dyn DataFrameUdf>>,
    schema: SchemaRef,
    maintain_order: bool,
    options: GroupbyOptions,
    acc_projections: Vec<Node>,
    projected_names: PlHashSet<Arc<str>>,
    projections_seen: usize,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<ALogicalPlan> {
    use ALogicalPlan::*;

    // the custom function may need all columns so we do the projections here.
    if let Some(f) = apply {
        let lp = Aggregate {
            input,
            keys,
            aggs,
            schema,
            apply: Some(f),
            maintain_order,
            options,
        };
        let input = lp_arena.add(lp);

        let builder = ALogicalPlanBuilder::new(input, expr_arena, lp_arena);
        Ok(proj_pd.finish_node(acc_projections, builder))
    } else {
        let has_pushed_down = !acc_projections.is_empty();

        // todo! remove unnecessary vec alloc.
        let (mut acc_projections, _local_projections, mut names) = split_acc_projections(
            acc_projections,
            lp_arena.get(input).schema(lp_arena).as_ref(),
            expr_arena,
            false,
        );

        // add the columns used in the aggregations to the projection only if they are used upstream
        let projected_aggs: Vec<Node> = aggs
            .into_iter()
            .filter(|agg| {
                if has_pushed_down && projections_seen > 0 {
                    expr_is_projected_upstream(agg, input, lp_arena, expr_arena, &projected_names)
                } else {
                    true
                }
            })
            .collect();

        for agg in &projected_aggs {
            add_expr_to_accumulated(*agg, &mut acc_projections, &mut names, expr_arena);
        }

        // make sure the keys are projected
        for key in &*keys {
            add_expr_to_accumulated(*key, &mut acc_projections, &mut names, expr_arena);
        }

        // make sure that the dynamic key is projected
        #[cfg(feature = "dynamic_groupby")]
        if let Some(options) = &options.dynamic {
            let node = expr_arena.add(AExpr::Column(Arc::from(options.index_column.as_str())));
            add_expr_to_accumulated(node, &mut acc_projections, &mut names, expr_arena);
        }
        // make sure that the rolling key is projected
        #[cfg(feature = "dynamic_groupby")]
        if let Some(options) = &options.rolling {
            let node = expr_arena.add(AExpr::Column(Arc::from(options.index_column.as_str())));
            add_expr_to_accumulated(node, &mut acc_projections, &mut names, expr_arena);
        }

        proj_pd.pushdown_and_assign(
            input,
            acc_projections,
            names,
            projections_seen,
            lp_arena,
            expr_arena,
        )?;

        let builder = ALogicalPlanBuilder::new(input, expr_arena, lp_arena).groupby(
            keys,
            projected_aggs,
            apply,
            maintain_order,
            options,
        );
        Ok(builder.build())
    }
}
