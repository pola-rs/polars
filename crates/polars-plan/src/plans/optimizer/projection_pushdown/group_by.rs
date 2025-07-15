use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_group_by(
    proj_pd: &mut ProjectionPushDown,
    input: Node,
    keys: Vec<ExprIR>,
    aggs: Vec<ExprIR>,
    apply: Option<Arc<dyn DataFrameUdf>>,
    schema: SchemaRef,
    maintain_order: bool,
    options: Arc<GroupbyOptions>,
    ctx: ProjectionContext,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<IR> {
    use IR::*;

    // the custom function may need all columns so we do the projections here.
    if let Some(f) = apply {
        let lp = GroupBy {
            input,
            keys,
            aggs,
            schema,
            apply: Some(f),
            maintain_order,
            options,
        };
        let input = lp_arena.add(lp);

        let builder = IRBuilder::new(input, expr_arena, lp_arena);
        Ok(proj_pd.finish_node_simple_projection(&ctx.acc_projections, builder))
    } else {
        let has_pushed_down = ctx.has_pushed_down();

        // TODO! remove unnecessary vec alloc.
        let (mut acc_projections, _local_projections, mut names) = split_acc_projections(
            ctx.acc_projections,
            lp_arena.get(input).schema(lp_arena).as_ref(),
            expr_arena,
            false,
        );

        // add the columns used in the aggregations to the projection only if they are used upstream
        let projected_aggs = aggs
            .into_iter()
            .filter(|agg| {
                if has_pushed_down && ctx.inner.projections_seen > 0 {
                    ctx.projected_names.contains(agg.output_name())
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();

        for agg in &projected_aggs {
            add_expr_to_accumulated(agg.node(), &mut acc_projections, &mut names, expr_arena);
        }

        // make sure the keys are projected
        for key in &*keys {
            add_expr_to_accumulated(key.node(), &mut acc_projections, &mut names, expr_arena);
        }

        // make sure that the dynamic key is projected
        #[cfg(feature = "dynamic_group_by")]
        if let Some(options) = &options.dynamic {
            let node = expr_arena.add(AExpr::Column(options.index_column.clone()));
            add_expr_to_accumulated(node, &mut acc_projections, &mut names, expr_arena);
        }
        // make sure that the rolling key is projected
        #[cfg(feature = "dynamic_group_by")]
        if let Some(options) = &options.rolling {
            let node = expr_arena.add(AExpr::Column(options.index_column.clone()));
            add_expr_to_accumulated(node, &mut acc_projections, &mut names, expr_arena);
        }
        let ctx = ProjectionContext::new(acc_projections, names, ctx.inner);

        proj_pd.pushdown_and_assign(input, ctx, lp_arena, expr_arena)?;

        let builder = IRBuilder::new(input, expr_arena, lp_arena).group_by(
            keys,
            projected_aggs,
            apply,
            maintain_order,
            options,
        );
        Ok(builder.build())
    }
}
