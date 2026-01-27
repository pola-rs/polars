use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_unpivot(
    proj_pd: &mut ProjectionPushDown,
    args: &Arc<UnpivotArgsIR>,
    input: Node,
    ctx: ProjectionContext,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<IR> {
    let input_schema = lp_arena.get(input).schema(lp_arena);

    let project_from_input: PlHashSet<&PlSmallStr> =
        args.index.iter().chain(args.on.iter()).collect();
    let (new_projections, new_projected_names): (Vec<ColumnNode>, PlHashSet<PlSmallStr>) =
        input_schema
            .iter_names()
            .filter(|x| project_from_input.contains(x))
            .map(|name| {
                (
                    ColumnNode(expr_arena.add(AExpr::Column(name.clone()))),
                    name.clone(),
                )
            })
            .unzip();

    let local_projections = ctx.acc_projections;
    let new_ctx = ProjectionContext::new(new_projections, new_projected_names, ctx.inner);
    proj_pd.pushdown_and_assign(input, new_ctx, lp_arena, expr_arena)?;

    // re-make unpivot node so that the schema is updated
    let lp = IRBuilder::new(input, expr_arena, lp_arena)
        .unpivot(args.clone())
        .build();

    Ok(IRBuilder::from_lp(lp, expr_arena, lp_arena)
        .project_simple_nodes(local_projections)
        .unwrap()
        .build())
}
