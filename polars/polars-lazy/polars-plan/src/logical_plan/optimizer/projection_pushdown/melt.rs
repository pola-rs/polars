use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_melt(
    proj_pd: &mut ProjectionPushDown,
    input: Node,
    args: Arc<MeltArgs>,
    schema: SchemaRef,
    acc_projections: Vec<Node>,
    mut projected_names: PlHashSet<Arc<str>>,
    projections_seen: usize,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<ALogicalPlan> {
    // all columns are used in melt
    if args.value_vars.is_empty() {
        // restart projection pushdown
        proj_pd.no_pushdown_restart_opt(
            ALogicalPlan::Melt {
                input,
                args,
                schema,
            },
            acc_projections,
            projections_seen,
            lp_arena,
            expr_arena,
        )
    } else {
        let (mut acc_projections, mut local_projections, names) = split_acc_projections(
            acc_projections,
            lp_arena.get(input).schema(lp_arena).as_ref(),
            expr_arena,
            false,
        );

        if !local_projections.is_empty() {
            local_projections.extend_from_slice(&acc_projections);
        }

        // make sure that the requested columns are projected
        args.id_vars.iter().for_each(|name| {
            add_str_to_accumulated(name, &mut acc_projections, &mut projected_names, expr_arena)
        });
        args.value_vars.iter().for_each(|name| {
            add_str_to_accumulated(name, &mut acc_projections, &mut projected_names, expr_arena)
        });

        proj_pd.pushdown_and_assign(
            input,
            acc_projections,
            names,
            projections_seen,
            lp_arena,
            expr_arena,
        )?;

        let builder = ALogicalPlanBuilder::new(input, expr_arena, lp_arena).melt(args);
        Ok(proj_pd.finish_node(local_projections, builder))
    }
}
