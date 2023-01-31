use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_generic(
    proj_pd: &mut ProjectionPushDown,
    lp: ALogicalPlan,
    acc_projections: Vec<Node>,
    projected_names: PlHashSet<Arc<str>>,
    projections_seen: usize,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<ALogicalPlan> {
    let inputs = lp.get_inputs();
    let exprs = lp.get_exprs();

    let mut first_schema = None;
    let mut names = None;

    let new_inputs = inputs
        .iter()
        .enumerate()
        .map(|(i, &node)| {
            let alp = lp_arena.take(node);
            let mut alp = proj_pd.push_down(
                alp,
                acc_projections.clone(),
                projected_names.clone(),
                projections_seen,
                lp_arena,
                expr_arena,
            )?;

            // double projection can mess up the schema ordering
            // here we ensure the ordering is maintained.
            if !acc_projections.is_empty() && inputs.len() > 1 {
                let schema = alp.schema(lp_arena);
                if i == 0 {
                    first_schema = Some(schema.into_owned());
                } else if first_schema.as_ref().unwrap() != schema.as_ref() {
                    if names.is_none() {
                        names = Some(
                            first_schema
                                .as_ref()
                                .unwrap()
                                .iter()
                                .map(|(name, _)| {
                                    expr_arena.add(AExpr::Column(Arc::from(name.as_str())))
                                })
                                .collect::<Vec<_>>(),
                        );
                    }
                    alp = ALogicalPlanBuilder::from_lp(alp, expr_arena, lp_arena)
                        .project(names.as_ref().unwrap().clone())
                        .build()
                }
            }
            lp_arena.replace(node, alp);
            Ok(node)
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    Ok(lp.with_exprs_and_input(exprs, new_inputs))
}
