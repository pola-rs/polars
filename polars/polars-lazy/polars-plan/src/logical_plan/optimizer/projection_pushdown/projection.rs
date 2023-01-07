use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_projection(
    proj_pd: &mut ProjectionPushDown,
    input: Node,
    exprs: Vec<Node>,
    mut acc_projections: Vec<Node>,
    mut projected_names: PlHashSet<Arc<str>>,
    projections_seen: usize,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<ALogicalPlan> {
    let mut local_projection = Vec::with_capacity(exprs.len());
    // A projection can consist of a chain of expressions followed by an alias.
    // We want to do the chain locally because it can have complicated side effects.
    // The only thing we push down is the root name of the projection.
    // So we:
    //      - add the root of the projections to accumulation,
    //      - also do the complete projection locally to keep the schema (column order) and the alias.

    // set this flag outside the loop as we modify within the loop
    let has_pushed_down = !acc_projections.is_empty();
    for e in &exprs {
        if has_pushed_down {
            // remove projections that are not used upstream
            if !expr_is_projected_upstream(e, input, lp_arena, expr_arena, &projected_names) {
                continue;
            }

            // in this branch we check a double projection case
            // df
            //   .select(col("foo").alias("bar"))
            //   .select(col("bar")
            //
            // In this query, bar cannot pass this projection, as it would not exist in DF.
            // THE ORDER IS IMPORTANT HERE!
            // this removes projection names, so any checks to upstream names should
            // be done before this branch.
            for (_, ae) in (&*expr_arena).iter(*e) {
                if let AExpr::Alias(_, name) = ae {
                    if projected_names.remove(name) {
                        acc_projections
                            .retain(|expr| !aexpr_to_leaf_names(*expr, expr_arena).contains(name));
                    }
                }
            }
        }
        // do local as we still need the effect of the projection
        // e.g. a projection is more than selecting a column, it can
        // also be a function/ complicated expression
        local_projection.push(*e);

        add_expr_to_accumulated(*e, &mut acc_projections, &mut projected_names, expr_arena);
    }

    proj_pd.pushdown_and_assign(
        input,
        acc_projections,
        projected_names,
        projections_seen + 1,
        lp_arena,
        expr_arena,
    )?;

    let builder = ALogicalPlanBuilder::new(input, expr_arena, lp_arena);
    let lp = proj_pd.finish_node(local_projection, builder);

    Ok(lp)
}
