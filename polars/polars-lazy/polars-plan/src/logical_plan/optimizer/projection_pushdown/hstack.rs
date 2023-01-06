use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_hstack(
    proj_pd: &mut ProjectionPushDown,
    input: Node,
    mut exprs: Vec<Node>,
    mut acc_projections: Vec<Node>,
    mut projected_names: PlHashSet<Arc<str>>,
    projections_seen: usize,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<ALogicalPlan> {
    if !acc_projections.is_empty() {
        let input_schema = lp_arena.get(input).schema(lp_arena);
        let mut pruned_with_cols = Vec::with_capacity(exprs.len());

        // Check if output names are used upstream
        // if not, we can prune the `with_column` expression
        // as it is not used in the output.
        for node in &exprs {
            let output_field = expr_arena
                .get(*node)
                .to_field(input_schema.as_ref(), Context::Default, expr_arena)
                .unwrap();
            let output_name = output_field.name();

            let is_used_upstream = projected_names.contains(output_name.as_str());

            if is_used_upstream {
                pruned_with_cols.push(*node);
            }
        }

        if pruned_with_cols.is_empty() {
            proj_pd.pushdown_and_assign(
                input,
                acc_projections,
                projected_names,
                projections_seen,
                lp_arena,
                expr_arena,
            )?;
            return Ok(lp_arena.take(input));
        }

        // Make sure that columns selected with_columns are available
        // only if not empty. If empty we already select everything.
        for expression in &pruned_with_cols {
            add_expr_to_accumulated(
                *expression,
                &mut acc_projections,
                &mut projected_names,
                expr_arena,
            );
        }

        exprs = pruned_with_cols
    }
    // projections that select columns added by
    // this `with_column` operation can be dropped
    // For instance in:
    //
    //  q
    //  .with_column(col("a").alias("b")
    //  .select(["a", "b"])
    //
    // we can drop the "b" projection at this level
    let (acc_projections, _, names) = split_acc_projections(
        acc_projections,
        &lp_arena.get(input).schema(lp_arena),
        expr_arena,
        false,
    );

    proj_pd.pushdown_and_assign(
        input,
        acc_projections,
        names,
        projections_seen,
        lp_arena,
        expr_arena,
    )?;
    let lp = ALogicalPlanBuilder::new(input, expr_arena, lp_arena)
        .with_columns(exprs)
        .build();
    Ok(lp)
}
