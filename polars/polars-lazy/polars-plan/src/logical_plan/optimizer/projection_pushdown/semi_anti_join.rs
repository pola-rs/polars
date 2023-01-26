use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_semi_anti_join(
    proj_pd: &mut ProjectionPushDown,
    input_left: Node,
    input_right: Node,
    left_on: Vec<Node>,
    right_on: Vec<Node>,
    options: JoinOptions,
    acc_projections: Vec<Node>,
    _projected_names: PlHashSet<Arc<str>>,
    projections_seen: usize,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<ALogicalPlan> {
    proj_pd.has_joins_or_unions = true;
    // n = 0 if no projections, so we don't allocate unneeded
    let n = acc_projections.len() * 2;
    let mut pushdown_left = Vec::with_capacity(n);
    let mut pushdown_right = Vec::with_capacity(n);
    let mut names_left = PlHashSet::with_capacity(n);
    let mut names_right = PlHashSet::with_capacity(n);
    let mut local_projection = Vec::with_capacity(n);

    // if there are no projections we don't have to do anything (all columns are projected)
    // otherwise we build local projections to sort out proper column names due to the
    // join operation
    //
    // Joins on columns with different names, for example
    // left_on = "a", right_on = "b
    // will remove the name "b" (it is "a" now). That columns should therefore not
    // be added to a local projection.
    if !acc_projections.is_empty() {
        let schema_left = lp_arena.get(input_left).schema(lp_arena);
        let schema_right = lp_arena.get(input_right).schema(lp_arena);

        // We need the join columns so we push the projection downwards
        for e in &left_on {
            add_expr_to_accumulated(*e, &mut pushdown_left, &mut names_left, expr_arena);
        }
        for e in &right_on {
            add_expr_to_accumulated(*e, &mut pushdown_right, &mut names_right, expr_arena);
        }

        for proj in acc_projections {
            let mut add_local = true;

            // if it is an alias we want to project the leaf column name downwards
            // but we don't want to project it a this level, otherwise we project both
            // the root and the alias, hence add_local = false.
            if let AExpr::Alias(expr, name) = expr_arena.get(proj).clone() {
                for root_name in aexpr_to_leaf_names(expr, expr_arena) {
                    let node = expr_arena.add(AExpr::Column(root_name));
                    let proj = expr_arena.add(AExpr::Alias(node, name.clone()));
                    local_projection.push(proj)
                }
                // now we don't
                add_local = false;
            }

            proj_pd.join_push_down(
                &schema_left,
                &schema_right,
                proj,
                &mut pushdown_left,
                &mut pushdown_right,
                &mut names_left,
                &mut names_right,
                expr_arena,
            );
            if add_local {
                // always also do the projection locally, because the join columns may not be
                // included in the projection.
                // for instance:
                //
                // SELECT [COLUMN temp]
                // FROM
                // JOIN (["days", "temp"]) WITH (["days", "rain"]) ON (left: days right: days)
                //
                // should drop the days column after the join.
                local_projection.push(proj);
            }
        }
    }

    proj_pd.pushdown_and_assign(
        input_left,
        pushdown_left,
        names_left,
        projections_seen,
        lp_arena,
        expr_arena,
    )?;
    proj_pd.pushdown_and_assign(
        input_right,
        pushdown_right,
        names_right,
        projections_seen,
        lp_arena,
        expr_arena,
    )?;

    let alp = ALogicalPlanBuilder::new(input_left, expr_arena, lp_arena)
        .join(input_right, left_on, right_on, options)
        .build();

    let root = lp_arena.add(alp);
    let builder = ALogicalPlanBuilder::new(root, expr_arena, lp_arena);

    Ok(proj_pd.finish_node(local_projection, builder))
}
