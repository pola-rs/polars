use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_semi_anti_join(
    proj_pd: &mut ProjectionPushDown,
    input_left: Node,
    input_right: Node,
    left_on: Vec<ExprIR>,
    right_on: Vec<ExprIR>,
    options: Arc<JoinOptions>,
    acc_projections: Vec<ColumnNode>,
    _projected_names: PlHashSet<Arc<str>>,
    projections_seen: usize,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<IR> {
    // n = 0 if no projections, so we don't allocate unneeded
    let n = acc_projections.len() * 2;
    let mut pushdown_left = Vec::with_capacity(n);
    let mut pushdown_right = Vec::with_capacity(n);
    let mut names_left = PlHashSet::with_capacity(n);
    let mut names_right = PlHashSet::with_capacity(n);

    if acc_projections.is_empty() {
        // Only project the join columns.
        for e in &right_on {
            add_expr_to_accumulated(e.node(), &mut pushdown_right, &mut names_right, expr_arena);
        }
    } else {
        // We build local projections to sort out proper column names due to the
        // join operation.
        // Joins on columns with different names, for example
        // left_on = "a", right_on = "b
        // will remove the name "b" (it is "a" now). That columns should therefore not
        // be added to a local projection.
        let schema_left = lp_arena.get(input_left).schema(lp_arena);
        let schema_right = lp_arena.get(input_right).schema(lp_arena);

        // We need the join columns so we push the projection downwards
        for e in &left_on {
            add_expr_to_accumulated(e.node(), &mut pushdown_left, &mut names_left, expr_arena);
        }
        for e in &right_on {
            add_expr_to_accumulated(e.node(), &mut pushdown_right, &mut names_right, expr_arena);
        }

        for proj in acc_projections {
            let _ = proj_pd.join_push_down(
                &schema_left,
                &schema_right,
                proj,
                &mut pushdown_left,
                &mut pushdown_right,
                &mut names_left,
                &mut names_right,
                expr_arena,
            );
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

    let alp = IRBuilder::new(input_left, expr_arena, lp_arena)
        .join(input_right, left_on, right_on, options)
        .build();

    let root = lp_arena.add(alp);
    let builder = IRBuilder::new(root, expr_arena, lp_arena);

    Ok(proj_pd.finish_node(vec![], builder))
}
