use super::*;

#[inline]
fn is_count(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    matches!(expr_arena.get(node), AExpr::Len)
}

/// In this function we check a double projection case
/// df
///   .select(col("foo").alias("bar"))
///   .select(col("bar")
///
/// In this query, bar cannot pass this projection, as it would not exist in DF.
/// THE ORDER IS IMPORTANT HERE!
/// this removes projection names, so any checks to upstream names should
/// be done before this branch.
fn check_double_projection(
    expr: &ExprIR,
    expr_arena: &mut Arena<AExpr>,
    acc_projections: &mut Vec<ColumnNode>,
    projected_names: &mut PlHashSet<Arc<str>>,
) {
    // Factor out the pruning function
    fn prune_projections_by_name(
        acc_projections: &mut Vec<ColumnNode>,
        name: &str,
        expr_arena: &Arena<AExpr>,
    ) {
        acc_projections.retain(|node| column_node_to_name(*node, expr_arena).as_ref() != name);
    }
    if let Some(name) = expr.get_non_projected_name() {
        if projected_names.remove(name) {
            prune_projections_by_name(acc_projections, name.as_ref(), expr_arena)
        }
    }

    for (_, ae) in (&*expr_arena).iter(expr.node()) {
        if let AExpr::Literal(LiteralValue::Series(s)) = ae {
            let name = s.name();
            if projected_names.remove(name) {
                prune_projections_by_name(acc_projections, name, expr_arena)
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_projection(
    proj_pd: &mut ProjectionPushDown,
    input: Node,
    mut exprs: Vec<ExprIR>,
    mut acc_projections: Vec<ColumnNode>,
    mut projected_names: PlHashSet<Arc<str>>,
    projections_seen: usize,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<IR> {
    let mut local_projection = Vec::with_capacity(exprs.len());

    // path for `SELECT count(*) FROM`
    // as there would be no projections and we would read
    // the whole file while we only want the count
    if exprs.len() == 1 && is_count(exprs[0].node(), expr_arena) {
        let input_schema = lp_arena.get(input).schema(lp_arena);
        let expr = if input_schema.is_empty() {
            // If the input schema is empty, we should just project
            // ourselves
            exprs[0].node()
        } else {
            // simply select the last column
            // NOTE: the first can be the inserted index column, so that might not work
            let (first_name, _) = input_schema.try_get_at_index(input_schema.len() - 1)?;
            let expr = expr_arena.add(AExpr::Column(ColumnName::from(first_name.as_str())));
            if !acc_projections.is_empty() {
                check_double_projection(
                    &exprs[0],
                    expr_arena,
                    &mut acc_projections,
                    &mut projected_names,
                );
            }
            expr
        };
        add_expr_to_accumulated(expr, &mut acc_projections, &mut projected_names, expr_arena);
        local_projection.push(exprs.pop().unwrap());
        proj_pd.is_count_star = true;
    } else {
        // A projection can consist of a chain of expressions followed by an alias.
        // We want to do the chain locally because it can have complicated side effects.
        // The only thing we push down is the root name of the projection.
        // So we:
        //      - add the root of the projections to accumulation,
        //      - also do the complete projection locally to keep the schema (column order) and the alias.

        // set this flag outside the loop as we modify within the loop
        let has_pushed_down = !acc_projections.is_empty();
        for e in exprs {
            if has_pushed_down {
                // remove projections that are not used upstream
                if !projected_names.contains(e.output_name_arc()) {
                    continue;
                }

                check_double_projection(&e, expr_arena, &mut acc_projections, &mut projected_names);
            }
            // do local as we still need the effect of the projection
            // e.g. a projection is more than selecting a column, it can
            // also be a function/ complicated expression
            local_projection.push(e);
        }

        // After we have checked double projections, we add the projections to the accumulated state.
        // We do this in two passes, otherwise we mutate while checking.
        for e in &local_projection {
            add_expr_to_accumulated(
                e.node(),
                &mut acc_projections,
                &mut projected_names,
                expr_arena,
            );
        }
    }
    proj_pd.pushdown_and_assign(
        input,
        acc_projections,
        projected_names,
        projections_seen + 1,
        lp_arena,
        expr_arena,
    )?;

    let builder = IRBuilder::new(input, expr_arena, lp_arena);
    let lp = proj_pd.finish_node(local_projection, builder);

    Ok(lp)
}
