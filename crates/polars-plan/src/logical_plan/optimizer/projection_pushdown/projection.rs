use super::*;

fn is_count(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    match expr_arena.get(node) {
        AExpr::Alias(node, _) => is_count(*node, expr_arena),
        AExpr::Count => true,
        _ => false,
    }
}

/// Make sure that the rolling key is projected
#[cfg(feature = "dynamic_group_by")]
fn add_rolling_key(
    node: Node,
    expr_arena: &mut Arena<AExpr>,
    acc_projections: &mut Vec<Node>,
    projected_names: &mut PlHashSet<Arc<str>>,
) {
    let arena = expr_arena.clone();
    (&arena).iter(node).for_each(|(_, e)| {
        if let AExpr::Window {
            options: WindowType::Rolling(options),
            ..
        } = e
        {
            let node = expr_arena.add(AExpr::Column(Arc::from(options.index_column.as_str())));
            add_expr_to_accumulated(node, acc_projections, projected_names, expr_arena);
        }
    });
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
    expr: &Node,
    expr_arena: &mut Arena<AExpr>,
    acc_projections: &mut Vec<Node>,
    projected_names: &mut PlHashSet<Arc<str>>,
) {
    for (_, ae) in (&*expr_arena).iter(*expr) {
        if let AExpr::Alias(_, name) = ae {
            if projected_names.remove(name) {
                acc_projections
                    .retain(|expr| !aexpr_to_leaf_names(*expr, expr_arena).contains(name));
            }
        }
    }
}

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

    // path for `SELECT count(*) FROM`
    // as there would be no projections and we would read
    // the whole file while we only want the count
    if exprs.len() == 1 && is_count(exprs[0], expr_arena) {
        let input_schema = lp_arena.get(input).schema(lp_arena);
        // simply select the first column
        let (first_name, _) = input_schema.try_get_at_index(0)?;
        let expr = expr_arena.add(AExpr::Column(Arc::from(first_name.as_str())));
        if !acc_projections.is_empty() {
            check_double_projection(
                &exprs[0],
                expr_arena,
                &mut acc_projections,
                &mut projected_names,
            );
        }
        add_expr_to_accumulated(expr, &mut acc_projections, &mut projected_names, expr_arena);
        local_projection.push(exprs[0]);
    } else {
        // A projection can consist of a chain of expressions followed by an alias.
        // We want to do the chain locally because it can have complicated side effects.
        // The only thing we push down is the root name of the projection.
        // So we:
        //      - add the root of the projections to accumulation,
        //      - also do the complete projection locally to keep the schema (column order) and the alias.

        // set this flag outside the loop as we modify within the loop
        let has_pushed_down = !acc_projections.is_empty();
        for e in &exprs {
            // rolling key is not an expression, but we must take it as projected.
            // Otherwise downstream might remove that column.
            #[cfg(feature = "dynamic_group_by")]
            add_rolling_key(*e, expr_arena, &mut acc_projections, &mut projected_names);

            if has_pushed_down {
                // remove projections that are not used upstream
                if !expr_is_projected_upstream(e, input, lp_arena, expr_arena, &projected_names) {
                    continue;
                }

                check_double_projection(e, expr_arena, &mut acc_projections, &mut projected_names);
            }
            // do local as we still need the effect of the projection
            // e.g. a projection is more than selecting a column, it can
            // also be a function/ complicated expression
            local_projection.push(*e);

            add_expr_to_accumulated(*e, &mut acc_projections, &mut projected_names, expr_arena);
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

    let builder = ALogicalPlanBuilder::new(input, expr_arena, lp_arena);
    let lp = proj_pd.finish_node(local_projection, builder);

    Ok(lp)
}
