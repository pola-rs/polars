use super::*;

#[inline]
pub(super) fn is_count(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    matches!(expr_arena.get(node), AExpr::Len)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_projection(
    proj_pd: &mut ProjectionPushDown,
    input: Node,
    mut exprs: Vec<ExprIR>,
    mut acc_projections: Vec<ColumnNode>,
    mut projected_names: PlHashSet<PlSmallStr>,
    projections_seen: usize,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    // Whether is SimpleProjection.
    simple: bool,
) -> PolarsResult<IR> {
    let mut local_projection = Vec::with_capacity(exprs.len());

    // Special path for `SELECT count(*) FROM`
    // as there would be no projections and we would read
    // the whole file while we only want the count
    if exprs.len() == 1 && is_count(exprs[0].node(), expr_arena) {
        let input_schema = lp_arena.get(input).schema(lp_arena);
        let expr = if input_schema.is_empty() {
            // If the input schema is empty, we should just project
            // ourselves
            exprs[0].node()
        } else {
            // Select the last column projection.
            let (last_name, _) = input_schema.try_get_at_index(input_schema.len() - 1)?;

            let name = match lp_arena.get(input) {
                IR::Select { expr: exprs, .. } | IR::HStack { exprs, .. } => (|| {
                    for e in exprs {
                        if !e.is_scalar(expr_arena) {
                            return e.output_name();
                        }
                    }

                    last_name
                })(),

                IR::Scan {
                    file_info,
                    output_schema,
                    ..
                } => {
                    let schema = output_schema.as_ref().unwrap_or(&file_info.schema);
                    // NOTE: the first can be the inserted index column, so that might not work
                    let (last_name, _) = schema.try_get_at_index(schema.len() - 1)?;
                    last_name
                },

                IR::DataFrameScan {
                    schema,
                    output_schema,
                    ..
                } => {
                    // NOTE: the first can be the inserted index column, so that might not work
                    let schema = output_schema.as_ref().unwrap_or(schema);
                    let (last_name, _) = schema.try_get_at_index(schema.len() - 1)?;
                    last_name
                },

                _ => last_name,
            };

            expr_arena.add(AExpr::Column(name.clone()))
        };

        // Clear all accumulated projections since we only project a single column from this level.
        acc_projections.clear();
        projected_names.clear();
        add_expr_to_accumulated(expr, &mut acc_projections, &mut projected_names, expr_arena);
        local_projection.push(exprs.pop().unwrap());
        proj_pd.is_count_star = true;
    } else {
        // `remove_names` tracks projected names that need to be removed as they may be aliased
        // names that are created on this level.
        let mut remove_names = PlHashSet::new();

        // If there are non-scalar projections we must project at least one of them to maintain the
        // output height.
        let mut opt_non_scalar = None;
        let mut projection_has_non_scalar = false;

        let projected_exprs: Vec<ExprIR> = exprs
            .into_iter()
            .filter(|e| {
                let is_non_scalar = !e.is_scalar(expr_arena);

                if opt_non_scalar.is_none() && is_non_scalar {
                    opt_non_scalar = Some(e.clone())
                }

                let name = match e.output_name_inner() {
                    OutputName::LiteralLhs(name) | OutputName::Alias(name) => {
                        remove_names.insert(name.clone());
                        name
                    },
                    #[cfg(feature = "dtype-struct")]
                    OutputName::Field(name) => {
                        remove_names.insert(name.clone());
                        name
                    },
                    OutputName::ColumnLhs(name) => name,
                    OutputName::None => {
                        if cfg!(debug_assertions) {
                            panic!()
                        } else {
                            return false;
                        }
                    },
                };

                let project = acc_projections.is_empty() || projected_names.contains(name);
                projection_has_non_scalar |= project & is_non_scalar;
                project
            })
            .collect();

        // Remove aliased before adding new ones.
        if !remove_names.is_empty() {
            if !projected_names.is_empty() {
                for name in remove_names.iter() {
                    projected_names.remove(name);
                }
            }

            acc_projections.retain(|c| !remove_names.contains(column_node_to_name(*c, expr_arena)));
        }

        for e in projected_exprs {
            add_expr_to_accumulated(
                e.node(),
                &mut acc_projections,
                &mut projected_names,
                expr_arena,
            );

            // do local as we still need the effect of the projection
            // e.g. a projection is more than selecting a column, it can
            // also be a function/ complicated expression
            local_projection.push(e);
        }

        if !projection_has_non_scalar {
            if let Some(non_scalar) = opt_non_scalar {
                add_expr_to_accumulated(
                    non_scalar.node(),
                    &mut acc_projections,
                    &mut projected_names,
                    expr_arena,
                );

                local_projection.push(non_scalar);
            }
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

    let lp = if !local_projection.is_empty() && simple {
        builder
            .project_simple_nodes(local_projection.into_iter().map(|e| e.node()))?
            .build()
    } else {
        proj_pd.finish_node(local_projection, builder)
    };

    Ok(lp)
}
