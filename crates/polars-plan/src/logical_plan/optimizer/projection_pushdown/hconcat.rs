use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_hconcat(
    proj_pd: &mut ProjectionPushDown,
    inputs: Vec<Node>,
    schema: SchemaRef,
    options: HConcatOptions,
    acc_projections: Vec<ColumnNode>,
    projections_seen: usize,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<IR> {
    // When applying projection pushdown to horizontal concatenation,
    // we apply pushdown to all of the inputs using the subset of accumulated projections relevant to each input,
    // then rebuild the concatenated schema.

    let schema = if acc_projections.is_empty() {
        schema
    } else {
        let mut remaining_projections: PlHashSet<_> = acc_projections.into_iter().collect();

        for input in inputs.iter() {
            let mut input_pushdown = Vec::new();
            let input_schema = lp_arena.get(*input).schema(lp_arena);

            for proj in remaining_projections.iter() {
                if check_input_column_node(*proj, input_schema.as_ref(), expr_arena) {
                    input_pushdown.push(*proj);
                }
            }

            let mut input_names = PlHashSet::new();
            for proj in &input_pushdown {
                remaining_projections.remove(proj);
                for name in aexpr_to_leaf_names(proj.0, expr_arena) {
                    input_names.insert(name);
                }
            }
            proj_pd.pushdown_and_assign(
                *input,
                input_pushdown,
                input_names,
                projections_seen,
                lp_arena,
                expr_arena,
            )?;
        }

        let mut schemas = Vec::with_capacity(inputs.len());
        for input in inputs.iter() {
            let schema = lp_arena.get(*input).schema(lp_arena).into_owned();
            schemas.push(schema);
        }
        let new_schema = merge_schemas(&schemas)?;
        Arc::new(new_schema)
    };

    Ok(IR::HConcat {
        inputs,
        schema,
        options,
    })
}
