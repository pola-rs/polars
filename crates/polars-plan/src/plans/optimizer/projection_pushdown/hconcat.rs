use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_hconcat(
    proj_pd: &mut ProjectionPushDown,
    mut inputs: Vec<Node>,
    schema: SchemaRef,
    options: HConcatOptions,
    ctx: ProjectionContext,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<IR> {
    // When applying projection pushdown to horizontal concatenation,
    // we apply pushdown to all of the inputs using the subset of accumulated projections relevant to each input,
    // then rebuild the concatenated schema.

    let schema = if ctx.acc_projections.is_empty() {
        schema
    } else {
        let mut remaining_projections: PlHashSet<_> = ctx.acc_projections.into_iter().collect();

        let mut result = Ok(());
        inputs.retain(|input| {
            let mut input_pushdown = Vec::new();
            let input_schema = lp_arena.get(*input).schema(lp_arena);

            for proj in remaining_projections.iter() {
                if check_input_column_node(*proj, input_schema.as_ref(), expr_arena) {
                    input_pushdown.push(*proj);
                }
            }

            if input_pushdown.is_empty() {
                // We can ignore this input since no columns are needed.
                return false;
            }

            let mut input_names = PlHashSet::new();
            for proj in &input_pushdown {
                remaining_projections.remove(proj);
                for name in aexpr_to_leaf_names(proj.0, expr_arena) {
                    input_names.insert(name);
                }
            }
            let ctx = ProjectionContext::new(input_pushdown, input_names, ctx.inner);
            if let Err(e) = proj_pd.pushdown_and_assign(*input, ctx, lp_arena, expr_arena) {
                result = Err(e);
            }
            true
        });
        result?;

        let mut schemas = Vec::with_capacity(inputs.len());
        for input in inputs.iter() {
            let schema = lp_arena.get(*input).schema(lp_arena).into_owned();
            schemas.push(schema);
        }
        let new_schema = merge_schemas(&schemas)?;
        Arc::new(new_schema)
    };

    // If only one input remains after projection pushdown eliminated the others,
    // collapse the HConcat into that single input. This allows subsequent predicate
    // pushdown to push filters directly into the input (e.g. into a parquet scan).
    if inputs.len() == 1 {
        return Ok(lp_arena.take(inputs[0]));
    }

    Ok(IR::HConcat {
        inputs,
        schema,
        options,
    })
}
