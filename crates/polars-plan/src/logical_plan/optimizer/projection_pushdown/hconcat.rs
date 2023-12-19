use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_hconcat(
    proj_pd: &mut ProjectionPushDown,
    inputs: Vec<Node>,
    schema: SchemaRef,
    options: HConcatOptions,
    acc_projections: Vec<Node>,
    projections_seen: usize,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<ALogicalPlan> {
    // When applying projection pushdown to horizontal concatenation,
    // we apply pushdown to all of the inputs using the subset of accumulated projections relevant to each input,
    // then rebuild the concatenated schema.

    let schema = if acc_projections.is_empty() {
        schema
    } else {
        let mut remaining_projections: PlHashSet<Node> = acc_projections.into_iter().collect();

        for input in inputs.iter() {
            let mut input_pushdown = Vec::new();

            for proj in remaining_projections.iter() {
                let input_schema = lp_arena.get(*input).schema(lp_arena);
                if check_input_node(*proj, input_schema.as_ref(), expr_arena) {
                    input_pushdown.push(*proj);
                }
            }

            let mut input_names = PlHashSet::new();
            for proj in &input_pushdown {
                remaining_projections.remove(proj);
                for name in aexpr_to_leaf_names(*proj, expr_arena) {
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

        let schema_size = inputs
            .iter()
            .map(|input| lp_arena.get(*input).schema(lp_arena).len())
            .sum();
        let mut new_schema = Schema::with_capacity(schema_size);
        for input in inputs.iter() {
            let schema = lp_arena.get(*input).schema(lp_arena);
            schema.as_ref().iter().for_each(|(name, dtype)| {
                new_schema.with_column(name.clone(), dtype.clone());
            });
        }

        Arc::new(new_schema)
    };

    Ok(ALogicalPlan::HConcat {
        inputs,
        schema,
        options,
    })
}
