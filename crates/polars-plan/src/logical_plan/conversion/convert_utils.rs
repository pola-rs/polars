use super::*;

pub(super) fn convert_st_union(
    inputs: &mut [Node],
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<()> {
    let mut schema = (**lp_arena.get(inputs[0]).schema(lp_arena)).clone();

    let mut changed = false;
    for input in inputs[1..].iter() {
        let schema_other = lp_arena.get(*input).schema(lp_arena);
        changed |= schema.to_supertype(schema_other.as_ref())?;
    }

    if changed {
        for input in inputs {
            let mut exprs = vec![];
            let input_schema = lp_arena.get(*input).schema(lp_arena);

            let to_cast = input_schema.iter().zip(schema.iter_dtypes()).flat_map(
                |((left_name, left_type), st)| {
                    if left_type != st {
                        Some(col(left_name.as_ref()).cast(st.clone()))
                    } else {
                        None
                    }
                },
            );
            exprs.extend(to_cast);

            if !exprs.is_empty() {
                let expr = to_expr_irs(exprs, expr_arena);
                let lp = IRBuilder::new(*input, expr_arena, lp_arena)
                    .with_columns(expr, Default::default())
                    .build();

                let node = lp_arena.add(lp);
                *input = node
            }
        }
    }
    Ok(())
}
