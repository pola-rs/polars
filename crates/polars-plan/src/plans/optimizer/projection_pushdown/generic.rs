use polars_utils::idx_vec::UnitVec;

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_generic(
    proj_pd: &mut ProjectionPushDown,
    lp: IR,
    ctx: ProjectionContext,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<IR> {
    let inputs = lp.get_inputs();
    let input_count = inputs.len();

    // let mut first_schema = None;
    // let mut names = None;

    let new_inputs = inputs
        .into_iter()
        .map(|node| {
            let alp = lp_arena.take(node);
            let mut alp = proj_pd.push_down(alp, ctx.clone(), lp_arena, expr_arena)?;

            // double projection can mess up the schema ordering
            // here we ensure the ordering is maintained.
            //
            // Consider this query
            // df1 => a, b
            // df2 => a, b
            //
            // df3 = df1.join(df2, on = a, b)
            //
            // concat([df1, df3]).select(a)
            //
            // schema after projection pd
            // df3 => a, b
            // df1 => a
            // so we ensure we do the 'a' projection again before we concatenate
            if !ctx.acc_projections.is_empty() && input_count > 1 {
                alp = IRBuilder::from_lp(alp, expr_arena, lp_arena)
                    .project_simple_nodes(ctx.acc_projections.iter().map(|e| e.0))
                    .unwrap()
                    .build();
            }
            lp_arena.replace(node, alp);
            Ok(node)
        })
        .collect::<PolarsResult<UnitVec<_>>>()?;

    Ok(lp.with_inputs(new_inputs))
}
