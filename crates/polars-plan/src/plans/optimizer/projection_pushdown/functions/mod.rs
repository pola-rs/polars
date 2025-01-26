#[cfg(feature = "pivot")]
mod unpivot;

#[cfg(feature = "pivot")]
use unpivot::process_unpivot;

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_functions(
    proj_pd: &mut ProjectionPushDown,
    input: Node,
    function: FunctionIR,
    mut ctx: ProjectionContext,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<IR> {
    use FunctionIR::*;
    match function {
        Rename {
            ref existing,
            ref new,
            swapping,
            schema: _,
        } => {
            let clear = ctx.has_pushed_down();
            process_rename(
                &mut ctx.acc_projections,
                &mut ctx.projected_names,
                expr_arena,
                existing,
                new,
                swapping,
            )?;
            proj_pd.pushdown_and_assign(input, ctx, lp_arena, expr_arena)?;

            if clear {
                function.clear_cached_schema()
            }

            let lp = IR::MapFunction { input, function };
            Ok(lp)
        },
        Explode { columns, .. } => {
            columns.iter().for_each(|name| {
                add_str_to_accumulated(
                    name.clone(),
                    &mut ctx.acc_projections,
                    &mut ctx.projected_names,
                    expr_arena,
                )
            });
            proj_pd.pushdown_and_assign(input, ctx, lp_arena, expr_arena)?;
            Ok(IRBuilder::new(input, expr_arena, lp_arena)
                .explode(columns.clone())
                .build())
        },
        #[cfg(feature = "pivot")]
        Unpivot { ref args, .. } => {
            let lp = IR::MapFunction {
                input,
                function: function.clone(),
            };

            process_unpivot(proj_pd, lp, args, input, ctx, lp_arena, expr_arena)
        },
        _ => {
            if function.allow_projection_pd() && ctx.has_pushed_down() {
                let original_acc_projection_len = ctx.acc_projections.len();

                // add columns needed for the function.
                for name in function.additional_projection_pd_columns().as_ref() {
                    let node = expr_arena.add(AExpr::Column(name.clone()));
                    add_expr_to_accumulated(
                        node,
                        &mut ctx.acc_projections,
                        &mut ctx.projected_names,
                        expr_arena,
                    )
                }
                let expands_schema = matches!(function, FunctionIR::Unnest { .. });

                let local_projections = proj_pd.pushdown_and_assign_check_schema(
                    input,
                    ctx,
                    lp_arena,
                    expr_arena,
                    expands_schema,
                )?;

                // Remove the cached schema
                function.clear_cached_schema();
                let lp = IR::MapFunction {
                    input,
                    function: function.clone(),
                };

                if local_projections.is_empty() {
                    Ok(lp)
                } else {
                    // if we would project, we would remove pushed down predicates
                    if local_projections.len() < original_acc_projection_len {
                        Ok(IRBuilder::from_lp(lp, expr_arena, lp_arena)
                            .with_columns_simple(local_projections, Default::default())
                            .build())
                        // all projections are local
                    } else {
                        Ok(IRBuilder::from_lp(lp, expr_arena, lp_arena)
                            .project_simple_nodes(local_projections)
                            .unwrap()
                            .build())
                    }
                }
            } else {
                let lp = IR::MapFunction {
                    input,
                    function: function.clone(),
                };
                // restart projection pushdown
                proj_pd.no_pushdown_restart_opt(lp, ctx, lp_arena, expr_arena)
            }
        },
    }
}
