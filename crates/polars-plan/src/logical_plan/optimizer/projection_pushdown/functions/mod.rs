mod melt;

use melt::process_melt;

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn process_functions(
    proj_pd: &mut ProjectionPushDown,
    input: Node,
    function: &FunctionNode,
    mut acc_projections: Vec<Node>,
    mut projected_names: PlHashSet<Arc<str>>,
    projections_seen: usize,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<ALogicalPlan> {
    use FunctionNode::*;
    match function {
        Rename {
            existing,
            new,
            swapping,
        } => {
            process_rename(
                &mut acc_projections,
                &mut projected_names,
                expr_arena,
                existing,
                new,
                *swapping,
            )?;
            proj_pd.pushdown_and_assign(
                input,
                acc_projections,
                projected_names,
                projections_seen,
                lp_arena,
                expr_arena,
            )?;

            let lp = ALogicalPlan::MapFunction {
                input,
                function: function.clone(),
            };
            Ok(lp)
        },
        Explode { columns, .. } => {
            columns.iter().for_each(|name| {
                add_str_to_accumulated(name, &mut acc_projections, &mut projected_names, expr_arena)
            });
            proj_pd.pushdown_and_assign(
                input,
                acc_projections,
                projected_names,
                projections_seen,
                lp_arena,
                expr_arena,
            )?;
            Ok(ALogicalPlanBuilder::new(input, expr_arena, lp_arena)
                .explode(columns.clone())
                .build())
        },
        Melt { args, .. } => {
            let lp = ALogicalPlan::MapFunction {
                input,
                function: function.clone(),
            };

            process_melt(
                proj_pd,
                lp,
                args,
                input,
                acc_projections,
                projections_seen,
                lp_arena,
                expr_arena,
            )
        },
        _ => {
            let lp = ALogicalPlan::MapFunction {
                input,
                function: function.clone(),
            };
            if function.allow_projection_pd() && !acc_projections.is_empty() {
                let original_acc_projection_len = acc_projections.len();

                // add columns needed for the function.
                for name in function.additional_projection_pd_columns().as_ref() {
                    let node = expr_arena.add(AExpr::Column(name.clone()));
                    add_expr_to_accumulated(
                        node,
                        &mut acc_projections,
                        &mut projected_names,
                        expr_arena,
                    )
                }
                let expands_schema = matches!(function, FunctionNode::Unnest { .. });

                let local_projections = proj_pd.pushdown_and_assign_check_schema(
                    input,
                    acc_projections,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                    expands_schema,
                )?;
                if local_projections.is_empty() {
                    Ok(lp)
                } else {
                    // if we would project, we would remove pushed down predicates
                    if local_projections.len() < original_acc_projection_len {
                        Ok(ALogicalPlanBuilder::from_lp(lp, expr_arena, lp_arena)
                            .with_columns(local_projections, Default::default())
                            .build())
                        // all projections are local
                    } else {
                        Ok(ALogicalPlanBuilder::from_lp(lp, expr_arena, lp_arena)
                            .project(local_projections, Default::default())
                            .build())
                    }
                }
            } else {
                // restart projection pushdown
                proj_pd.no_pushdown_restart_opt(
                    lp,
                    acc_projections,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )
            }
        },
    }
}
