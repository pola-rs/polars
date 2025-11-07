use std::sync::Arc;

use polars_core::prelude::InitHashMaps;
use polars_error::PolarsResult;
use polars_ops::frame::{JoinCoalesce, JoinType};
use polars_utils::arena::Arena;
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

use crate::plans::{
    AExpr, ColumnNode, ExprIR, ExprOrigin, IR, IRBuilder, OutputName, PlHashSet, det_join_schema,
};
use crate::prelude::optimizer::projection_pushdown::ProjectionContext;
use crate::prelude::{ProjectionOptions, ProjectionPushDown};
use crate::utils::{aexpr_to_leaf_names_iter, column_node_to_name};

/// # Panics
/// Panics if `join_ir` is not `IR::Join`.
pub(super) fn process_join(
    mut join_ir: IR,
    proj_cx: ProjectionContext,
    proj_pd: &mut ProjectionPushDown,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<IR> {
    let IR::Join {
        input_left,
        input_right,
        schema: join_output_schema,
        left_on,
        right_on,
        options,
    } = &mut join_ir
    else {
        panic!()
    };

    let is_projected =
        |name: &str| proj_cx.projected_names.contains(name) || !proj_cx.has_pushed_down();

    let input_schema_left = ir_arena.get(*input_left).schema(ir_arena).into_owned();
    let input_schema_right = ir_arena.get(*input_right).schema(ir_arena).into_owned();

    let mut project_left = PlHashSet::with_capacity(input_schema_left.len());
    let mut project_right = PlHashSet::with_capacity(input_schema_right.len());

    let mut coalesced_to_right: PlHashSet<PlSmallStr> = Default::default();
    if options.args.should_coalesce()
        && let JoinType::Right = &options.args.how
    {
        coalesced_to_right = left_on
            .iter()
            .map(|expr| {
                let node = match expr_arena.get(expr.node()) {
                    AExpr::Cast {
                        expr,
                        dtype: _,
                        options: _,
                    } => *expr,

                    _ => expr.node(),
                };

                let AExpr::Column(name) = expr_arena.get(node) else {
                    // All keys should be columns when coalesce=True
                    unreachable!()
                };

                name.clone()
            })
            .collect()
    }

    // Add accumulated projections
    for output_name in join_output_schema.iter_names() {
        if !is_projected(output_name) {
            continue;
        }

        match ExprOrigin::get_column_origin(
            output_name,
            &input_schema_left,
            &input_schema_right,
            options.args.suffix(),
            Some(&|name| coalesced_to_right.contains(name)),
        )? {
            ExprOrigin::None => {},
            ExprOrigin::Left => {
                project_left.insert(output_name.clone());
            },
            ExprOrigin::Right => {
                let name = if !input_schema_right.contains(output_name.as_str()) {
                    PlSmallStr::from_str(
                        output_name
                            .strip_suffix(options.args.suffix().as_str())
                            .unwrap(),
                    )
                } else {
                    output_name.clone()
                };

                debug_assert!(input_schema_right.contains(name.as_str()));

                project_right.insert(name);
            },
            ExprOrigin::Both => unreachable!(),
        }
    }

    // Add projections required by the join itself
    for expr_ir in left_on.as_slice() {
        for name in aexpr_to_leaf_names_iter(expr_ir.node(), expr_arena) {
            project_left.insert(name);
        }
    }

    for expr_ir in right_on.as_slice() {
        for name in aexpr_to_leaf_names_iter(expr_ir.node(), expr_arena) {
            project_right.insert(name);
        }
    }

    #[cfg(feature = "asof_join")]
    if let JoinType::AsOf(asof_options) = &options.args.how {
        if let Some(left_by) = asof_options.left_by.as_deref() {
            for name in left_by {
                project_left.insert(name.clone());
            }
        }

        if let Some(right_by) = asof_options.right_by.as_deref() {
            for name in right_by {
                project_right.insert(name.clone());
            }
        }
    }

    // Turn on coalesce if non-coalesced keys are not included in projection. Reduces materialization.
    if !options.args.should_coalesce()
        && matches!(options.args.how, JoinType::Inner | JoinType::Left)
        && left_on
            .iter()
            .all(|e| matches!(expr_arena.get(e.node()), AExpr::Column(_)))
        && right_on.iter().all(|e| {
            let AExpr::Column(name) = expr_arena.get(e.node()) else {
                return false;
            };

            let projected = if input_schema_left.contains(name.as_str()) {
                let name = format_pl_smallstr!("{}{}", name, options.args.suffix());
                is_projected(&name)
            } else {
                is_projected(name)
            };

            !projected
        })
    {
        Arc::make_mut(options).args.coalesce = JoinCoalesce::CoalesceColumns;
    }

    // Pushdown left/right projections.
    {
        let input = *input_left;
        let acc_projections = input_schema_left
            .iter_names()
            .filter(|x| project_left.contains(*x))
            .map(|name| ColumnNode(expr_arena.add(AExpr::Column(name.clone()))))
            .collect();
        let projected_names = project_left;

        proj_pd.pushdown_and_assign(
            input,
            ProjectionContext::new(acc_projections, projected_names, proj_cx.inner),
            ir_arena,
            expr_arena,
        )?;
    }

    {
        let input = *input_right;
        let acc_projections = input_schema_right
            .iter_names()
            .filter(|x| project_right.contains(*x))
            .map(|name| ColumnNode(expr_arena.add(AExpr::Column(name.clone()))))
            .collect();
        let projected_names = project_right;

        proj_pd.pushdown_and_assign(
            input,
            ProjectionContext::new(acc_projections, projected_names, proj_cx.inner),
            ir_arena,
            expr_arena,
        )?;
    }

    // Resolve new schemas after pushdown to left/right.
    let input_schema_left = ir_arena.get(*input_left).schema(ir_arena).into_owned();
    let input_schema_right = ir_arena.get(*input_right).schema(ir_arena).into_owned();
    let new_join_output_schema = det_join_schema(
        &input_schema_left,
        &input_schema_right,
        left_on,
        right_on,
        options,
        expr_arena,
    )
    .unwrap();

    let post_project: Option<Vec<ExprIR>> = if proj_cx.has_pushed_down() {
        let mut needs_post_project = proj_cx.acc_projections.len() != new_join_output_schema.len();

        // Build post-projection to re-order the columns and add suffixes if necessary.
        let post_project: Vec<ExprIR> = proj_cx
            .acc_projections
            .iter()
            .enumerate()
            .map(|(i, col_node)| {
                let original_projected_name = column_node_to_name(*col_node, expr_arena);

                if new_join_output_schema.index_of(original_projected_name.as_str()) != Some(i) {
                    needs_post_project = true;
                }

                if !new_join_output_schema.contains(original_projected_name.as_str()) {
                    // This name is no longer suffixed in the new output schema, we restore it with an
                    // alias here.
                    let new_output_name = PlSmallStr::from_str(
                        original_projected_name
                            .strip_suffix(options.args.suffix().as_str())
                            .unwrap(),
                    );

                    debug_assert!(new_join_output_schema.contains(new_output_name.as_str()));
                    let original_projected_name = original_projected_name.clone();

                    ExprIR::new(
                        expr_arena.add(AExpr::Column(new_output_name)),
                        OutputName::Alias(original_projected_name),
                    )
                } else {
                    ExprIR::from_node(col_node.0, expr_arena)
                }
            })
            .collect();

        needs_post_project.then_some(post_project)
    } else {
        None
    };

    *join_output_schema = new_join_output_schema;

    let out: IR = if let Some(post_project) = post_project {
        IRBuilder::from_lp(join_ir, expr_arena, ir_arena)
            .project(
                post_project,
                ProjectionOptions {
                    run_parallel: false,
                    duplicate_check: false,
                    should_broadcast: false,
                },
            )
            .build()
    } else {
        join_ir
    };

    Ok(out)
}
