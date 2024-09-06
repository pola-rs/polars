use arrow::legacy::error::PolarsResult;
use polars_utils::arena::Arena;
use crate::dsl::{Expr, FunctionExpr};
use crate::plans::AExpr;
use crate::prelude::FunctionOptions;
use super::*;

pub fn resolve_join(
    input_left: Arc<DslPlan>,
    input_right: Arc<DslPlan>,
    left_on: Vec<Expr>,
    right_on: Vec<Expr>,
    predicates: Vec<Expr>,
    mut options: Arc<JoinOptions>,
    ctxt: &mut DslConversionContext
) -> PolarsResult<IR> {
    let owned = Arc::unwrap_or_clone;
    if matches!(options.args.how, JoinType::Cross) {
        polars_ensure!(left_on.len() + right_on.len() == 0, InvalidOperation: "a 'cross' join doesn't expect any join keys");
    } else {
        let mut turn_off_coalesce = false;
        for e in left_on.iter().chain(right_on.iter()) {
            if has_expr(e, |e| matches!(e, Expr::Alias(_, _))) {
                polars_bail!(
                            ComputeError:
                            "'alias' is not allowed in a join key, use 'with_columns' first",
                        )
            }
            // Any expression that is not a simple column expression will turn of coalescing.
            turn_off_coalesce |= has_expr(e, |e| !matches!(e, Expr::Column(_)));
        }
        if turn_off_coalesce {
            let options = Arc::make_mut(&mut options);
            if matches!(options.args.coalesce, JoinCoalesce::CoalesceColumns) {
                polars_warn!("coalescing join requested but not all join keys are column references, turning off key coalescing");
            }
            options.args.coalesce = JoinCoalesce::KeepColumns;
        }

        options.args.validation.is_valid_join(&options.args.how)?;

        polars_ensure!(
                    left_on.len() == right_on.len(),
                    ComputeError:
                        format!(
                            "the number of columns given as join key (left: {}, right:{}) should be equal",
                            left_on.len(),
                            right_on.len()
                        )
                );
    }

    let input_left = to_alp_impl(owned(input_left), ctxt)
        .map_err(|e| e.context(failed_input!(join left)))?;
    let input_right = to_alp_impl(owned(input_right), ctxt)
        .map_err(|e| e.context(failed_input!(join, right)))?;

    let schema_left = ctxt.lp_arena.get(input_left).schema(ctxt.lp_arena);
    let schema_right = ctxt.lp_arena.get(input_right).schema(ctxt.lp_arena);

    let schema =
        det_join_schema(&schema_left, &schema_right, &left_on, &right_on, &options)
            .map_err(|e| e.context(failed_here!(join schema resolving)))?;

    let left_on = to_expr_irs_ignore_alias(left_on, ctxt.expr_arena)?;
    let right_on = to_expr_irs_ignore_alias(right_on, ctxt.expr_arena)?;
    let mut joined_on = PlHashSet::new();
    for (l, r) in left_on.iter().zip(right_on.iter()) {
        polars_ensure!(
                    joined_on.insert((l.output_name(), r.output_name())),
                    InvalidOperation: "joining with repeated key names; already joined on {} and {}",
                    l.output_name(),
                    r.output_name()
                )
    }
    drop(joined_on);

    ctxt.conversion_optimizer
        .fill_scratch(&left_on, ctxt.expr_arena);
    ctxt.conversion_optimizer
        .fill_scratch(&right_on, ctxt.expr_arena);

    // Every expression must be elementwise so that we are
    // guaranteed the keys for a join are all the same length.
    let all_elementwise =
        |aexprs: &[ExprIR]| all_streamable(aexprs, &*ctxt.expr_arena, Context::Default);
    polars_ensure!(
                all_elementwise(&left_on) && all_elementwise(&right_on),
                InvalidOperation: "All join key expressions must be elementwise."
            );
    let lp = IR::Join {
        input_left,
        input_right,
        schema,
        left_on,
        right_on,
        options,
    };
    Ok(lp)

}