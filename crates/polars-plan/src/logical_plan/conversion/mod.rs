mod dsl_to_ir;
mod ir_to_dsl;

use std::borrow::Cow;

use polars_core::prelude::*;
use polars_utils::vec::ConvertVec;
use recursive::recursive;

use crate::constants::get_len_name;
use crate::prelude::*;
pub use dsl_to_ir::*;
pub use ir_to_dsl::*;



fn expr_irs_to_exprs(expr_irs: Vec<ExprIR>, expr_arena: &Arena<AExpr>) -> Vec<Expr> {
    expr_irs.convert_owned(|e| e.to_expr(expr_arena))
}

impl IR {
    #[recursive]
    fn into_lp<F, LPA>(
        self,
        conversion_fn: &F,
        lp_arena: &mut LPA,
        expr_arena: &Arena<AExpr>,
    ) -> LogicalPlan
        where
            F: Fn(Node, &mut LPA) -> IR,
    {
        let lp = self;
        let convert_to_lp = |node: Node, lp_arena: &mut LPA| {
            conversion_fn(node, lp_arena).into_lp(conversion_fn, lp_arena, expr_arena)
        };
        match lp {
            IR::Scan {
                paths,
                file_info,
                predicate,
                scan_type,
                output_schema: _,
                file_options: options,
            } => LogicalPlan::Scan {
                paths,
                file_info,
                predicate: predicate.map(|e| e.to_expr(expr_arena)),
                scan_type,
                file_options: options,
            },
            #[cfg(feature = "python")]
            IR::PythonScan { options, .. } => LogicalPlan::PythonScan { options },
            IR::Union { inputs, options } => {
                let inputs = inputs
                    .into_iter()
                    .map(|node| convert_to_lp(node, lp_arena))
                    .collect();
                LogicalPlan::Union { inputs, options }
            },
            IR::HConcat {
                inputs,
                schema,
                options,
            } => {
                let inputs = inputs
                    .into_iter()
                    .map(|node| convert_to_lp(node, lp_arena))
                    .collect();
                LogicalPlan::HConcat {
                    inputs,
                    schema: schema.clone(),
                    options,
                }
            },
            IR::Slice { input, offset, len } => {
                let lp = convert_to_lp(input, lp_arena);
                LogicalPlan::Slice {
                    input: Arc::new(lp),
                    offset,
                    len,
                }
            },
            IR::Filter { input, predicate } => {
                let lp = convert_to_lp(input, lp_arena);
                let predicate = predicate.to_expr(expr_arena);
                LogicalPlan::Filter {
                    input: Arc::new(lp),
                    predicate,
                }
            },
            IR::DataFrameScan {
                df,
                schema,
                output_schema,
                projection,
                selection,
            } => LogicalPlan::DataFrameScan {
                df,
                schema,
                output_schema,
                projection,
                selection: selection.map(|e| e.to_expr(expr_arena)),
            },
            IR::Select {
                expr,
                input,
                schema,
                options,
            } => {
                let i = convert_to_lp(input, lp_arena);
                let expr = expr_irs_to_exprs(expr.all_exprs(), expr_arena);
                LogicalPlan::Select {
                    expr,
                    input: Arc::new(i),
                    schema,
                    options,
                }
            },
            IR::SimpleProjection { input, columns, .. } => {
                let input = convert_to_lp(input, lp_arena);
                let expr = columns
                    .iter_names()
                    .map(|name| Expr::Column(ColumnName::from(name.as_str())))
                    .collect::<Vec<_>>();
                LogicalPlan::Select {
                    expr,
                    input: Arc::new(input),
                    schema: columns.clone(),
                    options: Default::default(),
                }
            },
            IR::Sort {
                input,
                by_column,
                slice,
                sort_options,
            } => {
                let input = Arc::new(convert_to_lp(input, lp_arena));
                let by_column = expr_irs_to_exprs(by_column, expr_arena);
                LogicalPlan::Sort {
                    input,
                    by_column,
                    slice,
                    sort_options,
                }
            },
            IR::Cache {
                input,
                id,
                cache_hits,
            } => {
                let input = Arc::new(convert_to_lp(input, lp_arena));
                LogicalPlan::Cache {
                    input,
                    id,
                    cache_hits,
                }
            },
            IR::GroupBy {
                input,
                keys,
                aggs,
                schema,
                apply,
                maintain_order,
                options: dynamic_options,
            } => {
                let i = convert_to_lp(input, lp_arena);
                let keys = Arc::new(expr_irs_to_exprs(keys, expr_arena));
                let aggs = expr_irs_to_exprs(aggs, expr_arena);

                LogicalPlan::GroupBy {
                    input: Arc::new(i),
                    keys,
                    aggs,
                    schema,
                    apply,
                    maintain_order,
                    options: dynamic_options,
                }
            },
            IR::Join {
                input_left,
                input_right,
                schema,
                left_on,
                right_on,
                options,
            } => {
                let i_l = convert_to_lp(input_left, lp_arena);
                let i_r = convert_to_lp(input_right, lp_arena);

                let left_on = expr_irs_to_exprs(left_on, expr_arena);
                let right_on = expr_irs_to_exprs(right_on, expr_arena);

                LogicalPlan::Join {
                    input_left: Arc::new(i_l),
                    input_right: Arc::new(i_r),
                    schema,
                    left_on,
                    right_on,
                    options,
                }
            },
            IR::HStack {
                input,
                exprs,
                schema,
                options,
            } => {
                let i = convert_to_lp(input, lp_arena);
                let exprs = expr_irs_to_exprs(exprs.all_exprs(), expr_arena);

                LogicalPlan::HStack {
                    input: Arc::new(i),
                    exprs,
                    schema,
                    options,
                }
            },
            IR::Distinct { input, options } => {
                let i = convert_to_lp(input, lp_arena);
                LogicalPlan::Distinct {
                    input: Arc::new(i),
                    options,
                }
            },
            IR::MapFunction { input, function } => {
                let input = Arc::new(convert_to_lp(input, lp_arena));
                LogicalPlan::MapFunction { input, function }
            },
            IR::ExtContext {
                input,
                contexts,
                schema,
            } => {
                let input = Arc::new(convert_to_lp(input, lp_arena));
                let contexts = contexts
                    .into_iter()
                    .map(|node| convert_to_lp(node, lp_arena))
                    .collect();
                LogicalPlan::ExtContext {
                    input,
                    contexts,
                    schema,
                }
            },
            IR::Sink { input, payload } => {
                let input = Arc::new(convert_to_lp(input, lp_arena));
                LogicalPlan::Sink { input, payload }
            },
            IR::Invalid => unreachable!(),
        }
    }
}
