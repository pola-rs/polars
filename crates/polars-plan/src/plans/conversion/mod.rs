mod convert_utils;
mod dsl_to_ir;
mod expr_expansion;
mod expr_to_ir;
mod ir_to_dsl;
#[cfg(any(
    feature = "ipc",
    feature = "parquet",
    feature = "csv",
    feature = "json"
))]
mod scans;
mod stack_opt;

use std::borrow::Cow;
use std::sync::{Arc, Mutex, RwLock};

pub use dsl_to_ir::*;
pub use expr_to_ir::*;
pub use ir_to_dsl::*;
use polars_core::prelude::*;
use polars_utils::vec::ConvertVec;
use recursive::recursive;
pub(crate) mod type_coercion;

pub(crate) use expr_expansion::{expand_selectors, is_regex_projection, prepare_projection};

use crate::constants::get_len_name;
use crate::prelude::*;

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
    ) -> DslPlan
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
                hive_parts,
                predicate,
                scan_type,
                output_schema: _,
                file_options: options,
            } => DslPlan::Scan {
                paths: Arc::new(Mutex::new((paths, true))),
                file_info: Arc::new(RwLock::new(Some(file_info))),
                hive_parts,
                predicate: predicate.map(|e| e.to_expr(expr_arena)),
                scan_type,
                file_options: options,
            },
            #[cfg(feature = "python")]
            IR::PythonScan { options, .. } => DslPlan::PythonScan { options },
            IR::Union { inputs, .. } => {
                let inputs = inputs
                    .into_iter()
                    .map(|node| convert_to_lp(node, lp_arena))
                    .collect();
                DslPlan::Union {
                    inputs,
                    args: Default::default(),
                }
            },
            IR::HConcat {
                inputs,
                schema: _,
                options,
            } => {
                let inputs = inputs
                    .into_iter()
                    .map(|node| convert_to_lp(node, lp_arena))
                    .collect();
                DslPlan::HConcat { inputs, options }
            },
            IR::Slice { input, offset, len } => {
                let lp = convert_to_lp(input, lp_arena);
                DslPlan::Slice {
                    input: Arc::new(lp),
                    offset,
                    len,
                }
            },
            IR::Filter { input, predicate } => {
                let lp = convert_to_lp(input, lp_arena);
                let predicate = predicate.to_expr(expr_arena);
                DslPlan::Filter {
                    input: Arc::new(lp),
                    predicate,
                }
            },
            IR::DataFrameScan {
                df,
                schema,
                output_schema,
                filter: selection,
            } => DslPlan::DataFrameScan {
                df,
                schema,
                output_schema,
                filter: selection.map(|e| e.to_expr(expr_arena)),
            },
            IR::Select {
                expr,
                input,
                schema: _,
                options,
            } => {
                let i = convert_to_lp(input, lp_arena);
                let expr = expr_irs_to_exprs(expr, expr_arena);
                DslPlan::Select {
                    expr,
                    input: Arc::new(i),
                    options,
                }
            },
            IR::Reduce { exprs, input, .. } => {
                let i = convert_to_lp(input, lp_arena);
                let expr = expr_irs_to_exprs(exprs, expr_arena);
                DslPlan::Select {
                    expr,
                    input: Arc::new(i),
                    options: Default::default(),
                }
            },
            IR::SimpleProjection { input, columns } => {
                let input = convert_to_lp(input, lp_arena);
                let expr = columns
                    .iter_names()
                    .map(|name| Expr::Column(ColumnName::from(name.as_str())))
                    .collect::<Vec<_>>();
                DslPlan::Select {
                    expr,
                    input: Arc::new(input),
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
                DslPlan::Sort {
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
                DslPlan::Cache {
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
                let keys = expr_irs_to_exprs(keys, expr_arena);
                let aggs = expr_irs_to_exprs(aggs, expr_arena);

                DslPlan::GroupBy {
                    input: Arc::new(i),
                    keys,
                    aggs,
                    apply: apply.map(|apply| (apply, schema)),
                    maintain_order,
                    options: dynamic_options,
                }
            },
            IR::Join {
                input_left,
                input_right,
                schema: _,
                left_on,
                right_on,
                options,
            } => {
                let i_l = convert_to_lp(input_left, lp_arena);
                let i_r = convert_to_lp(input_right, lp_arena);

                let left_on = expr_irs_to_exprs(left_on, expr_arena);
                let right_on = expr_irs_to_exprs(right_on, expr_arena);

                DslPlan::Join {
                    input_left: Arc::new(i_l),
                    input_right: Arc::new(i_r),
                    left_on,
                    right_on,
                    options,
                }
            },
            IR::HStack {
                input,
                exprs,
                options,
                ..
            } => {
                let i = convert_to_lp(input, lp_arena);
                let exprs = expr_irs_to_exprs(exprs, expr_arena);

                DslPlan::HStack {
                    input: Arc::new(i),
                    exprs,
                    options,
                }
            },
            IR::Distinct { input, options } => {
                let i = convert_to_lp(input, lp_arena);
                let options = DistinctOptionsDSL {
                    subset: options.subset.map(|s| {
                        s.iter()
                            .map(|name| Expr::Column(name.clone()).into())
                            .collect()
                    }),
                    maintain_order: options.maintain_order,
                    keep_strategy: options.keep_strategy,
                };
                DslPlan::Distinct {
                    input: Arc::new(i),
                    options,
                }
            },
            IR::MapFunction { input, function } => {
                let input = Arc::new(convert_to_lp(input, lp_arena));
                DslPlan::MapFunction {
                    input,
                    function: function.into(),
                }
            },
            IR::ExtContext {
                input, contexts, ..
            } => {
                let input = Arc::new(convert_to_lp(input, lp_arena));
                let contexts = contexts
                    .into_iter()
                    .map(|node| convert_to_lp(node, lp_arena))
                    .collect();
                DslPlan::ExtContext { input, contexts }
            },
            IR::Sink { input, payload } => {
                let input = Arc::new(convert_to_lp(input, lp_arena));
                DslPlan::Sink { input, payload }
            },
            IR::Invalid => unreachable!(),
        }
    }
}
