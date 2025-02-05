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
use std::sync::{Arc, Mutex};

pub use dsl_to_ir::*;
pub use expr_to_ir::*;
pub use ir_to_dsl::*;
use polars_core::prelude::*;
use polars_utils::idx_vec::UnitVec;
use polars_utils::unitvec;
use polars_utils::vec::ConvertVec;
use recursive::recursive;
mod functions;
mod join;
pub(crate) mod type_check;
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
            ir @ IR::Scan { .. } => {
                let IR::Scan {
                    sources,
                    file_info,
                    hive_parts: _,
                    predicate: _,
                    scan_type,
                    output_schema: _,
                    file_options,
                } = ir.clone()
                else {
                    unreachable!()
                };

                DslPlan::Scan {
                    sources: sources.clone(),
                    file_info: Some(file_info.clone()),
                    scan_type: scan_type.clone(),
                    file_options: file_options.clone(),
                    cached_ir: Arc::new(Mutex::new(Some(ir))),
                }
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
                output_schema: _,
            } => DslPlan::DataFrameScan { df, schema },
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
            IR::SimpleProjection { input, columns } => {
                let input = convert_to_lp(input, lp_arena);
                let expr = columns
                    .iter_names()
                    .map(|name| Expr::Column(name.clone()))
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
                cache_hits: _,
            } => {
                let input = Arc::new(convert_to_lp(input, lp_arena));
                DslPlan::Cache { input, id }
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
                    predicates: Default::default(),
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
            #[cfg(feature = "merge_sorted")]
            IR::MergeSorted {
                input_left,
                input_right,
                key,
            } => {
                let input_left = Arc::new(convert_to_lp(input_left, lp_arena));
                let input_right = Arc::new(convert_to_lp(input_right, lp_arena));

                DslPlan::MergeSorted {
                    input_left,
                    input_right,
                    key,
                }
            },
            IR::Invalid => unreachable!(),
        }
    }
}

fn get_input(lp_arena: &Arena<IR>, lp_node: Node) -> UnitVec<Node> {
    let plan = lp_arena.get(lp_node);
    let mut inputs: UnitVec<Node> = unitvec!();

    // Used to get the schema of the input.
    if is_scan(plan) {
        inputs.push(lp_node);
    } else {
        plan.copy_inputs(&mut inputs);
    };
    inputs
}

fn get_schema(lp_arena: &Arena<IR>, lp_node: Node) -> Cow<'_, SchemaRef> {
    let inputs = get_input(lp_arena, lp_node);
    if inputs.is_empty() {
        // Files don't have an input, so we must take their schema.
        Cow::Borrowed(lp_arena.get(lp_node).scan_schema())
    } else {
        let input = inputs[0];
        lp_arena.get(input).schema(lp_arena)
    }
}
