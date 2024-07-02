mod dsl;

use polars_core::error::{polars_bail, PolarsResult};

use crate::dsl::Expr;
use crate::plans::{DslFunction, DslPlan, FileScan, FunctionNode};

fn is_cloud_eligible(dsl: &DslPlan) -> PolarsResult<()> {
    let mut expr_stack = vec![];
    for plan_node in dsl.into_iter() {
        match plan_node {
            DslPlan::MapFunction {
                function: DslFunction::FunctionNode(function),
                ..
            } => match function {
                FunctionNode::Opaque { .. } => {
                    polars_bail!(InvalidOperation: "opaque function not eligible for cloud")
                },
                #[cfg(feature = "python")]
                FunctionNode::OpaquePython { .. } => {
                    polars_bail!(InvalidOperation: "python function not eligible for cloud")
                },
                _ => {},
            },
            DslPlan::Scan {
                scan_type: FileScan::Anonymous { .. },
                ..
            } => {
                polars_bail!(InvalidOperation: "anonymous scan not eligible for cloud")
            },
            plan => {
                plan.get_expr(&mut expr_stack);

                for expr in expr_stack.drain(..) {
                    for expr_node in expr.into_iter() {
                        match expr_node {
                            Expr::AnonymousFunction { .. } => {
                                todo!()
                            },
                            _ => {},
                        }
                    }
                }
            },
        }
    }
    Ok(())
}
