mod dsl;

use polars_core::error::{polars_err, PolarsResult};

use crate::dsl::Expr;
use crate::plans::{DslFunction, DslPlan, FileScan, FunctionNode};

/// Assert that the given [`DslPlan`] is eligible to be executed on Polars Cloud.
pub fn assert_cloud_eligible(dsl: &DslPlan) -> PolarsResult<()> {
    let mut expr_stack = vec![];
    for plan_node in dsl.into_iter() {
        match plan_node {
            DslPlan::MapFunction {
                function: DslFunction::FunctionNode(function),
                ..
            } => match function {
                FunctionNode::Opaque { .. } => return ineligible_error("contains opaque function"),
                #[cfg(feature = "python")]
                FunctionNode::OpaquePython { .. } => {
                    return ineligible_error("contains Python function")
                },
                _ => {},
            },
            DslPlan::Scan {
                scan_type: FileScan::Anonymous { .. },
                ..
            } => return ineligible_error("contains anonymous scan"),
            plan => {
                plan.get_expr(&mut expr_stack);

                for expr in expr_stack.drain(..) {
                    for expr_node in expr.into_iter() {
                        match expr_node {
                            Expr::AnonymousFunction { .. } => {
                                return ineligible_error("contains anonymous function")
                            },
                            _ => (),
                        }
                    }
                }
            },
        }
    }
    Ok(())
}

fn ineligible_error(message: &str) -> PolarsResult<()> {
    Err(polars_err!(
        InvalidOperation:
        "logical plan ineligible for execution on Polars Cloud: {message}"
    ))
}
