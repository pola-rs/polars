mod dsl;

use polars_core::error::{polars_err, PolarsResult};
use polars_io::path_utils::is_cloud_url;

use crate::dsl::Expr;
use crate::plans::options::SinkType;
use crate::plans::{DslFunction, DslPlan, FunctionNode};

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
                _ => (),
            },
            #[cfg(feature = "python")]
            DslPlan::PythonScan { .. } => return ineligible_error("contains Python scan"),
            DslPlan::GroupBy { apply: Some(_), .. } => {
                return ineligible_error("contains Python function in group by operation")
            },
            DslPlan::Scan { paths, .. }
                if paths.lock().unwrap().0.iter().any(|p| !is_cloud_url(p)) =>
            {
                return ineligible_error("contains scan of local file system")
            },
            DslPlan::Sink { payload, .. } => {
                if !matches!(payload, SinkType::Cloud { .. }) {
                    return ineligible_error("contains sink to non-cloud location");
                }
            },
            plan => {
                plan.get_expr(&mut expr_stack);

                for expr in expr_stack.drain(..) {
                    for expr_node in expr.into_iter() {
                        match expr_node {
                            Expr::AnonymousFunction { .. } => {
                                return ineligible_error("contains anonymous function")
                            },
                            Expr::RenameAlias { .. } => {
                                return ineligible_error("contains custom name remapping")
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
