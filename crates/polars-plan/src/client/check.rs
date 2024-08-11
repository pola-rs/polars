use polars_core::error::{polars_err, PolarsResult};
use polars_io::path_utils::is_cloud_url;

use crate::dsl::Expr;
use crate::plans::options::SinkType;
use crate::plans::{DslFunction, DslPlan, FileScan, FunctionIR};

/// Assert that the given [`DslPlan`] is eligible to be executed on Polars Cloud.
pub(super) fn assert_cloud_eligible(dsl: &DslPlan) -> PolarsResult<()> {
    let mut expr_stack = vec![];
    for plan_node in dsl.into_iter() {
        match plan_node {
            DslPlan::MapFunction { function, .. } => match function {
                DslFunction::FunctionIR(FunctionIR::Opaque { .. }) => {
                    return ineligible_error("contains opaque function")
                },
                #[cfg(feature = "python")]
                DslFunction::OpaquePython { .. } => {
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
            DslPlan::Scan {
                scan_type: FileScan::Anonymous { .. },
                ..
            } => return ineligible_error("contains anonymous scan"),
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

impl DslPlan {
    fn inputs<'a>(&'a self, scratch: &mut Vec<&'a DslPlan>) {
        use DslPlan::*;
        match self {
            Select { input, .. }
            | GroupBy { input, .. }
            | Filter { input, .. }
            | Distinct { input, .. }
            | Sort { input, .. }
            | Slice { input, .. }
            | HStack { input, .. }
            | MapFunction { input, .. }
            | Sink { input, .. }
            | Cache { input, .. } => scratch.push(input),
            Union { inputs, .. } | HConcat { inputs, .. } => scratch.extend(inputs),
            Join {
                input_left,
                input_right,
                ..
            } => {
                scratch.push(input_left);
                scratch.push(input_right);
            },
            ExtContext { input, contexts } => {
                scratch.push(input);
                scratch.extend(contexts);
            },
            IR { dsl, .. } => scratch.push(dsl),
            Scan { .. } | DataFrameScan { .. } => (),
            #[cfg(feature = "python")]
            PythonScan { .. } => (),
        }
    }

    fn get_expr<'a>(&'a self, scratch: &mut Vec<&'a Expr>) {
        use DslPlan::*;
        match self {
            Filter { predicate, .. } => scratch.push(predicate),
            Scan { predicate, .. } => {
                if let Some(expr) = predicate {
                    scratch.push(expr)
                }
            },
            DataFrameScan { filter, .. } => {
                if let Some(expr) = filter {
                    scratch.push(expr)
                }
            },
            Select { expr, .. } => scratch.extend(expr),
            HStack { exprs, .. } => scratch.extend(exprs),
            Sort { by_column, .. } => scratch.extend(by_column),
            GroupBy { keys, aggs, .. } => {
                scratch.extend(keys);
                scratch.extend(aggs);
            },
            Join {
                left_on, right_on, ..
            } => {
                scratch.extend(left_on);
                scratch.extend(right_on);
            },
            Cache { .. }
            | Distinct { .. }
            | Slice { .. }
            | MapFunction { .. }
            | Union { .. }
            | HConcat { .. }
            | ExtContext { .. }
            | Sink { .. }
            | IR { .. } => (),
            #[cfg(feature = "python")]
            PythonScan { .. } => (),
        }
    }
}

pub struct DslPlanIter<'a> {
    stack: Vec<&'a DslPlan>,
}

impl<'a> Iterator for DslPlanIter<'a> {
    type Item = &'a DslPlan;

    fn next(&mut self) -> Option<Self::Item> {
        self.stack
            .pop()
            .inspect(|next| next.inputs(&mut self.stack))
    }
}

impl<'a> IntoIterator for &'a DslPlan {
    type Item = &'a DslPlan;
    type IntoIter = DslPlanIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        DslPlanIter { stack: vec![self] }
    }
}
