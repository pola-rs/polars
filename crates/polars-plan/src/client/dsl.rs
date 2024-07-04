use crate::dsl::Expr;
use crate::prelude::DslPlan;

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
            Scan { .. } | DataFrameScan { .. } | PythonScan { .. } => (),
        }
    }

    pub(super) fn get_expr<'a>(&'a self, scratch: &mut Vec<&'a Expr>) {
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
            PythonScan { .. }
            | Cache { .. }
            | Distinct { .. }
            | Slice { .. }
            | MapFunction { .. }
            | Union { .. }
            | HConcat { .. }
            | ExtContext { .. }
            | Sink { .. }
            | IR { .. } => (),
        }
    }
}

pub struct DslPlanIter<'a> {
    stack: Vec<&'a DslPlan>,
}

impl<'a> Iterator for DslPlanIter<'a> {
    type Item = &'a DslPlan;

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop().map(|next| {
            next.inputs(&mut self.stack);
            next
        })
    }
}

impl<'a> IntoIterator for &'a DslPlan {
    type Item = &'a DslPlan;
    type IntoIter = DslPlanIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        DslPlanIter { stack: vec![self] }
    }
}
