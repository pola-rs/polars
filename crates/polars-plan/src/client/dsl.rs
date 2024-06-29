use crate::dsl::Expr;
use crate::prelude::DslPlan;

impl DslPlan {
    fn inputs<'a>(&'a self, scratch: &mut Vec<&'a DslPlan>) {
        match self {
            DslPlan::Join {
                input_left,
                input_right,
                ..
            } => {
                scratch.push(input_left);
                scratch.push(input_right);
            },
            DslPlan::Union { inputs, .. } => scratch.extend(inputs),
            DslPlan::Filter { input, .. } => scratch.push(input),
            _ => todo!(),
        }
    }

    pub(super) fn get_expr(&self, scratch: &mut Vec<&Expr>) {
        todo!()
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
