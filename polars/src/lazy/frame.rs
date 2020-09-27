use crate::frame::select::Selection;
use crate::{lazy::prelude::*, prelude::*};

impl DataFrame {
    /// Convert the `DataFrame` into a lazy `DataFrame`
    pub fn lazy(self) -> LazyFrame {
        LogicalPlanBuilder::from_existing_df(self).build().into()
    }
}

/// abstraction over a logical plan
pub struct LazyFrame {
    logical_plan: LogicalPlan,
}

impl From<LogicalPlan> for LazyFrame {
    fn from(plan: LogicalPlan) -> Self {
        Self { logical_plan: plan }
    }
}

impl LazyFrame {
    fn get_plan_builder(self) -> LogicalPlanBuilder {
        LogicalPlanBuilder::from(self.logical_plan)
    }

    pub fn select<'a, K, S: Selection<'a, K>>(self, columns: S) -> Self {
        let expr = columns
            .to_selection_vec()
            .into_iter()
            .map(|s| col(s))
            .collect::<Vec<_>>();
        self.get_plan_builder().project(expr).build().into()
    }

    pub fn sort(self, by_column: &str, reverse: bool) -> Self {
        let expr = vec![Expr::Sort {
            reverse,
            expr: Box::new(col(by_column)),
        }];
        self.get_plan_builder().sort(expr).build().into()
    }

    pub fn collect(self) -> Result<DataFrame> {
        let logical_plan = self.get_plan_builder().build();
        // todo: optimize plan.

        let planner = DefaultPlanner::default();
        let physical_plan = planner.create_physical_plan(&logical_plan)?;
        physical_plan.execute()
    }
}

#[cfg(test)]
mod test {
    use crate::lazy::tests::get_df;

    #[test]
    fn test_lazy_exec() {
        let df = get_df();
        let new = df
            .lazy()
            .select("sepal.width")
            .sort("sepal.width", false)
            .collect();

        println!("{:?}", new)
    }
}
