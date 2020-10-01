use crate::frame::select::Selection;
use crate::{lazy::prelude::*, prelude::*};
use std::rc::Rc;

impl DataFrame {
    /// Convert the `DataFrame` into a lazy `DataFrame`
    pub fn lazy(self) -> LazyFrame {
        LogicalPlanBuilder::from_existing_df(self).build().into()
    }
}

/// abstraction over a logical plan
#[derive(Clone)]
pub struct LazyFrame {
    pub(crate) logical_plan: LogicalPlan,
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

    pub fn describe_plan(&self) -> String {
        self.logical_plan.describe()
    }

    pub fn describe_optimized_plan(&self) -> String {
        let logical_plan = self.clone().get_plan_builder().build();
        let predicate_pushdown_opt = PredicatePushDown {};
        let projection_pushdown_opt = ProjectionPushDown {};
        let logical_plan = predicate_pushdown_opt.optimize(logical_plan);
        let logical_plan = projection_pushdown_opt.optimize(logical_plan);
        logical_plan.describe()
    }

    pub fn sort(self, by_column: &str, reverse: bool) -> Self {
        self.get_plan_builder()
            .sort(by_column.into(), reverse)
            .build()
            .into()
    }

    pub fn collect(self) -> Result<DataFrame> {
        let logical_plan = self.get_plan_builder().build();

        let predicate_pushdown_opt = PredicatePushDown {};
        let projection_pushdown_opt = ProjectionPushDown {};

        let logical_plan = if cfg!(debug_assertions) {
            // check that the optimization don't interfere with the schema result.
            let prev_schema = logical_plan.schema().clone();
            let logical_plan = predicate_pushdown_opt.optimize(logical_plan);
            assert_eq!(logical_plan.schema(), &prev_schema);
            let prev_schema = logical_plan.schema().clone();
            let logical_plan = projection_pushdown_opt.optimize(logical_plan);
            assert_eq!(logical_plan.schema(), &prev_schema);
            logical_plan
        } else {
            // NOTE: the order is important. Projection pushdown must be later than predicate pushdown,
            // because I want the projections to occur before the filtering.
            let logical_plan = predicate_pushdown_opt.optimize(logical_plan);
            projection_pushdown_opt.optimize(logical_plan)
        };
        let planner = DefaultPlanner::default();
        let physical_plan = planner.create_physical_plan(&logical_plan)?;
        physical_plan.execute()
    }

    pub fn filter(self, predicate: Expr) -> Self {
        self.get_plan_builder().filter(predicate).build().into()
    }

    pub fn select<E: AsRef<[Expr]>>(self, exprs: E) -> Self {
        self.get_plan_builder()
            .project(exprs.as_ref().to_vec())
            .build()
            .into()
    }

    pub fn groupby<'g, J, S: Selection<'g, J>>(self, by: S) -> LazyGroupBy {
        let keys = by
            .to_selection_vec()
            .iter()
            .map(|&s| s.to_owned())
            .collect();
        LazyGroupBy {
            logical_plan: self.logical_plan,
            keys,
        }
    }

    pub fn left_join(self, other: LazyFrame, left_on: &str, right_on: &str) -> LazyFrame {
        self.get_plan_builder()
            .join(
                other.logical_plan,
                JoinType::Left,
                Rc::new(left_on.into()),
                Rc::new(right_on.into()),
            )
            .build()
            .into()
    }
}

pub struct LazyGroupBy {
    pub(crate) logical_plan: LogicalPlan,
    keys: Vec<String>,
}

impl LazyGroupBy {
    pub fn agg(self, aggs: Vec<Expr>) -> LazyFrame {
        LogicalPlanBuilder::from(self.logical_plan)
            .groupby(Rc::new(self.keys), aggs)
            .build()
            .into()
    }
}

#[cfg(test)]
mod test {
    use crate::lazy::prelude::*;
    use crate::lazy::tests::get_df;
    use crate::prelude::*;

    #[test]
    fn test_lazy_exec() {
        let df = get_df();
        let new = df
            .clone()
            .lazy()
            .select(&[col("sepal.width"), col("variety")])
            .sort("sepal.width", false)
            .collect();
        println!("{:?}", new);

        let new = df
            .clone()
            .lazy()
            .filter(not(col("sepal.width").lt(lit(3.5))))
            .collect()
            .unwrap();

        let check = new.column("sepal.width").unwrap().f64().unwrap().gt(3.4);
        assert!(check.all_true())
    }

    #[test]
    fn test_lazy_alias() {
        let df = get_df();
        let new = df
            .lazy()
            .select(&[col("sepal.width").alias("petals"), col("sepal.width")])
            .collect()
            .unwrap();
        assert_eq!(new.columns(), &["petals", "sepal.width"]);
    }

    #[test]
    fn test_lazy_is_null() {
        let df = get_df();
        let new = df
            .clone()
            .lazy()
            .filter(col("sepal.width").is_null())
            .collect()
            .unwrap();

        assert_eq!(new.height(), 0);

        let new = df
            .clone()
            .lazy()
            .filter(col("sepal.width").is_not_null())
            .collect()
            .unwrap();
        assert_eq!(new.height(), df.height());

        let new = df
            .lazy()
            .groupby("variety")
            .agg(vec![col("sepal.width").agg_min()])
            .collect()
            .unwrap();

        println!("{:?}", new);
        assert_eq!(new.shape(), (1, 2));
    }

    #[test]
    fn test_lazy_pushdown_through_agg() {
        // An aggregation changes the schema names, check if the pushdown succeeds.
        let df = get_df();
        let new = df
            .lazy()
            .groupby(&["variety"])
            .agg(vec![
                col("sepal.length").agg_min(),
                col("petal.length").agg_min().alias("foo"),
            ])
            .select(&[col("foo")])
            // second selection is to test if optimizer can handle that
            .select(&[col("foo").alias("bar")])
            .collect()
            .unwrap();

        println!("{:?}", new);
    }
}
