//! Lazy variant of a [DataFrame](crate::prelude::DataFrame).
use crate::frame::select::Selection;
use crate::{lazy::prelude::*, prelude::*};
use std::sync::Arc;

impl DataFrame {
    /// Convert the `DataFrame` into a lazy `DataFrame`
    pub fn lazy(self) -> LazyFrame {
        LogicalPlanBuilder::from_existing_df(self).build().into()
    }
}

/// Lazy abstraction over an eager `DataFrame`.
/// It really is an abstraction over a logical plan. The methods of this struct will incrementally
/// modify a logical plan until output is requested (via [collect](crate::lazy::frame::LazyFrame::collect))
#[derive(Clone)]
pub struct LazyFrame {
    pub(crate) logical_plan: LogicalPlan,
    projection_pushdown: bool,
    predicate_pushdown: bool,
    type_coercion: bool,
}

impl Default for LazyFrame {
    fn default() -> Self {
        LazyFrame {
            logical_plan: LogicalPlan::default(),
            projection_pushdown: false,
            predicate_pushdown: false,
            type_coercion: false,
        }
    }
}

impl From<LogicalPlan> for LazyFrame {
    fn from(plan: LogicalPlan) -> Self {
        Self {
            logical_plan: plan,
            projection_pushdown: true,
            predicate_pushdown: true,
            type_coercion: true,
        }
    }
}

struct OptState {
    projection_pushdown: bool,
    predicate_pushdown: bool,
    type_coercion: bool,
}

impl LazyFrame {
    fn get_plan_builder(self) -> LogicalPlanBuilder {
        LogicalPlanBuilder::from(self.logical_plan)
    }

    fn get_opt_state(&self) -> OptState {
        OptState {
            projection_pushdown: self.projection_pushdown,
            predicate_pushdown: self.predicate_pushdown,
            type_coercion: self.type_coercion,
        }
    }

    fn from_logical_plan(logical_plan: LogicalPlan, opt_state: OptState) -> Self {
        LazyFrame {
            logical_plan,
            projection_pushdown: opt_state.projection_pushdown,
            predicate_pushdown: opt_state.predicate_pushdown,
            type_coercion: opt_state.type_coercion,
        }
    }

    /// Toggle projection pushdown optimization on or off.
    pub fn with_projection_pushdown_optimization(mut self, toggle: bool) -> Self {
        self.projection_pushdown = toggle;
        self
    }

    /// Toggle predicate pushdown optimization on or off.
    pub fn with_predicate_pushdown_optimization(mut self, toggle: bool) -> Self {
        self.predicate_pushdown = toggle;
        self
    }

    /// Toggle type coercion optimization on or off.
    pub fn with_type_coercion_optimization(mut self, toggle: bool) -> Self {
        self.type_coercion = toggle;
        self
    }

    /// Describe the logical plan.
    pub fn describe_plan(&self) -> String {
        self.logical_plan.describe()
    }

    /// Describe the optimized logical plan.
    pub fn describe_optimized_plan(&self) -> Result<String> {
        let logical_plan = self.clone().get_plan_builder().build();
        let predicate_pushdown_opt = PredicatePushDown {};
        let projection_pushdown_opt = ProjectionPushDown {};
        let logical_plan = predicate_pushdown_opt.optimize(logical_plan)?;
        let logical_plan = projection_pushdown_opt.optimize(logical_plan)?;
        Ok(logical_plan.describe())
    }

    /// Add a sort operation to the logical plan.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars::prelude::*;
    ///
    /// /// Sort DataFrame by 'sepal.width' column
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///         .sort("sepal.width", false)
    /// }
    /// ```
    pub fn sort(self, by_column: &str, reverse: bool) -> Self {
        let opt_state = self.get_opt_state();
        let lp = self
            .get_plan_builder()
            .sort(by_column.into(), reverse)
            .build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Execute all the lazy operations and collect them into a [DataFrame](crate::prelude::DataFrame).
    /// Before execution the query is being optimized.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars::prelude::*;
    /// use polars::lazy::dsl::*;
    ///
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///       df.lazy()
    ///         .groupby("foo")
    ///         .agg(vec!(col("bar").agg_sum(),
    ///                   col("ham").agg_mean().alias("avg_ham")))
    ///         .collect()
    /// }
    /// ```
    pub fn collect(self) -> Result<DataFrame> {
        let predicate_pushdown = self.predicate_pushdown;
        let projection_pushdown = self.projection_pushdown;
        let type_coercion = self.type_coercion;
        let mut logical_plan = self.get_plan_builder().build();

        let predicate_pushdown_opt = PredicatePushDown {};
        let projection_pushdown_opt = ProjectionPushDown {};
        let type_coercion_opt = TypeCoercion {};

        if cfg!(debug_assertions) {
            // check that the optimization don't interfere with the schema result.
            let prev_schema = logical_plan.schema().clone();
            if projection_pushdown {
                logical_plan = projection_pushdown_opt.optimize(logical_plan)?;
            }
            assert_eq!(logical_plan.schema(), &prev_schema);

            let prev_schema = logical_plan.schema().clone();
            if predicate_pushdown {
                logical_plan = predicate_pushdown_opt.optimize(logical_plan)?;
            }
            assert_eq!(logical_plan.schema(), &prev_schema);
        } else {
            // NOTE: the order is important. Projection pushdown must be before predicate pushdown,
            // The projection may have aliases that interfere with the predicate expressions.
            if projection_pushdown {
                logical_plan = projection_pushdown_opt.optimize(logical_plan)?;
            }
            if predicate_pushdown {
                logical_plan = predicate_pushdown_opt.optimize(logical_plan)?;
            }
        };

        if type_coercion {
            logical_plan = type_coercion_opt.optimize(logical_plan)?;
        }

        let planner = DefaultPlanner::default();
        let physical_plan = planner.create_physical_plan(&logical_plan)?;
        physical_plan.execute()
    }

    /// Filter by some predicate expression.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars::prelude::*;
    /// use polars::lazy::dsl::*;
    ///
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///         .filter(col("sepal.width").is_not_null())
    ///         .select(&[col("sepal.width"), col("sepal.length")])
    /// }
    /// ```
    pub fn filter(self, predicate: Expr) -> Self {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().filter(predicate).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Select (and rename) columns from the query.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars::prelude::*;
    /// use polars::lazy::dsl::*;
    ///
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///         .select(&[col("foo"),
    ///                   col("bar").alias("ham")])
    /// }
    /// ```
    pub fn select<E: AsRef<[Expr]>>(self, exprs: E) -> Self {
        let opt_state = self.get_opt_state();
        let lp = self
            .get_plan_builder()
            .project(exprs.as_ref().to_vec())
            .build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Group by and aggregate.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars::prelude::*;
    /// use polars::lazy::dsl::*;
    ///
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///        .groupby("date")
    ///        .agg(vec![
    ///            col("rain").agg_min(),
    ///            col("rain").agg_sum(),
    ///            col("rain").agg_quantile(0.5).alias("median_rain"),
    ///        ])
    ///        .sort("date", false)
    /// }
    /// ```
    pub fn groupby<'g, J, S: Selection<'g, J>>(self, by: S) -> LazyGroupBy {
        let opt_state = self.get_opt_state();
        let keys = by
            .to_selection_vec()
            .iter()
            .map(|&s| s.to_owned())
            .collect();
        LazyGroupBy {
            logical_plan: self.logical_plan,
            opt_state,
            keys,
        }
    }

    /// Join query with other lazy query.
    pub fn left_join(self, other: LazyFrame, left_on: &str, right_on: &str) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self
            .get_plan_builder()
            .join(
                other.logical_plan,
                JoinType::Left,
                Arc::new(left_on.into()),
                Arc::new(right_on.into()),
            )
            .build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Join query with other lazy query.
    pub fn outer_join(self, other: LazyFrame, left_on: &str, right_on: &str) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self
            .get_plan_builder()
            .join(
                other.logical_plan,
                JoinType::Outer,
                Arc::new(left_on.into()),
                Arc::new(right_on.into()),
            )
            .build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Join query with other lazy query.
    pub fn inner_join(self, other: LazyFrame, left_on: &str, right_on: &str) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self
            .get_plan_builder()
            .join(
                other.logical_plan,
                JoinType::Inner,
                Arc::new(left_on.into()),
                Arc::new(right_on.into()),
            )
            .build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Add a column to a DataFrame
    pub fn with_column(self, expr: Expr) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().with_columns(vec![expr]).build();
        Self::from_logical_plan(lp, opt_state)
    }

    pub fn with_columns(self, exprs: Vec<Expr>) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().with_columns(exprs).build();
        Self::from_logical_plan(lp, opt_state)
    }
}

/// Utility struct for lazy groupby operation.
pub struct LazyGroupBy {
    pub(crate) logical_plan: LogicalPlan,
    opt_state: OptState,
    keys: Vec<String>,
}

impl LazyGroupBy {
    /// Group by and aggregate.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars::prelude::*;
    /// use polars::lazy::dsl::*;
    ///
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///        .groupby("date")
    ///        .agg(vec![
    ///            col("rain").agg_min(),
    ///            col("rain").agg_sum(),
    ///            col("rain").agg_quantile(0.5).alias("median_rain"),
    ///        ])
    ///        .sort("date", false)
    /// }
    /// ```
    pub fn agg(self, aggs: Vec<Expr>) -> LazyFrame {
        let lp = LogicalPlanBuilder::from(self.logical_plan)
            .groupby(Arc::new(self.keys), aggs)
            .build();
        LazyFrame::from_logical_plan(lp, self.opt_state)
    }
}

#[cfg(test)]
mod test {
    use crate::lazy::prelude::*;
    use crate::lazy::tests::get_df;
    use crate::prelude::*;

    #[test]
    fn test_lazy_ternary() {
        let df = get_df()
            .lazy()
            .with_column(
                when(col("sepal.length").lt(lit(5.0)))
                    .then(lit(10))
                    .otherwise(lit(1)
                    )
                    .alias("new")
                ,
            )
            .collect()
            .unwrap();
        assert_eq!(Some(43), df.column("new").unwrap().sum::<i32>());
    }

    #[test]
    fn test_lazy_with_column() {
        let df = get_df()
            .lazy()
            .with_column(lit(10).alias("foo"))
            .collect()
            .unwrap();
        println!("{:?}", df);
        assert_eq!(df.width(), 6);
        assert!(df.column("foo").is_ok());

        let df = get_df()
            .lazy()
            .with_column(lit(10).alias("foo"))
            .select(&[col("foo"), col("sepal.width")])
            .collect()
            .unwrap();
        println!("{:?}", df);
    }

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

    #[test]
    fn test_lazy_agg() {
        let s0 = Date32Chunked::parse_from_str_slice(
            "date",
            &[
                "2020-08-21",
                "2020-08-21",
                "2020-08-22",
                "2020-08-23",
                "2020-08-22",
            ],
            "%Y-%m-%d",
        )
        .into_series();
        let s1 = Series::new("temp", [20, 10, 7, 9, 1].as_ref());
        let s2 = Series::new("rain", [0.2, 0.1, 0.3, 0.1, 0.01].as_ref());
        let df = DataFrame::new(vec![s0, s1, s2]).unwrap();

        let lf = df
            .lazy()
            .groupby("date")
            .agg(vec![
                col("rain").agg_min(),
                col("rain").agg_sum(),
                col("rain").agg_quantile(0.5).alias("median_rain"),
            ])
            .sort("date", false);

        println!("{:?}", lf.describe_plan());
        println!("{:?}", lf.describe_optimized_plan());
        let new = lf.collect().unwrap();
        println!("{:?}", new);
    }
}
