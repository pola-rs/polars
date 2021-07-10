//! Lazy variant of a [DataFrame](polars_core::frame::DataFrame).
#[cfg(any(feature = "parquet", feature = "csv-file"))]
use polars_core::datatypes::PlHashMap;
use polars_core::frame::hash_join::JoinType;
use polars_core::prelude::*;
use polars_core::toggle_string_cache;
use std::sync::Arc;

use crate::logical_plan::optimizer::aggregate_pushdown::AggregatePushdown;
#[cfg(any(feature = "parquet", feature = "csv-file"))]
use crate::logical_plan::optimizer::aggregate_scan_projections::AggScanProjection;
use crate::logical_plan::optimizer::simplify_expr::SimplifyExprRule;
use crate::logical_plan::optimizer::stack_opt::{OptimizationRule, StackOptimizer};
use crate::logical_plan::optimizer::{
    predicate_pushdown::PredicatePushDown, projection_pushdown::ProjectionPushDown,
};
use crate::physical_plan::state::ExecutionState;
#[cfg(any(feature = "parquet", feature = "csv-file"))]
use crate::prelude::aggregate_scan_projections::agg_projection;
use crate::prelude::simplify_expr::SimplifyBooleanRule;
use crate::utils::combine_predicates_expr;
use crate::{logical_plan::FETCH_ROWS, prelude::*};
use polars_io::csv::NullValues;

#[derive(Clone)]
#[cfg(feature = "csv-file")]
pub struct LazyCsvReader<'a> {
    path: String,
    delimiter: u8,
    has_header: bool,
    ignore_errors: bool,
    skip_rows: usize,
    stop_after_n_rows: Option<usize>,
    cache: bool,
    schema: Option<SchemaRef>,
    schema_overwrite: Option<&'a Schema>,
    low_memory: bool,
    comment_char: Option<u8>,
    null_values: Option<NullValues>,
}

#[cfg(feature = "csv-file")]
impl<'a> LazyCsvReader<'a> {
    pub fn new(path: String) -> Self {
        LazyCsvReader {
            path,
            delimiter: b',',
            has_header: true,
            ignore_errors: false,
            skip_rows: 0,
            stop_after_n_rows: None,
            cache: true,
            schema: None,
            schema_overwrite: None,
            low_memory: false,
            comment_char: None,
            null_values: None,
        }
    }

    /// Try to stop parsing when `n` rows are parsed. During multithreaded parsing the upper bound `n` cannot
    /// be guaranteed.
    pub fn with_stop_after_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.stop_after_n_rows = num_rows;
        self
    }

    /// Continue with next batch when a ParserError is encountered.
    pub fn with_ignore_parser_errors(mut self, ignore: bool) -> Self {
        self.ignore_errors = ignore;
        self
    }

    /// Set the CSV file's schema
    pub fn with_schema(mut self, schema: SchemaRef) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Skip the first `n` rows during parsing.
    pub fn with_skip_rows(mut self, skip_rows: usize) -> Self {
        self.skip_rows = skip_rows;
        self
    }

    /// Overwrite the schema with the dtypes in this given Schema. The given schema may be a subset
    /// of the total schema.
    pub fn with_dtype_overwrite(mut self, schema: Option<&'a Schema>) -> Self {
        self.schema_overwrite = schema;
        self
    }

    /// Set whether the CSV file has headers
    pub fn has_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    /// Set the CSV file's column delimiter as a byte character
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Set the comment character. Lines starting with this character will be ignored.
    pub fn with_comment_char(mut self, comment_char: Option<u8>) -> Self {
        self.comment_char = comment_char;
        self
    }

    /// Set values that will be interpreted as missing/ null.
    pub fn with_null_values(mut self, null_values: Option<NullValues>) -> Self {
        self.null_values = null_values;
        self
    }

    /// Cache the DataFrame after reading.
    pub fn with_cache(mut self, cache: bool) -> Self {
        self.cache = cache;
        self
    }

    /// Reduce memory usage in expensive of performance
    pub fn low_memory(mut self, toggle: bool) -> Self {
        self.low_memory = toggle;
        self
    }

    pub fn finish(self) -> LazyFrame {
        let mut lf: LazyFrame = LogicalPlanBuilder::scan_csv(
            self.path,
            self.delimiter,
            self.has_header,
            self.ignore_errors,
            self.skip_rows,
            self.stop_after_n_rows,
            self.cache,
            self.schema,
            self.schema_overwrite,
            self.low_memory,
            self.comment_char,
            self.null_values,
        )
        .build()
        .into();
        lf.opt_state.agg_scan_projection = true;
        lf
    }
}

#[derive(Copy, Clone, Debug)]
pub struct JoinOptions {
    pub allow_parallel: bool,
    pub force_parallel: bool,
}

impl Default for JoinOptions {
    fn default() -> Self {
        JoinOptions {
            allow_parallel: true,
            force_parallel: false,
        }
    }
}

pub trait IntoLazy {
    fn lazy(self) -> LazyFrame;
}

impl IntoLazy for DataFrame {
    /// Convert the `DataFrame` into a lazy `DataFrame`
    fn lazy(self) -> LazyFrame {
        LogicalPlanBuilder::from_existing_df(self).build().into()
    }
}

/// Lazy abstraction over an eager `DataFrame`.
/// It really is an abstraction over a logical plan. The methods of this struct will incrementally
/// modify a logical plan until output is requested (via [collect](crate::frame::LazyFrame::collect))
#[derive(Clone)]
pub struct LazyFrame {
    pub(crate) logical_plan: LogicalPlan,
    pub(crate) opt_state: OptState,
}

impl Default for LazyFrame {
    fn default() -> Self {
        LazyFrame {
            logical_plan: LogicalPlan::default(),
            opt_state: Default::default(),
        }
    }
}

impl From<LogicalPlan> for LazyFrame {
    fn from(plan: LogicalPlan) -> Self {
        Self {
            logical_plan: plan,
            opt_state: Default::default(),
        }
    }
}

#[derive(Copy, Clone)]
/// State of the allowed optimizations
pub struct OptState {
    pub projection_pushdown: bool,
    pub predicate_pushdown: bool,
    pub type_coercion: bool,
    pub simplify_expr: bool,
    /// Make sure that all needed columns are scannedn
    pub agg_scan_projection: bool,
    pub aggregate_pushdown: bool,
    pub global_string_cache: bool,
    pub join_pruning: bool,
}

impl Default for OptState {
    fn default() -> Self {
        OptState {
            projection_pushdown: true,
            predicate_pushdown: true,
            type_coercion: true,
            simplify_expr: true,
            global_string_cache: true,
            join_pruning: true,
            // will be toggled by a scan operation such as csv scan or parquet scan
            agg_scan_projection: false,
            aggregate_pushdown: false,
        }
    }
}

/// AllowedOptimizations
pub type AllowedOptimizations = OptState;

impl LazyFrame {
    /// Create a LazyFrame directly from a parquet scan.
    #[cfg(feature = "parquet")]
    pub fn new_from_parquet(path: String, stop_after_n_rows: Option<usize>, cache: bool) -> Self {
        let mut lf: LazyFrame = LogicalPlanBuilder::scan_parquet(path, stop_after_n_rows, cache)
            .build()
            .into();
        lf.opt_state.agg_scan_projection = true;
        lf
    }

    /// Get a dot language representation of the LogicalPlan.
    pub fn to_dot(&self, optimized: bool) -> Result<String> {
        let mut s = String::with_capacity(512);

        let mut logical_plan = self.clone().get_plan_builder().build();
        if optimized {
            // initialize arena's
            let mut expr_arena = Arena::with_capacity(64);
            let mut lp_arena = Arena::with_capacity(32);

            let lp_top = self.clone().optimize(&mut lp_arena, &mut expr_arena)?;
            logical_plan = node_to_lp(lp_top, &mut expr_arena, &mut lp_arena);
        }

        logical_plan.dot(&mut s, (0, 0), "").expect("io error");
        s.push_str("\n}");
        Ok(s)
    }

    fn get_plan_builder(self) -> LogicalPlanBuilder {
        LogicalPlanBuilder::from(self.logical_plan)
    }

    fn get_opt_state(&self) -> OptState {
        self.opt_state
    }

    fn from_logical_plan(logical_plan: LogicalPlan, opt_state: OptState) -> Self {
        LazyFrame {
            logical_plan,
            opt_state,
        }
    }

    #[cfg(test)]
    pub(crate) fn into_alp(self) -> (Node, Arena<AExpr>, Arena<ALogicalPlan>) {
        let mut expr_arena = Arena::with_capacity(64);
        let mut lp_arena = Arena::with_capacity(32);
        let root = to_alp(self.logical_plan, &mut expr_arena, &mut lp_arena);
        (root, expr_arena, lp_arena)
    }

    /// Toggle projection pushdown optimization.
    pub fn with_projection_pushdown(mut self, toggle: bool) -> Self {
        self.opt_state.projection_pushdown = toggle;
        self
    }

    /// Toggle predicate pushdown optimization.
    pub fn with_predicate_pushdown(mut self, toggle: bool) -> Self {
        self.opt_state.predicate_pushdown = toggle;
        self
    }

    /// Toggle type coercion optimization.
    pub fn with_type_coercion(mut self, toggle: bool) -> Self {
        self.opt_state.type_coercion = toggle;
        self
    }

    /// Toggle expression simplification optimization on or off
    pub fn with_simplify_expr(mut self, toggle: bool) -> Self {
        self.opt_state.simplify_expr = toggle;
        self
    }

    /// Toggle aggregate pushdown.
    pub fn with_aggregate_pushdown(mut self, toggle: bool) -> Self {
        self.opt_state.aggregate_pushdown = toggle;
        self
    }

    /// Toggle global string cache.
    pub fn with_string_cache(mut self, toggle: bool) -> Self {
        self.opt_state.global_string_cache = toggle;
        self
    }

    /// Toggle join pruning optimization
    pub fn with_join_pruning(mut self, toggle: bool) -> Self {
        self.opt_state.join_pruning = toggle;
        self
    }

    /// Describe the logical plan.
    pub fn describe_plan(&self) -> String {
        self.logical_plan.describe()
    }

    /// Describe the optimized logical plan.
    pub fn describe_optimized_plan(&self) -> Result<String> {
        let mut expr_arena = Arena::with_capacity(512);
        let mut lp_arena = Arena::with_capacity(512);
        let lp_top = self.clone().optimize(&mut lp_arena, &mut expr_arena)?;
        let logical_plan = node_to_lp(lp_top, &mut expr_arena, &mut lp_arena);
        Ok(logical_plan.describe())
    }

    /// Add a sort operation to the logical plan.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
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
            .sort(vec![col(by_column)], vec![reverse])
            .build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Add a sort operation to the logical plan.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    ///
    /// /// Sort DataFrame by 'sepal.width' column
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///         .sort_by_exprs(vec![col("sepal.width")], vec![false])
    /// }
    /// ```
    pub fn sort_by_exprs(self, by_exprs: Vec<Expr>, reverse: Vec<bool>) -> Self {
        if by_exprs.is_empty() {
            self
        } else {
            let opt_state = self.get_opt_state();
            let lp = self.get_plan_builder().sort(by_exprs, reverse).build();
            Self::from_logical_plan(lp, opt_state)
        }
    }

    /// Reverse the DataFrame
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    ///
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///         .reverse()
    /// }
    /// ```
    pub fn reverse(self) -> Self {
        self.select_local(vec![col("*").reverse()])
    }

    /// Rename a column in the DataFrame
    pub fn with_column_renamed(self, existing_name: &str, new_name: &str) -> Self {
        let schema = self.logical_plan.schema();
        let schema = schema
            .rename(&[existing_name], &[new_name])
            .expect("cannot rename non existing column");

        // first make sure that the column is projected, then we
        let init = self.with_column(col(existing_name));

        let existing_name = existing_name.to_string();
        let new_name = new_name.to_string();
        let f = move |mut df: DataFrame| {
            df.rename(&existing_name, &new_name)?;
            Ok(df)
        };
        init.map(f, Some(AllowedOptimizations::default()), Some(schema))
    }

    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with `Nones`.
    ///
    /// See the method on [Series](polars_core::series::SeriesTrait::shift) for more info on the `shift` operation.
    pub fn shift(self, periods: i64) -> Self {
        self.select_local(vec![col("*").shift(periods)])
    }

    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with the result of the `fill_value` expression.
    ///
    /// See the method on [Series](polars_core::series::SeriesTrait::shift) for more info on the `shift` operation.
    pub fn shift_and_fill(self, periods: i64, fill_value: Expr) -> Self {
        self.select_local(vec![col("*").shift_and_fill(periods, fill_value)])
    }

    /// Fill none values in the DataFrame
    pub fn fill_none(self, fill_value: Expr) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().fill_none(fill_value).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Caches the result into a new LazyFrame. This should be used to prevent computations
    /// running multiple times
    pub fn cache(self) -> Self {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().cache().build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Fetch is like a collect operation, but it overwrites the number of rows read by every scan
    /// operation. This is a utility that helps debug a query on a smaller number of rows.
    ///
    /// Note that the fetch does not guarantee the final number of rows in the DataFrame.
    /// Filter, join operations and a lower number of rows available in the scanned file influence
    /// the final number of rows.
    pub fn fetch(self, n_rows: usize) -> Result<DataFrame> {
        FETCH_ROWS.with(|fetch_rows| fetch_rows.set(Some(n_rows)));
        let res = self.collect();
        FETCH_ROWS.with(|fetch_rows| fetch_rows.set(None));
        res
    }

    pub fn optimize(
        self,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<Node> {
        // get toggle values
        let predicate_pushdown = self.opt_state.predicate_pushdown;
        let projection_pushdown = self.opt_state.projection_pushdown;
        let type_coercion = self.opt_state.type_coercion;
        let simplify_expr = self.opt_state.simplify_expr;

        #[cfg(any(feature = "parquet", feature = "csv-file"))]
        let agg_scan_projection = self.opt_state.agg_scan_projection;
        let aggregate_pushdown = self.opt_state.aggregate_pushdown;

        let logical_plan = self.get_plan_builder().build();

        // gradually fill the rules passed to the optimizer
        let mut rules: Vec<Box<dyn OptimizationRule>> = Vec::with_capacity(8);

        let predicate_pushdown_opt = PredicatePushDown::default();
        let projection_pushdown_opt = ProjectionPushDown {};

        // during debug we check if the optimizations have not modified the final schema
        #[cfg(debug_assertions)]
        let prev_schema = logical_plan.schema().clone();

        let mut lp_top = to_alp(logical_plan, expr_arena, lp_arena);

        if projection_pushdown {
            let alp = lp_arena.take(lp_top);
            let alp = projection_pushdown_opt
                .optimize(alp, lp_arena, expr_arena)
                .expect("projection pushdown failed");
            lp_arena.replace(lp_top, alp);
        }

        if predicate_pushdown {
            let alp = lp_arena.take(lp_top);
            let alp = predicate_pushdown_opt
                .optimize(alp, lp_arena, expr_arena)
                .expect("predicate pushdown failed");
            lp_arena.replace(lp_top, alp);
        }

        if type_coercion {
            rules.push(Box::new(TypeCoercionRule {}))
        }

        if simplify_expr {
            rules.push(Box::new(SimplifyExprRule {}));
            rules.push(Box::new(SimplifyBooleanRule {}));
        }
        if aggregate_pushdown {
            rules.push(Box::new(AggregatePushdown::new()))
        }

        #[cfg(any(feature = "parquet", feature = "csv-file"))]
        if agg_scan_projection {
            // scan the LP to aggregate all the column used in scans
            // these columns will be added to the state of the AggScanProjection rule
            let mut columns = PlHashMap::with_capacity(32);
            agg_projection(lp_top, &mut columns, lp_arena);

            let opt = AggScanProjection { columns };
            rules.push(Box::new(opt));
        }

        let opt = StackOptimizer {};
        lp_top = opt.optimize_loop(&mut rules, expr_arena, lp_arena, lp_top);

        // during debug we check if the optimizations have not modified the final schema
        #[cfg(debug_assertions)]
        {
            // only check by names because we may supercast types.
            assert_eq!(
                prev_schema
                    .fields()
                    .iter()
                    .map(|f| f.name())
                    .collect::<Vec<_>>(),
                lp_arena
                    .get(lp_top)
                    .schema(lp_arena)
                    .fields()
                    .iter()
                    .map(|f| f.name())
                    .collect::<Vec<_>>()
            );
        };

        Ok(lp_top)
    }

    /// Execute all the lazy operations and collect them into a [DataFrame](polars_core::frame::DataFrame).
    /// Before execution the query is being optimized.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    ///
    /// fn example(df: DataFrame) -> Result<DataFrame> {
    ///       df.lazy()
    ///         .groupby(vec![col("foo")])
    ///         .agg(vec!(col("bar").sum(),
    ///                   col("ham").mean().alias("avg_ham")))
    ///         .collect()
    /// }
    /// ```
    pub fn collect(self) -> Result<DataFrame> {
        let use_string_cache = self.opt_state.global_string_cache;
        let mut expr_arena = Arena::with_capacity(256);
        let mut lp_arena = Arena::with_capacity(128);
        let lp_top = self.optimize(&mut lp_arena, &mut expr_arena)?;

        // if string cache was already set, we skip this and global settings are respected
        if use_string_cache {
            toggle_string_cache(use_string_cache);
        }
        let planner = DefaultPlanner::default();
        let mut physical_plan =
            planner.create_physical_plan(lp_top, &mut lp_arena, &mut expr_arena)?;

        let state = ExecutionState::new();
        let out = physical_plan.execute(&state);
        if use_string_cache {
            toggle_string_cache(!use_string_cache);
        }
        out
    }

    /// Filter by some predicate expression.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
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
    /// Columns can be selected with [col](crate::dsl::col);
    /// If you want to select all columns use `col("*")`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    ///
    /// /// This function selects column "foo" and column "bar".
    /// /// Column "bar" is renamed to "ham".
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///         .select(&[col("foo"),
    ///                   col("bar").alias("ham")])
    /// }
    ///
    /// /// This function selects all columns except "foo"
    /// fn exclude_a_column(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///         .select(&[col("*"),
    ///                   except("foo")])
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

    /// A projection that doesn't get optimized and may drop projections if they are not in
    /// schema after optimization
    fn select_local(self, exprs: Vec<Expr>) -> Self {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().project_local(exprs).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Group by and aggregate.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    ///
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///        .groupby(vec![col("date")])
    ///        .agg(vec![
    ///            col("rain").min(),
    ///            col("rain").sum(),
    ///            col("rain").quantile(0.5).alias("median_rain"),
    ///        ])
    ///        .sort("date", false)
    /// }
    /// ```
    pub fn groupby(self, by: Vec<Expr>) -> LazyGroupBy {
        let opt_state = self.get_opt_state();
        LazyGroupBy {
            logical_plan: self.logical_plan,
            opt_state,
            keys: by,
        }
    }

    /// Join query with other lazy query.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    /// fn join_dataframes(ldf: LazyFrame, other: LazyFrame) -> LazyFrame {
    ///         ldf
    ///         .left_join(other, col("foo"), col("bar"))
    /// }
    /// ```
    pub fn left_join(self, other: LazyFrame, left_on: Expr, right_on: Expr) -> LazyFrame {
        self.join(other, vec![left_on], vec![right_on], JoinType::Left)
    }

    /// Join query with other lazy query.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    /// fn join_dataframes(ldf: LazyFrame, other: LazyFrame) -> LazyFrame {
    ///         ldf
    ///         .outer_join(other, col("foo"), col("bar"))
    /// }
    /// ```
    pub fn outer_join(self, other: LazyFrame, left_on: Expr, right_on: Expr) -> LazyFrame {
        self.join(other, vec![left_on], vec![right_on], JoinType::Outer)
    }

    /// Join query with other lazy query.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    /// fn join_dataframes(ldf: LazyFrame, other: LazyFrame) -> LazyFrame {
    ///         ldf
    ///         .inner_join(other, col("foo"), col("bar").cast(DataType::Utf8))
    /// }
    /// ```
    pub fn inner_join(self, other: LazyFrame, left_on: Expr, right_on: Expr) -> LazyFrame {
        self.join(other, vec![left_on], vec![right_on], JoinType::Inner)
    }

    /// Creates the cartesian product from both frames, preserves the order of the left keys.
    #[cfg(feature = "cross_join")]
    pub fn cross_join(self, other: LazyFrame) -> LazyFrame {
        self.join(other, vec![], vec![], JoinType::Cross)
    }

    /// Generic join function that can join on multiple columns.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    ///
    /// fn example(ldf: LazyFrame, other: LazyFrame) -> LazyFrame {
    ///         ldf
    ///         .join(other, vec![col("foo"), col("bar")], vec![col("foo"), col("bar")], JoinType::Inner)
    /// }
    /// ```
    pub fn join(
        self,
        other: LazyFrame,
        left_on: Vec<Expr>,
        right_on: Vec<Expr>,
        how: JoinType,
    ) -> LazyFrame {
        self.join_builder()
            .with(other)
            .left_on(left_on)
            .right_on(right_on)
            .how(how)
            .finish()
    }

    pub fn join_builder(self) -> JoinBuilder {
        JoinBuilder::new(self)
    }

    /// Add a column to a DataFrame
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    /// fn add_column(df: DataFrame) -> LazyFrame {
    ///     df.lazy()
    ///         .with_column(
    ///             when(col("sepal.length").lt(lit(5.0)))
    ///             .then(lit(10))
    ///             .otherwise(lit(1))
    ///             .alias("new_column_name"),
    ///             )
    /// }
    /// ```
    pub fn with_column(self, expr: Expr) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().with_columns(vec![expr]).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Add multiple columns to a DataFrame.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    /// fn add_columns(df: DataFrame) -> LazyFrame {
    ///     df.lazy()
    ///         .with_columns(
    ///             vec![lit(10).alias("foo"), lit(100).alias("bar")]
    ///          )
    /// }
    /// ```
    pub fn with_columns(self, exprs: Vec<Expr>) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().with_columns(exprs).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Aggregate all the columns as their maximum values.
    pub fn max(self) -> LazyFrame {
        self.select_local(vec![col("*").max()])
    }

    /// Aggregate all the columns as their minimum values.
    pub fn min(self) -> LazyFrame {
        self.select_local(vec![col("*").min()])
    }

    /// Aggregate all the columns as their sum values.
    pub fn sum(self) -> LazyFrame {
        self.select_local(vec![col("*").sum()])
    }

    /// Aggregate all the columns as their mean values.
    pub fn mean(self) -> LazyFrame {
        self.select_local(vec![col("*").mean()])
    }

    /// Aggregate all the columns as their median values.
    pub fn median(self) -> LazyFrame {
        self.select_local(vec![col("*").median()])
    }

    /// Aggregate all the columns as their quantile values.
    pub fn quantile(self, quantile: f64) -> LazyFrame {
        self.select_local(vec![col("*").quantile(quantile)])
    }

    /// Aggregate all the columns as their standard deviation values.
    pub fn std(self) -> LazyFrame {
        self.select_local(vec![col("*").std()])
    }

    /// Aggregate all the columns as their variance values.
    pub fn var(self) -> LazyFrame {
        self.select_local(vec![col("*").var()])
    }

    /// Apply explode operation. [See eager explode](polars_core::frame::DataFrame::explode).
    pub fn explode(self, columns: &[Expr]) -> LazyFrame {
        let columns = columns
            .iter()
            .map(|e| {
                if let Expr::Column(name) = e {
                    (**name).clone()
                } else {
                    panic!("expected column expression")
                }
            })
            .collect();
        // Note: this operation affects multiple columns. Therefore it isn't implemented as expression.
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().explode(columns).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Drop duplicate rows. [See eager](polars_core::prelude::DataFrame::drop_duplicates).
    pub fn drop_duplicates(self, maintain_order: bool, subset: Option<Vec<String>>) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self
            .get_plan_builder()
            .drop_duplicates(maintain_order, subset)
            .build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Drop null rows.
    ///
    /// Equal to `LazyFrame::filter(col("*").is_not_null())`
    pub fn drop_nulls(self, subset: Option<Vec<Expr>>) -> LazyFrame {
        match subset {
            None => self.filter(col("*").is_not_null()),
            Some(subset) => {
                let it = subset.into_iter().map(|e| e.is_not_null());
                let predicate = combine_predicates_expr(it);
                self.filter(predicate)
            }
        }
    }

    /// Slice the DataFrame.
    pub fn slice(self, offset: i64, len: usize) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().slice(offset, len).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Get the first row.
    pub fn first(self) -> LazyFrame {
        self.slice(0, 1)
    }

    /// Get the last row
    pub fn last(self) -> LazyFrame {
        self.slice(-1, 1)
    }

    /// Get the n last rows
    pub fn tail(self, n: usize) -> LazyFrame {
        let neg_tail = -(n as i64);
        self.slice(neg_tail, n)
    }

    /// Melt the DataFrame from wide to long format
    pub fn melt(self, id_vars: Vec<String>, value_vars: Vec<String>) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self
            .get_plan_builder()
            .melt(Arc::new(id_vars), Arc::new(value_vars))
            .build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Limit the DataFrame to the first `n` rows. Note if you don't want the rows to be scanned,
    /// use [fetch](LazyFrame::fetch).
    pub fn limit(self, n: usize) -> LazyFrame {
        self.slice(0, n)
    }

    /// Apply a function/closure once the logical plan get executed.
    ///
    /// ## Warning
    /// This can blow up in your face if the schema is changed due to the operation. The optimizer
    /// relies on a correct schema.
    ///
    /// You can toggle certain optimizations off.
    pub fn map<F>(
        self,
        function: F,
        optimizations: Option<AllowedOptimizations>,
        schema: Option<Schema>,
    ) -> LazyFrame
    where
        F: DataFrameUdf + 'static,
    {
        let opt_state = self.get_opt_state();
        let lp = self
            .get_plan_builder()
            .map(
                function,
                optimizations.unwrap_or_default(),
                schema.map(Arc::new),
            )
            .build();
        Self::from_logical_plan(lp, opt_state)
    }
}

/// Utility struct for lazy groupby operation.
pub struct LazyGroupBy {
    pub(crate) logical_plan: LogicalPlan,
    opt_state: OptState,
    keys: Vec<Expr>,
}

impl LazyGroupBy {
    /// Group by and aggregate.
    ///
    /// Select a column with [col](crate::dsl::col) and choose an aggregation.
    /// If you want to aggregate all columns use `col("*")`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    ///
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///        .groupby(vec![col("date")])
    ///        .agg(vec![
    ///            col("rain").min(),
    ///            col("rain").sum(),
    ///            col("rain").quantile(0.5).alias("median_rain"),
    ///        ])
    ///        .sort("date", false)
    /// }
    /// ```
    pub fn agg(self, aggs: Vec<Expr>) -> LazyFrame {
        let lp = LogicalPlanBuilder::from(self.logical_plan)
            .groupby(Arc::new(self.keys), aggs, None)
            .build();
        LazyFrame::from_logical_plan(lp, self.opt_state)
    }

    /// Apply a function over the groups as a new `DataFrame`. It is not recommended that you use
    /// this as materializing the `DataFrame` is quite expensive.
    pub fn apply<F>(self, f: F) -> LazyFrame
    where
        F: 'static + Fn(DataFrame) -> Result<DataFrame> + Send + Sync,
    {
        let lp = LogicalPlanBuilder::from(self.logical_plan)
            .groupby(Arc::new(self.keys), vec![], Some(Arc::new(f)))
            .build();
        LazyFrame::from_logical_plan(lp, self.opt_state)
    }
}

pub struct JoinBuilder {
    lf: LazyFrame,
    how: JoinType,
    other: Option<LazyFrame>,
    left_on: Vec<Expr>,
    right_on: Vec<Expr>,
    allow_parallel: bool,
    force_parallel: bool,
}
impl JoinBuilder {
    fn new(lf: LazyFrame) -> Self {
        Self {
            lf,
            other: None,
            how: JoinType::Inner,
            left_on: vec![],
            right_on: vec![],
            allow_parallel: true,
            force_parallel: false,
        }
    }

    pub fn with(mut self, other: LazyFrame) -> Self {
        self.other = Some(other);
        self
    }

    pub fn how(mut self, how: JoinType) -> Self {
        self.how = how;
        self
    }

    pub fn left_on(mut self, on: Vec<Expr>) -> Self {
        self.left_on = on;
        self
    }

    pub fn right_on(mut self, on: Vec<Expr>) -> Self {
        self.right_on = on;
        self
    }
    pub fn allow_parallel(mut self, allow: bool) -> Self {
        self.allow_parallel = allow;
        self
    }
    pub fn force_parallel(mut self, allow: bool) -> Self {
        self.allow_parallel = allow;
        self
    }
    pub fn finish(self) -> LazyFrame {
        let opt_state = self.lf.opt_state;

        let lp = self
            .lf
            .get_plan_builder()
            .join(
                self.other.expect("with not set").logical_plan,
                self.how,
                self.left_on,
                self.right_on,
                self.allow_parallel,
                self.force_parallel,
            )
            .build();
        LazyFrame::from_logical_plan(lp, opt_state)
    }
}

#[cfg(test)]
mod test {
    #[cfg(feature = "temporal")]
    use polars_core::utils::chrono::{NaiveDate, NaiveDateTime, NaiveTime};
    use polars_core::*;

    use crate::functions::{argsort_by, pearson_corr};
    use crate::tests::get_df;

    use super::*;
    use std::iter::FromIterator;

    fn scan_foods_csv() -> LazyFrame {
        let path = "../../examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv";
        LazyCsvReader::new(path.to_string()).finish()
    }

    #[test]
    fn test_lazy_ternary() {
        let df = get_df()
            .lazy()
            .with_column(
                when(col("sepal.length").lt(lit(5.0)))
                    .then(lit(10))
                    .otherwise(lit(1))
                    .alias("new"),
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
        assert_eq!(new.get_column_names(), &["petals", "sepal.width"]);
    }

    #[test]
    fn test_lazy_melt() {
        let df = get_df();
        let out = df
            .lazy()
            .melt(
                vec!["petal.width".to_string(), "petal.length".to_string()],
                vec!["sepal.length".to_string(), "sepal.width".to_string()],
            )
            .filter(col("variable").eq(lit("sepal.length")))
            .select(vec![col("variable"), col("petal.width"), col("value")])
            .collect()
            .unwrap();
        assert_eq!(out.shape(), (7, 3));
        dbg!(out);
    }

    #[test]
    fn test_lazy_drop_nulls() {
        let df = df! {
            "foo" => &[Some(1), None, Some(3)],
            "bar" => &[Some(1), Some(2), None]
        }
        .unwrap();

        let new = df.clone().lazy().drop_nulls(None).collect().unwrap();
        let out = df! {
            "foo" => &[Some(1)],
            "bar" => &[Some(1)]
        }
        .unwrap();
        assert!(new.frame_equal(&out));
    }

    #[test]
    fn test_lazy_udf() {
        let df = get_df();
        let new = df
            .lazy()
            .select(&[col("sepal.width").map(|s| Ok(s * 200.0), None)])
            .collect()
            .unwrap();
        assert_eq!(
            new.column("sepal.width").unwrap().f64().unwrap().get(0),
            Some(700.0)
        );
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
            .groupby(vec![col("variety")])
            .agg(vec![col("sepal.width").min()])
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
            .groupby(vec![col("variety")])
            .agg(vec![
                col("sepal.length").min(),
                col("petal.length").min().alias("foo"),
            ])
            .select(&[col("foo")])
            // second selection is to test if optimizer can handle that
            .select(&[col("foo").alias("bar")])
            .collect()
            .unwrap();

        println!("{:?}", new);
    }

    #[test]
    #[cfg(feature = "temporal")]
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
            .groupby(vec![col("date")])
            .agg(vec![
                col("rain").min(),
                col("rain").sum(),
                col("rain").quantile(0.5).alias("median_rain"),
            ])
            .sort("date", false);

        println!("{:?}", lf.describe_plan());
        println!("{:?}", lf.describe_optimized_plan());
        let new = lf.collect().unwrap();
        println!("{:?}", new);
    }

    #[test]
    fn test_lazy_shift() {
        let df = get_df();
        let new = df
            .lazy()
            .select(&[col("sepal.width").alias("foo").shift(2)])
            .collect()
            .unwrap();
        assert_eq!(new.column("foo").unwrap().f64().unwrap().get(0), None);
    }

    #[test]
    fn test_lazy_ternary_and_predicates() {
        let df = get_df();
        // test if this runs. This failed because is_not_null changes the schema name, so we
        // really need to check the root column
        let ldf = df
            .clone()
            .lazy()
            .with_column(lit(3).alias("foo"))
            .filter(col("foo").is_not_null());
        let _new = ldf.collect().unwrap();

        let ldf = df
            .lazy()
            .with_column(
                when(col("sepal.length").lt(lit(5.0)))
                    .then(
                        lit(3), // is another type on purpose to check type coercion
                    )
                    .otherwise(col("sepal.width"))
                    .alias("foo"),
            )
            .filter(col("foo").gt(lit(3.0)));

        let new = ldf.collect().unwrap();
        dbg!(new);
    }

    #[test]
    fn test_lazy_binary_ops() {
        let df = df!("a" => &[1, 2, 3, 4, 5, ]).unwrap();
        let new = df
            .lazy()
            .select(&[col("a").eq(lit(2)).alias("foo")])
            .collect()
            .unwrap();
        assert_eq!(new.column("foo").unwrap().sum::<i32>(), Some(1));
    }

    fn load_df() -> DataFrame {
        df!("a" => &[1, 2, 3, 4, 5],
                     "b" => &["a", "a", "b", "c", "c"],
                     "c" => &[1, 2, 3, 4, 5]
        )
        .unwrap()
    }

    #[test]
    fn test_lazy_query_1() {
        // test on aggregation pushdown
        // and a filter that is not in the projection
        let df_a = load_df();
        let df_b = df_a.clone();
        df_a.lazy()
            .left_join(df_b.lazy(), col("b"), col("b"))
            .filter(col("a").lt(lit(2)))
            .groupby(vec![col("b")])
            .agg(vec![col("b").first(), col("c").first()])
            .select(&[col("b"), col("c_first")])
            .collect()
            .unwrap();
    }

    #[test]
    fn test_lazy_query_2() {
        let df = load_df();
        let ldf = df
            .lazy()
            .with_column(col("a").map(|s| Ok(s * 2), None).alias("foo"))
            .filter(col("a").lt(lit(2)))
            .select(&[col("b"), col("a")]);

        let new = ldf.collect().unwrap();
        assert_eq!(new.shape(), (1, 2));
    }

    #[test]
    fn test_lazy_query_3() {
        // query checks if schema of scanning is not changed by aggregation
        let _ = scan_foods_csv()
            .groupby(vec![col("calories")])
            .agg(vec![col("fats_g").max()])
            .collect()
            .unwrap();
    }

    #[test]
    fn test_lazy_query_4() {
        let df = df! {
            "uid" => [0, 0, 0, 1, 1, 1],
            "day" => [1, 2, 3, 1, 2, 3],
            "cumcases" => [10, 12, 15, 25, 30, 41]
        }
        .unwrap();

        let base_df = df.lazy();

        let out = base_df
            .clone()
            .groupby(vec![col("uid")])
            .agg(vec![
                col("day").list().alias("day"),
                col("cumcases")
                    .map(
                        |s: Series| {
                            // determine the diff per column
                            let a: ListChunked = s
                                .list()
                                .unwrap()
                                .into_iter()
                                .map(|opt_s| opt_s.map(|s| &s - &(s.shift(1))))
                                .collect();
                            Ok(a.into_series())
                        },
                        None,
                    )
                    .alias("diff_cases"),
            ])
            .explode(&[col("day"), col("diff_cases")])
            .join(
                base_df,
                vec![col("uid"), col("day")],
                vec![col("uid"), col("day")],
                JoinType::Inner,
            )
            .collect()
            .unwrap();
        assert_eq!(
            Vec::from(out.column("diff_cases").unwrap().i32().unwrap()),
            &[None, Some(2), Some(3), None, Some(5), Some(11)]
        );
    }

    #[test]
    fn test_lazy_query_5() {
        // if this one fails, the list builder probably does not handle offsets
        let df = df! {
            "uid" => [0, 0, 0, 1, 1, 1],
            "day" => [1, 2, 4, 1, 2, 3],
            "cumcases" => [10, 12, 15, 25, 30, 41]
        }
        .unwrap();

        let out = df
            .lazy()
            .groupby(vec![col("uid")])
            .agg(vec![col("day").head(Some(2))])
            .collect()
            .unwrap();
        dbg!(&out);
        let s = out
            .select_at_idx(1)
            .unwrap()
            .list()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(s.len(), 2);
        let s = out
            .select_at_idx(1)
            .unwrap()
            .list()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn test_lazy_query_6() -> Result<()> {
        let df = df! {
            "uid" => [0, 0, 0, 1, 1, 1],
            "day" => [1, 2, 4, 1, 2, 3],
            "cumcases" => [10, 12, 15, 25, 30, 41]
        }
        .unwrap();

        let out = df
            .lazy()
            .groupby(vec![col("uid")])
            // a double aggregation expression.
            .agg(vec![pearson_corr(col("day"), col("cumcases")).pow(2.0)])
            .sort("uid", false)
            .collect()
            .unwrap();
        let s = out.column("pearson_corr")?.f64()?;
        assert!((s.get(0).unwrap() - 0.994360902255639).abs() < 0.000001);
        assert!((s.get(1).unwrap() - 0.9552238805970149).abs() < 0.000001);
        Ok(())
    }

    #[test]
    #[cfg(feature = "is_in")]
    fn test_lazy_query_8() -> Result<()> {
        // https://github.com/pola-rs/polars/issues/842
        let df = df![
            "A" => [1, 2, 3],
            "B" => [1, 2, 3],
            "C" => [1, 2, 3],
            "D" => [1, 2, 3],
            "E" => [1, 2, 3]
        ]?;

        let mut selection = vec![];

        for c in &["A", "B", "C", "D", "E"] {
            let e = when(col(c).is_in(col("E")))
                .then(col("A"))
                .otherwise(Null {}.lit())
                .alias(c);
            selection.push(e);
        }

        let out = df
            .lazy()
            .select(selection)
            .filter(col("D").gt(lit(1)))
            .collect()?;
        assert_eq!(out.shape(), (2, 5));
        Ok(())
    }

    #[test]
    fn test_lazy_query_9() -> Result<()> {
        // https://github.com/pola-rs/polars/issues/958
        let cities = df![
            "Cities.City"=> ["Moscow", "Berlin", "Paris","Hamburg", "Lyon", "Novosibirsk"],
            "Cities.Population"=> [11.92, 3.645, 2.161, 1.841, 0.513, 1.511],
            "Cities.Country"=> ["Russia", "Germany", "France", "Germany", "France", "Russia"]
        ]?;

        let sales = df![
                   "Sales.City"=> ["Moscow", "Berlin", "Paris", "Moscow", "Berlin", "Paris", "Moscow", "Berlin", "Paris"],
        "Sales.Item"=> ["Item A", "Item A","Item A",
                       "Item B", "Item B","Item B",
                       "Item C", "Item C","Item C"],
        "Sales.Amount"=> [200, 180, 100,
                        3, 30, 20,
                        90, 130, 125]
            ]?;

        let out = sales
            .lazy()
            .join(
                cities.lazy(),
                vec![col("Sales.City")],
                vec![col("Cities.City")],
                JoinType::Inner,
            )
            .groupby(vec![col("Cities.Country")])
            .agg(vec![col("Sales.Amount").sum().alias("sum")])
            .sort("sum", false)
            .collect()?;
        let vals = out
            .column("sum")?
            .i32()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
        assert_eq!(vals, &[245, 293, 340]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "temporal")]
    fn test_lazy_query_7() {
        let date = NaiveDate::from_ymd(2021, 3, 5);
        let dates = vec![
            NaiveDateTime::new(date, NaiveTime::from_hms(12, 0, 0)),
            NaiveDateTime::new(date, NaiveTime::from_hms(12, 1, 0)),
            NaiveDateTime::new(date, NaiveTime::from_hms(12, 2, 0)),
            NaiveDateTime::new(date, NaiveTime::from_hms(12, 3, 0)),
            NaiveDateTime::new(date, NaiveTime::from_hms(12, 4, 0)),
            NaiveDateTime::new(date, NaiveTime::from_hms(12, 5, 0)),
        ];
        let data = vec![Some(1.), Some(2.), Some(3.), Some(4.), None, None];
        let df = DataFrame::new(vec![
            Date64Chunked::new_from_naive_datetime("date", &*dates).into(),
            Series::new("data", data),
        ])
        .unwrap();
        // this tests if predicate pushdown not interferes with the shift data.
        let out = df
            .lazy()
            .with_column(col("data").shift(-1).alias("output"))
            .with_column(col("output").shift(2).alias("shifted"))
            .filter(col("date").gt(lit(NaiveDateTime::new(date, NaiveTime::from_hms(12, 2, 0)))))
            .collect()
            .unwrap();
        let a = out.column(&"shifted").unwrap().sum::<f64>().unwrap() - 7.0;
        assert!(a < 0.01 && a > -0.01);
    }

    #[test]
    fn test_lazy_shift_and_fill_all() {
        let data = &[1, 2, 3];
        let df = DataFrame::new(vec![Series::new("data", data)]).unwrap();
        let out = df
            .lazy()
            .with_column(col("data").shift(1).fill_none(lit(0)).alias("output"))
            .collect()
            .unwrap();
        assert_eq!(
            Vec::from(out.column("output").unwrap().i32().unwrap()),
            vec![Some(0), Some(2), Some(3)]
        );
    }

    #[test]
    fn test_lazy_shift_operation_no_filter() {
        // check if predicate pushdown optimization does not fail
        let df = df! {
            "a" => &[1, 2, 3],
            "b" => &[1, 2, 3]
        }
        .unwrap();
        df.lazy()
            .with_column(col("b").shift(1).alias("output"))
            .collect()
            .unwrap();
    }

    #[test]
    fn test_simplify_expr() {
        // Test if expression containing literals is simplified
        let df = get_df();

        let plan = df
            .lazy()
            .select(&[lit(1.0f32) + lit(1.0f32) + col("sepal.width")])
            .logical_plan;

        let mut expr_arena = Arena::new();
        let mut lp_arena = Arena::new();
        let rules: &mut [Box<dyn OptimizationRule>] = &mut [Box::new(SimplifyExprRule {})];

        let optimizer = StackOptimizer {};
        let mut lp_top = to_alp(plan, &mut expr_arena, &mut lp_arena);
        lp_top = optimizer.optimize_loop(rules, &mut expr_arena, &mut lp_arena, lp_top);
        let plan = node_to_lp(lp_top, &mut expr_arena, &mut lp_arena);
        assert!(
            matches!(plan, LogicalPlan::Projection{ expr, ..} if matches!(&expr[0], Expr::BinaryExpr{left, ..} if **left == Expr::Literal(LiteralValue::Float32(2.0))))
        );
    }

    #[test]
    fn test_lazy_wildcard() {
        let df = load_df();
        let new = df.clone().lazy().select(&[col("*")]).collect().unwrap();
        assert_eq!(new.shape(), (5, 3));

        let new = df
            .lazy()
            .groupby(vec![col("b")])
            .agg(vec![col("*").sum(), col("*").first()])
            .collect()
            .unwrap();
        assert_eq!(new.shape(), (3, 6));
    }

    #[test]
    fn test_lazy_reverse() {
        let df = load_df();
        assert!(df
            .clone()
            .lazy()
            .reverse()
            .collect()
            .unwrap()
            .frame_equal_missing(&df.reverse()))
    }

    #[test]
    fn test_lazy_filter_and_rename() {
        let df = load_df();
        let lf = df
            .clone()
            .lazy()
            .with_column_renamed("a", "x")
            .filter(col("x").map(
                |s: Series| Ok(s.gt(3).into_series()),
                Some(DataType::Boolean),
            ))
            .select(&[col("x")]);

        let correct = df! {
            "x" => &[4, 5]
        }
        .unwrap();
        assert!(lf.collect().unwrap().frame_equal(&correct));

        // now we check if the column is rename or added when we don't select
        let lf = df.lazy().with_column_renamed("a", "x").filter(col("x").map(
            |s: Series| Ok(s.gt(3).into_series()),
            Some(DataType::Boolean),
        ));

        assert_eq!(lf.collect().unwrap().get_column_names(), &["x", "b", "c"]);
    }

    #[test]
    fn test_lazy_agg_scan() {
        let lf = scan_foods_csv;
        let df = lf().min().collect().unwrap();
        assert!(df.frame_equal_missing(&lf().collect().unwrap().min()));
        let df = lf().max().collect().unwrap();
        assert!(df.frame_equal_missing(&lf().collect().unwrap().max()));
        // mean is not yet aggregated at scan.
        let df = lf().mean().collect().unwrap();
        assert!(df.frame_equal_missing(&lf().collect().unwrap().mean()));
    }

    #[test]
    fn test_lazy_df_aggregations() {
        let df = load_df();

        assert!(df
            .clone()
            .lazy()
            .min()
            .collect()
            .unwrap()
            .frame_equal_missing(&df.min()));
        assert!(df
            .clone()
            .lazy()
            .median()
            .collect()
            .unwrap()
            .frame_equal_missing(&df.median()));
        assert!(df
            .clone()
            .lazy()
            .quantile(0.5)
            .collect()
            .unwrap()
            .frame_equal_missing(&df.quantile(0.5).unwrap()));
    }

    #[test]
    fn test_lazy_predicate_pushdown_binary_expr() {
        let df = load_df();
        df.lazy()
            .filter(col("a").eq(col("b")))
            .select(&[col("c")])
            .collect()
            .unwrap();
    }

    #[test]
    fn test_lazy_update_column() {
        let df = load_df();
        df.lazy().with_column(col("a") / lit(10)).collect().unwrap();
    }

    #[test]
    fn test_lazy_fill_none() {
        let df = df! {
            "a" => &[None, Some(2)],
            "b" => &[Some(1), None]
        }
        .unwrap();
        let out = df.lazy().fill_none(lit(10.0)).collect().unwrap();
        let correct = df! {
            "a" => &[Some(10.0), Some(2.0)],
            "b" => &[Some(1.0), Some(10.0)]
        }
        .unwrap();
        assert!(out.frame_equal(&correct));
        assert_eq!(out.get_column_names(), vec!["a", "b"])
    }

    #[test]
    fn test_lazy_window_functions() {
        let df = df! {
            "groups" => &[1, 1, 2, 2, 1, 2, 3, 3, 1],
            "values" => &[1, 2, 3, 4, 5, 6, 7, 8, 8]
        }
        .unwrap();

        // sums
        // 1 => 16
        // 2 => 13
        // 3 => 15
        let correct = [16, 16, 13, 13, 16, 13, 15, 15, 16]
            .iter()
            .copied()
            .map(Some)
            .collect::<Vec<_>>();

        // test if groups is available after projection pushdown.
        let _ = df
            .clone()
            .lazy()
            .select(&[avg("values").over(vec![col("groups")]).alias("part")])
            .collect()
            .unwrap();
        // test if partition aggregation is correct
        let out = df
            .lazy()
            .select(&[col("groups"), sum("values").over(vec![col("groups")])])
            .collect()
            .unwrap();
        assert_eq!(
            Vec::from(out.select_at_idx(1).unwrap().i32().unwrap()),
            correct
        );
        dbg!(out);
    }

    #[test]
    fn test_lazy_double_projection() {
        let df = df! {
            "foo" => &[1, 2, 3]
        }
        .unwrap();
        df.lazy()
            .select(&[col("foo").alias("bar")])
            .select(&[col("bar")])
            .collect()
            .unwrap();
    }

    #[test]
    fn test_type_coercion() {
        let df = df! {
            "foo" => &[1, 2, 3],
            "bar" => &[1.0, 2.0, 3.0]
        }
        .unwrap();

        let lp = df.lazy().select(&[col("foo") * col("bar")]).logical_plan;

        let mut expr_arena = Arena::new();
        let mut lp_arena = Arena::new();
        let rules: &mut [Box<dyn OptimizationRule>] = &mut [Box::new(TypeCoercionRule {})];

        let optimizer = StackOptimizer {};
        let mut lp_top = to_alp(lp, &mut expr_arena, &mut lp_arena);
        lp_top = optimizer.optimize_loop(rules, &mut expr_arena, &mut lp_arena, lp_top);
        let lp = node_to_lp(lp_top, &mut expr_arena, &mut lp_arena);

        if let LogicalPlan::Projection { expr, .. } = lp {
            if let Expr::BinaryExpr { left, right, .. } = &expr[0] {
                assert!(matches!(&**left, Expr::Cast { .. }));
                assert!(matches!(&**right, Expr::Cast { .. }));
            } else {
                panic!()
            }
        };
    }

    #[test]
    fn test_lazy_partition_agg() {
        let df = df! {
            "foo" => &[1, 1, 2, 2, 3],
            "bar" => &[1.0, 1.0, 2.0, 2.0, 3.0]
        }
        .unwrap();

        let out = df
            .lazy()
            .groupby(vec![col("foo")])
            .agg(vec![col("bar").mean()])
            .sort("foo", false)
            .collect()
            .unwrap();

        assert_eq!(
            Vec::from(out.column("bar_mean").unwrap().f64().unwrap()),
            &[Some(1.0), Some(2.0), Some(3.0)]
        );

        let out = scan_foods_csv()
            .groupby(vec![col("category")])
            .agg(vec![col("calories").list()])
            .sort("category", false)
            .collect()
            .unwrap();
        dbg!(&out);
        let cat_agg_list = out.select_at_idx(1).unwrap();
        let fruit_series = cat_agg_list.list().unwrap().get(0).unwrap();
        let fruit_list = fruit_series.i64().unwrap();
        dbg!(fruit_list);
        assert_eq!(
            Vec::from(fruit_list),
            &[
                Some(60),
                Some(30),
                Some(50),
                Some(30),
                Some(60),
                Some(130),
                Some(50),
            ]
        )
    }

    #[test]
    fn test_select_except() {
        let df = df! {
            "foo" => &[1, 1, 2, 2, 3],
            "bar" => &[1.0, 1.0, 2.0, 2.0, 3.0],
            "ham" => &[1.0, 1.0, 2.0, 2.0, 3.0]
        }
        .unwrap();

        let out = df
            .lazy()
            .select(&[col("*"), except("foo")])
            .collect()
            .unwrap();

        assert_eq!(out.get_column_names(), &["ham", "bar"]);
    }

    #[test]
    fn test_lazy_groupby_apply() {
        let df = df! {
            "A" => &[1, 2, 3, 4, 5],
            "fruits" => &["banana", "banana", "apple", "apple", "banana"],
            "B" => &[5, 4, 3, 2, 1],
            "cars" => &["beetle", "audi", "beetle", "beetle", "beetle"]
        }
        .unwrap();

        df.lazy()
            .groupby(vec![col("fruits")])
            .agg(vec![col("cars").map(
                |s: Series| {
                    let ca: UInt32Chunked = s
                        .list()?
                        .into_iter()
                        .map(|opt_s| opt_s.map(|s| s.len() as u32))
                        .collect();
                    Ok(ca.into_series())
                },
                None,
            )])
            .collect()
            .unwrap();
    }

    #[test]
    fn test_lazy_shift_and_fill() {
        let df = df! {
            "A" => &[1, 2, 3, 4, 5],
            "B" => &[5, 4, 3, 2, 1]
        }
        .unwrap();
        let out = df
            .clone()
            .lazy()
            .with_column(col("A").shift_and_fill(2, col("B").mean()))
            .collect()
            .unwrap();
        assert_eq!(out.column("A").unwrap().null_count(), 0);

        // shift from the other side
        let out = df
            .clone()
            .lazy()
            .with_column(col("A").shift_and_fill(-2, col("B").mean()))
            .collect()
            .unwrap();
        assert_eq!(out.column("A").unwrap().null_count(), 0);

        let out = df
            .clone()
            .lazy()
            .shift_and_fill(-1, col("B").std())
            .collect()
            .unwrap();
        assert_eq!(out.column("A").unwrap().null_count(), 0);
    }

    #[test]
    fn test_lazy_groupby() {
        let df = df! {
            "a" => &[Some(1.0), None, Some(3.0), Some(4.0), Some(5.0)],
            "groups" => &["a", "a", "b", "c", "c"]
        }
        .unwrap();

        let out = df
            .lazy()
            .groupby(vec![col("groups")])
            .agg(vec![col("a").mean()])
            .sort("a_mean", false)
            .collect()
            .unwrap();

        assert_eq!(
            out.column("a_mean").unwrap().f64().unwrap().get(0),
            Some(1.0)
        );
    }

    #[test]
    fn test_lazy_tail() {
        let df = df! {
            "A" => &[1, 2, 3, 4, 5],
            "B" => &[5, 4, 3, 2, 1]
        }
        .unwrap();

        let _out = df.clone().lazy().tail(3).collect().unwrap();
    }

    #[test]
    fn test_lazy_groupby_sort() {
        let df = df! {
            "a" => ["a", "b", "a", "b", "b", "c"],
            "b" => [1, 2, 3, 4, 5, 6]
        }
        .unwrap();

        let out = df
            .clone()
            .lazy()
            .groupby(vec![col("a")])
            .agg(vec![col("b").sort(false).first()])
            .collect()
            .unwrap()
            .sort("a", false)
            .unwrap();

        assert_eq!(
            Vec::from(out.column("b_first").unwrap().i32().unwrap()),
            [Some(1), Some(2), Some(6)]
        );

        let out = df
            .lazy()
            .groupby(vec![col("a")])
            .agg(vec![col("b").sort(false).last()])
            .collect()
            .unwrap()
            .sort("a", false)
            .unwrap();

        assert_eq!(
            Vec::from(out.column("b_last").unwrap().i32().unwrap()),
            [Some(3), Some(5), Some(6)]
        );
    }

    #[test]
    fn test_lazy_groupby_sort_by() {
        let df = df! {
            "a" => ["a", "a", "a", "b", "b", "c"],
            "b" => [1, 2, 3, 4, 5, 6],
            "c" => [6, 1, 4, 3, 2, 1]
        }
        .unwrap();

        let out = df
            .lazy()
            .groupby(vec![col("a")])
            .agg(vec![col("b").sort_by(col("c"), true).first()])
            .collect()
            .unwrap()
            .sort("a", false)
            .unwrap();

        assert_eq!(
            Vec::from(out.column("b_first").unwrap().i32().unwrap()),
            [Some(1), Some(4), Some(6)]
        );
    }

    #[test]
    #[cfg(feature = "dtype-date64")]
    fn test_lazy_groupby_cast() {
        let df = df! {
            "a" => ["a", "a", "a", "b", "b", "c"],
            "b" => [1, 2, 3, 4, 5, 6]
        }
        .unwrap();

        // test if it runs in groupby context
        let _out = df
            .lazy()
            .groupby(vec![col("a")])
            .agg(vec![col("b").mean().cast(DataType::Date64)])
            .collect()
            .unwrap();
    }

    #[test]
    fn test_lazy_groupby_binary_expr() {
        let df = df! {
            "a" => ["a", "a", "a", "b", "b", "c"],
            "b" => [1, 2, 3, 4, 5, 6]
        }
        .unwrap();

        // test if it runs in groupby context
        let out = df
            .lazy()
            .groupby(vec![col("a")])
            .agg(vec![col("b").mean() * lit(2)])
            .sort("a", false)
            .collect()
            .unwrap();
        assert_eq!(
            Vec::from(out.column("b_mean").unwrap().f64().unwrap()),
            [Some(4.0), Some(9.0), Some(12.0)]
        );
    }

    #[test]
    fn test_lazy_groupby_filter() -> Result<()> {
        let df = df! {
            "a" => ["a", "a", "a", "b", "b", "c"],
            "b" => [1, 2, 3, 4, 5, 6]
        }?;

        // We test if the filters work in the groupby context
        // and that the aggregations can deal with empty sets

        let out = df
            .lazy()
            .groupby(vec![col("a")])
            .agg(vec![
                col("b").filter(col("a").eq(lit("a"))).sum(),
                col("b").filter(col("a").eq(lit("a"))).first(),
                col("b").filter(col("a").eq(lit("e"))).mean(),
                col("b").filter(col("a").eq(lit("a"))).last(),
            ])
            .sort("a", false)
            .collect()?;

        assert_eq!(
            Vec::from(out.column("b_sum").unwrap().i32().unwrap()),
            [Some(6), None, None]
        );
        assert_eq!(
            Vec::from(out.column("b_first").unwrap().i32().unwrap()),
            [Some(1), None, None]
        );
        assert_eq!(
            Vec::from(out.column("b_mean").unwrap().f64().unwrap()),
            [None, None, None]
        );
        assert_eq!(
            Vec::from(out.column("b_last").unwrap().i32().unwrap()),
            [Some(3), None, None]
        );

        Ok(())
    }

    #[test]
    fn test_groupby_projection_pd_same_column() -> Result<()> {
        // this query failed when projection pushdown was enabled

        let a = || {
            let df = df![
                "col1" => ["a", "ab", "abc"],
                "col2" => [1, 2, 3]
            ]
            .unwrap();

            df.lazy()
                .select(vec![col("col1").alias("foo"), col("col2").alias("bar")])
        };

        let out = a()
            .left_join(a(), col("foo"), col("foo"))
            .select(vec![col("bar")])
            .collect()?;

        let a = out.column("bar")?.i32()?;
        assert_eq!(Vec::from(a), &[Some(1), Some(2), Some(3)]);

        Ok(())
    }

    #[test]
    fn test_groupby_sort_slice() -> Result<()> {
        let df = df![
            "groups" => [1, 2, 2, 3, 3, 3],
            "vals" => [1, 5, 6, 3, 9, 8]
        ]?;
        // get largest two values per groups

        // expected:
        // group      values
        // 1          1
        // 2          6, 5
        // 3          9, 8

        let out1 = df
            .clone()
            .lazy()
            .sort("vals", true)
            .groupby(vec![col("groups")])
            .agg(vec![col("vals").head(Some(2)).alias("foo")])
            .sort("groups", false)
            .collect()?;

        let out2 = df
            .lazy()
            .groupby(vec![col("groups")])
            .agg(vec![col("vals").sort(true).head(Some(2)).alias("foo")])
            .sort("groups", false)
            .collect()?;

        assert!(out1.column("foo")?.series_equal(out2.column("foo")?));
        dbg!(out1, out2);
        Ok(())
    }

    #[test]
    fn test_groupby_cumsum() -> Result<()> {
        let df = df![
            "groups" => [1, 2, 2, 3, 3, 3],
            "vals" => [1, 5, 6, 3, 9, 8]
        ]?;

        let out = df
            .lazy()
            .groupby(vec![col("groups")])
            .agg(vec![col("vals").cum_sum(false)])
            .sort("groups", false)
            .collect()?;

        assert_eq!(
            Vec::from(out.column("vals")?.explode()?.i32()?),
            [1, 5, 11, 3, 12, 20]
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<_>>()
        );

        Ok(())
    }

    #[test]
    fn test_argsort_multiple() -> Result<()> {
        let df = df![
            "int" => [1, 2, 3, 1, 2],
            "flt" => [3.0, 2.0, 1.0, 2.0, 1.0],
            "str" => ["a", "a", "a", "b", "b"]
        ]?;

        let out = df
            .clone()
            .lazy()
            .select(vec![argsort_by(
                vec![col("int"), col("flt")],
                &[true, false],
            )])
            .collect()?;

        assert_eq!(
            Vec::from(out.column("int")?.u32()?),
            [2, 4, 1, 3, 0]
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<_>>()
        );

        // check if this runs
        let _out = df
            .lazy()
            .select(vec![argsort_by(
                vec![col("str"), col("flt")],
                &[true, false],
            )])
            .collect()?;
        Ok(())
    }

    #[test]
    fn test_multiple_explode() -> Result<()> {
        let df = df![
            "a" => [0, 1, 2, 0, 2],
            "b" => [5, 4, 3, 2, 1],
            "c" => [2, 3, 4, 1, 5]
        ]?;

        let out = df
            .lazy()
            .groupby(vec![col("a")])
            .agg(vec![
                col("b").list().alias("b_list"),
                col("c").list().alias("c_list"),
            ])
            .explode(&[col("c_list"), col("b_list")])
            .collect()?;
        assert_eq!(out.shape(), (5, 3));

        Ok(())
    }

    #[test]
    fn test_filter_and_alias() -> Result<()> {
        let df = df![
            "a" => [0, 1, 2, 0, 2]
        ]?;

        let out = df
            .lazy()
            .with_column(col("a").pow(2.0).alias("a_squared"))
            .filter(col("a_squared").gt(lit(1)).and(col("a").gt(lit(1))))
            .collect()?;

        let expected = df![
            "a" => [2, 2],
            "a_squared" => [4, 4]
        ]?;

        assert!(out.frame_equal(&expected));
        Ok(())
    }

    #[test]
    fn test_filter_lit() {
        // see https://github.com/pola-rs/polars/issues/790
        // failed due to broadcasting filters and splitting threads.
        let iter = (0..100).map(|i| ('A'..='Z').nth(i % 26).unwrap().to_string());
        let a = Series::from_iter(iter);
        let df = DataFrame::new([a].into()).unwrap();

        let out = df.lazy().filter(lit(true)).collect().unwrap();
        assert_eq!(out.shape(), (100, 1));
    }

    #[test]
    fn test_ternary_null() -> Result<()> {
        let df = df![
            "a" => ["a", "b", "c"]
        ]?;

        let out = df
            .lazy()
            .select(vec![when(col("a").eq(lit("c")))
                .then(Null {}.lit())
                .otherwise(col("a"))
                .alias("foo")])
            .collect()?;

        assert_eq!(
            out.column("foo")?.is_null().into_iter().collect::<Vec<_>>(),
            &[Some(false), Some(false), Some(true)]
        );
        Ok(())
    }

    #[test]
    fn test_fill_forward() -> Result<()> {
        let df = df![
            "a" => ["a", "b", "a"],
            "b" => [Some(1), None, None]
        ]?;

        let out = df
            .lazy()
            .select(vec![col("b").forward_fill().over(vec![col("a")])])
            .collect()?;
        let agg = out.column("b")?.list()?;

        let a: Series = agg.get(0).unwrap();
        assert!(a.series_equal(&Series::new("b", &[1, 1])));
        let a: Series = agg.get(2).unwrap();
        assert!(a.series_equal(&Series::new("b", &[1, 1])));
        let a: Series = agg.get(1).unwrap();
        assert_eq!(a.null_count(), 1);
        Ok(())
    }

    #[cfg(feature = "cross_join")]
    #[test]
    fn test_cross_join() -> Result<()> {
        let df1 = df![
            "a" => ["a", "b", "a"],
            "b" => [Some(1), None, None]
        ]?;

        let df2 = df![
            "a" => [1, 2],
            "b" => [None, Some(12)]
        ]?;

        let out = df1.lazy().cross_join(df2.lazy()).collect()?;
        assert_eq!(out.shape(), (6, 4));
        Ok(())
    }
}
