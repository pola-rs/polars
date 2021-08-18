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
use crate::prelude::fast_projection::FastProjection;
use crate::prelude::simplify_expr::SimplifyBooleanRule;
use crate::utils::{combine_predicates_expr, expr_to_root_column_names};
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
    /// Get a hold on the schema of the current LazyFrame computation.
    pub fn schema(&self) -> SchemaRef {
        let logical_plan = self.clone().get_plan_builder().build();
        logical_plan.schema().clone()
    }

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

    /// Fill NaN values in the DataFrame
    pub fn fill_nan(self, fill_value: Expr) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().fill_nan(fill_value).build();
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

        rules.push(Box::new(FastProjection {}));

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
    ///         .select(&[col("*").exclude("foo")])
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
    pub fn explode(self, columns: Vec<Expr>) -> LazyFrame {
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

    /// Return first n rows of each group
    pub fn head(self, n: Option<usize>) -> LazyFrame {
        let keys = self
            .keys
            .iter()
            .map(|k| expr_to_root_column_names(k).into_iter())
            .flatten()
            .collect::<Vec<_>>();

        self.agg(vec![col("*").exclude(&keys).head(n).list().keep_name()])
            .explode(vec![col("*").exclude(&keys)])
    }

    /// Return last n rows of each group
    pub fn tail(self, n: Option<usize>) -> LazyFrame {
        let keys = self
            .keys
            .iter()
            .map(|k| expr_to_root_column_names(k).into_iter())
            .flatten()
            .collect::<Vec<_>>();

        self.agg(vec![col("*").exclude(&keys).tail(n).list().keep_name()])
            .explode(vec![col("*").exclude(&keys)])
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
