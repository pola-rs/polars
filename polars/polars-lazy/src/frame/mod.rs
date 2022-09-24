//! Lazy variant of a [DataFrame](polars_core::frame::DataFrame).
#[cfg(feature = "csv-file")]
mod csv;
#[cfg(feature = "ipc")]
mod ipc;
#[cfg(feature = "json")]
mod ndjson;
#[cfg(feature = "parquet")]
mod parquet;
#[cfg(feature = "python")]
mod python;

mod anonymous_scan;
#[cfg(feature = "pivot")]
pub mod pivot;

use std::borrow::Cow;
use std::sync::Arc;

pub use anonymous_scan::*;
#[cfg(feature = "csv-file")]
pub use csv::*;
#[cfg(feature = "ipc")]
pub use ipc::*;
#[cfg(feature = "json")]
pub use ndjson::*;
#[cfg(feature = "parquet")]
pub use parquet::*;
use polars_arrow::prelude::QuantileInterpolOptions;
#[cfg(any(feature = "parquet", feature = "csv-file", feature = "ipc"))]
use polars_core::datatypes::PlHashMap;
use polars_core::frame::explode::MeltArgs;
use polars_core::frame::hash_join::JoinType;
use polars_core::prelude::*;
use polars_io::RowCount;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::logical_plan::optimizer::aggregate_pushdown::AggregatePushdown;
#[cfg(any(
    feature = "parquet",
    feature = "csv-file",
    feature = "ipc",
    feature = "cse"
))]
use crate::logical_plan::optimizer::file_caching::FileCacher;
use crate::logical_plan::optimizer::predicate_pushdown::PredicatePushDown;
use crate::logical_plan::optimizer::projection_pushdown::ProjectionPushDown;
use crate::logical_plan::optimizer::simplify_expr::SimplifyExprRule;
use crate::logical_plan::optimizer::stack_opt::{OptimizationRule, StackOptimizer};
use crate::logical_plan::FETCH_ROWS;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::delay_rechunk::DelayRechunk;
use crate::prelude::drop_nulls::ReplaceDropNulls;
use crate::prelude::fast_projection::FastProjectionAndCollapse;
#[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
use crate::prelude::file_caching::collect_fingerprints;
#[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
use crate::prelude::file_caching::find_column_union_and_fingerprints;
use crate::prelude::simplify_expr::SimplifyBooleanRule;
use crate::prelude::slice_pushdown_lp::SlicePushDown;
use crate::prelude::*;
use crate::utils::{combine_predicates_expr, expr_to_leaf_column_names};

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct JoinOptions {
    pub allow_parallel: bool,
    pub force_parallel: bool,
    pub how: JoinType,
    pub suffix: Cow<'static, str>,
    pub slice: Option<(i64, usize)>,
}

impl Default for JoinOptions {
    fn default() -> Self {
        JoinOptions {
            allow_parallel: true,
            force_parallel: false,
            how: JoinType::Left,
            suffix: "_right".into(),
            slice: None,
        }
    }
}

pub trait IntoLazy {
    fn lazy(self) -> LazyFrame;
}

impl IntoLazy for DataFrame {
    /// Convert the `DataFrame` into a lazy `DataFrame`
    fn lazy(self) -> LazyFrame {
        let lp = LogicalPlanBuilder::from_existing_df(self).build();
        LazyFrame {
            logical_plan: lp,
            opt_state: Default::default(),
        }
    }
}

/// Lazy abstraction over an eager `DataFrame`.
/// It really is an abstraction over a logical plan. The methods of this struct will incrementally
/// modify a logical plan until output is requested (via [collect](crate::frame::LazyFrame::collect))
#[derive(Clone, Default)]
#[must_use]
pub struct LazyFrame {
    pub logical_plan: LogicalPlan,
    pub(crate) opt_state: OptState,
}

impl From<LogicalPlan> for LazyFrame {
    fn from(plan: LogicalPlan) -> Self {
        Self {
            logical_plan: plan,
            opt_state: OptState {
                file_caching: true,
                ..Default::default()
            },
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
    pub file_caching: bool,
    pub aggregate_pushdown: bool,
    pub slice_pushdown: bool,
    #[cfg(feature = "cse")]
    pub common_subplan_elimination: bool,
}

impl Default for OptState {
    fn default() -> Self {
        OptState {
            projection_pushdown: true,
            predicate_pushdown: true,
            type_coercion: true,
            simplify_expr: true,
            slice_pushdown: true,
            // will be toggled by a scan operation such as csv scan or parquet scan
            file_caching: false,
            aggregate_pushdown: false,
            #[cfg(feature = "cse")]
            common_subplan_elimination: true,
        }
    }
}

/// AllowedOptimizations
pub type AllowedOptimizations = OptState;

impl LazyFrame {
    /// Get a hold on the schema of the current LazyFrame computation.
    pub fn schema(&self) -> PolarsResult<SchemaRef> {
        let logical_plan = self.clone().get_plan_builder().build();
        logical_plan.schema().map(|schema| schema.into_owned())
    }

    pub(crate) fn get_plan_builder(self) -> LogicalPlanBuilder {
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

    /// Set allowed optimizations
    pub fn with_optimizations(mut self, opt_state: OptState) -> Self {
        self.opt_state = opt_state;
        self
    }

    /// Turn off all optimizations
    pub fn without_optimizations(self) -> Self {
        self.with_optimizations(OptState {
            projection_pushdown: false,
            predicate_pushdown: false,
            type_coercion: true,
            simplify_expr: false,
            slice_pushdown: false,
            // will be toggled by a scan operation such as csv scan or parquet scan
            file_caching: false,
            aggregate_pushdown: false,
            #[cfg(feature = "cse")]
            common_subplan_elimination: false,
        })
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

    /// Toggle common subplan elimination optimization on or off
    #[cfg(feature = "cse")]
    #[cfg_attr(docsrs, doc(cfg(feature = "cse")))]
    pub fn with_common_subplan_elimination(mut self, toggle: bool) -> Self {
        self.opt_state.common_subplan_elimination = toggle;
        self
    }

    /// Toggle aggregate pushdown.
    pub fn with_aggregate_pushdown(mut self, toggle: bool) -> Self {
        self.opt_state.aggregate_pushdown = toggle;
        self
    }

    /// Toggle slice pushdown optimization
    pub fn with_slice_pushdown(mut self, toggle: bool) -> Self {
        self.opt_state.slice_pushdown = toggle;
        self
    }

    /// Describe the logical plan.
    pub fn describe_plan(&self) -> String {
        self.logical_plan.describe()
    }

    /// Describe the optimized logical plan.
    pub fn describe_optimized_plan(&self) -> PolarsResult<String> {
        let mut expr_arena = Arena::with_capacity(64);
        let mut lp_arena = Arena::with_capacity(64);
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
    ///         .sort("sepal.width", Default::default())
    /// }
    /// ```
    pub fn sort(self, by_column: &str, options: SortOptions) -> Self {
        let reverse = options.descending;
        let nulls_last = options.nulls_last;

        let opt_state = self.get_opt_state();
        let lp = self
            .get_plan_builder()
            .sort(vec![col(by_column)], vec![reverse], nulls_last)
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
    ///         .sort_by_exprs(vec![col("sepal.width")], vec![false], false)
    /// }
    /// ```
    pub fn sort_by_exprs<E: AsRef<[Expr]>, B: AsRef<[bool]>>(
        self,
        by_exprs: E,
        reverse: B,
        nulls_last: bool,
    ) -> Self {
        let by_exprs = by_exprs.as_ref().to_vec();
        let reverse = reverse.as_ref().to_vec();
        if by_exprs.is_empty() {
            self
        } else {
            let opt_state = self.get_opt_state();
            let lp = self
                .get_plan_builder()
                .sort(by_exprs, reverse, nulls_last)
                .build();
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

    fn rename_impl_swapping(self, mut existing: Vec<String>, mut new: Vec<String>) -> Self {
        assert_eq!(new.len(), existing.len());
        let mut removed = 0;
        for mut idx in 0..existing.len() {
            // remove "name" -> "name
            // these are no ops.
            idx -= removed;
            if existing[idx] == new[idx] {
                existing.swap_remove(idx);
                new.swap_remove(idx);
                removed += 1;
            }
        }

        let existing2 = existing.clone();
        let new2 = new.clone();
        let udf_schema = move |s: &Schema| {
            // schema after renaming
            let mut new_schema = s.clone();
            for (old, new) in existing2.iter().zip(new2.iter()) {
                new_schema
                    .rename(old, new.to_string())
                    .ok_or_else(|| PolarsError::NotFound(old.to_string().into()))?
            }
            Ok(Arc::new(new_schema))
        };

        let prefix = "__POLARS_TEMP_";

        let new: Vec<String> = new
            .iter()
            .map(|name| format!("{}{}", prefix, name))
            .collect();

        self.with_columns(
            existing
                .iter()
                .zip(&new)
                .map(|(old, new)| col(old).alias(new))
                .collect::<Vec<_>>(),
        )
        .map(
            move |mut df: DataFrame| {
                let mut cols = std::mem::take(df.get_columns_mut());
                // we must find the indices before we start swapping,
                // because swapping may influence the positions we find if columns are swapped for instance.
                // e.g. a -> b
                //      b -> a
                #[allow(clippy::needless_collect)]
                let existing_idx = existing
                    .iter()
                    .map(|name| cols.iter().position(|s| s.name() == name.as_str()).unwrap())
                    .collect::<Vec<_>>();
                let new_idx = new
                    .iter()
                    .map(|name| cols.iter().position(|s| s.name() == name.as_str()).unwrap())
                    .collect::<Vec<_>>();

                for (existing_i, new_i) in existing_idx.into_iter().zip(new_idx) {
                    cols.swap(existing_i, new_i);
                    let s = &mut cols[existing_i];
                    let name = &s.name()[prefix.len()..].to_string();
                    s.rename(name);
                }
                cols.truncate(cols.len() - existing.len());
                DataFrame::new(cols)
            },
            None,
            Some(Arc::new(udf_schema)),
            Some("RENAME_SWAPPING"),
        )
    }

    fn rename_impl(self, existing: Vec<String>, new: Vec<String>) -> Self {
        let existing2 = existing.clone();
        let new2 = new.clone();
        let udf_schema = move |s: &Schema| {
            let mut new_schema = s.clone();
            for (old, new) in existing2.iter().zip(&new2) {
                let _ = new_schema.rename(old, new.clone());
            }
            Ok(Arc::new(new_schema))
        };

        self.with_columns(
            existing
                .iter()
                .zip(&new)
                .map(|(old, new)| col(old).alias(new.as_ref()))
                .collect::<Vec<_>>(),
        )
        .map(
            move |mut df: DataFrame| {
                let cols = df.get_columns_mut();
                for (existing, new) in existing.iter().zip(new.iter()) {
                    let idx_a = cols
                        .iter()
                        .position(|s| s.name() == existing.as_str())
                        .unwrap();
                    let idx_b = cols.iter().position(|s| s.name() == new.as_str()).unwrap();
                    cols.swap(idx_a, idx_b);
                }
                cols.truncate(cols.len() - existing.len());
                Ok(df)
            },
            None,
            Some(Arc::new(udf_schema)),
            Some("RENAME"),
        )
    }

    /// Rename columns in the DataFrame.
    pub fn rename<I, J, T, S>(self, existing: I, new: J) -> Self
    where
        I: IntoIterator<Item = T>,
        J: IntoIterator<Item = S>,
        T: AsRef<str>,
        S: AsRef<str>,
    {
        // We dispatch to 2 implementations.
        // 1 is swapping eg. rename a -> b and b -> a
        // 2 is non-swapping eg. rename a -> new_name
        // the latter allows predicate pushdown.
        let existing = existing
            .into_iter()
            .map(|a| a.as_ref().to_string())
            .collect::<Vec<_>>();
        let new = new
            .into_iter()
            .map(|a| a.as_ref().to_string())
            .collect::<Vec<_>>();

        fn inner(lf: LazyFrame, existing: Vec<String>, new: Vec<String>) -> LazyFrame {
            // remove mappings that map to themselves.
            let (existing, new): (Vec<_>, Vec<_>) = existing
                .into_iter()
                .zip(new)
                .flat_map(|(a, b)| if a == b { None } else { Some((a, b)) })
                .unzip();

            // todo! make delayed
            let schema = &*lf.schema().unwrap();
            // a column gets swapped
            if new.iter().any(|name| schema.get(name).is_some()) {
                lf.rename_impl_swapping(existing, new)
            } else {
                lf.rename_impl(existing, new)
            }
        }

        inner(self, existing, new)
    }

    /// Removes columns from the DataFrame.
    /// Note that its better to only select the columns you need
    /// and let the projection pushdown optimize away the unneeded columns.
    pub fn drop_columns<I, T>(self, columns: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: AsRef<str>,
    {
        let columns: Vec<String> = columns
            .into_iter()
            .map(|name| name.as_ref().to_string())
            .collect();
        self.drop_columns_impl(&columns)
    }

    #[allow(clippy::ptr_arg)]
    fn drop_columns_impl(self, columns: &Vec<String>) -> Self {
        self.select_local(vec![col("*").exclude(columns)])
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
    pub fn shift_and_fill<E: Into<Expr>>(self, periods: i64, fill_value: E) -> Self {
        self.select_local(vec![col("*").shift_and_fill(periods, fill_value.into())])
    }

    /// Fill none values in the DataFrame
    pub fn fill_null<E: Into<Expr>>(self, fill_value: E) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().fill_null(fill_value.into()).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Fill NaN values in the DataFrame
    pub fn fill_nan<E: Into<Expr>>(self, fill_value: E) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().fill_nan(fill_value.into()).build();
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
    pub fn fetch(self, n_rows: usize) -> PolarsResult<DataFrame> {
        FETCH_ROWS.with(|fetch_rows| fetch_rows.set(Some(n_rows)));
        let res = self.collect();
        FETCH_ROWS.with(|fetch_rows| fetch_rows.set(None));
        res
    }

    pub fn optimize(
        self,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<Node> {
        // get toggle values
        let predicate_pushdown = self.opt_state.predicate_pushdown;
        let projection_pushdown = self.opt_state.projection_pushdown;
        let type_coercion = self.opt_state.type_coercion;
        let simplify_expr = self.opt_state.simplify_expr;
        let slice_pushdown = self.opt_state.slice_pushdown;
        #[cfg(feature = "cse")]
        let cse = self.opt_state.common_subplan_elimination;

        let agg_scan_projection = self.opt_state.file_caching;
        let aggregate_pushdown = self.opt_state.aggregate_pushdown;

        let logical_plan = self.get_plan_builder().build();
        let mut scratch = vec![];

        // gradually fill the rules passed to the optimizer
        let opt = StackOptimizer {};
        let mut rules: Vec<Box<dyn OptimizationRule>> = Vec::with_capacity(8);

        // during debug we check if the optimizations have not modified the final schema
        #[cfg(debug_assertions)]
        let prev_schema = logical_plan.schema()?.into_owned();

        let mut lp_top = to_alp(logical_plan, expr_arena, lp_arena)?;

        #[cfg(feature = "cse")]
        let cse_changed = if cse {
            let (lp, changed) = cse::elim_cmn_subplans(lp_top, lp_arena, expr_arena);
            lp_top = lp;
            changed
        } else {
            false
        };
        #[cfg(not(feature = "cse"))]
        let cse_changed = false;

        // we do simplification
        if simplify_expr {
            rules.push(Box::new(SimplifyExprRule {}));
        }

        // should be run before predicate pushdown
        if projection_pushdown {
            let projection_pushdown_opt = ProjectionPushDown {};
            let alp = lp_arena.take(lp_top);
            let alp = projection_pushdown_opt.optimize(alp, lp_arena, expr_arena)?;
            lp_arena.replace(lp_top, alp);
            cache_states::set_cache_states(lp_top, lp_arena, expr_arena, &mut scratch, cse_changed);
        }

        if predicate_pushdown {
            let predicate_pushdown_opt = PredicatePushDown::default();
            let alp = lp_arena.take(lp_top);
            let alp = predicate_pushdown_opt.optimize(alp, lp_arena, expr_arena)?;
            lp_arena.replace(lp_top, alp);
        }

        // make sure its before slice pushdown.
        if projection_pushdown {
            rules.push(Box::new(FastProjectionAndCollapse {}));
        }
        rules.push(Box::new(DelayRechunk {}));

        if slice_pushdown {
            let slice_pushdown_opt = SlicePushDown {};
            let alp = lp_arena.take(lp_top);
            let alp = slice_pushdown_opt.optimize(alp, lp_arena, expr_arena)?;

            lp_arena.replace(lp_top, alp);

            // expressions use the stack optimizer
            rules.push(Box::new(slice_pushdown_opt));
        }
        if type_coercion {
            rules.push(Box::new(TypeCoercionRule {}))
        }
        // this optimization removes branches, so we must do it when type coercion
        // is completed
        if simplify_expr {
            rules.push(Box::new(SimplifyBooleanRule {}));
        }

        if aggregate_pushdown {
            rules.push(Box::new(AggregatePushdown::new()))
        }

        // make sure that we do that once slice pushdown
        // and predicate pushdown are done. At that moment
        // the file fingerprints are finished.
        #[cfg(any(
            feature = "cse",
            feature = "parquet",
            feature = "ipc",
            feature = "csv-file"
        ))]
        if agg_scan_projection || cse_changed {
            // we do this so that expressions are simplified created by the pushdown optimizations
            // we must clean up the predicates, because the agg_scan_projection
            // uses them in the hashtable to determine duplicates.
            let simplify_bools =
                &mut [Box::new(SimplifyBooleanRule {}) as Box<dyn OptimizationRule>];
            lp_top = opt.optimize_loop(simplify_bools, expr_arena, lp_arena, lp_top);

            // scan the LP to aggregate all the column used in scans
            // these columns will be added to the state of the AggScanProjection rule
            let mut file_predicate_to_columns_and_count = PlHashMap::with_capacity(32);
            find_column_union_and_fingerprints(
                lp_top,
                &mut file_predicate_to_columns_and_count,
                lp_arena,
                expr_arena,
            );

            let mut file_cacher = FileCacher::new(file_predicate_to_columns_and_count);
            file_cacher.assign_unions(lp_top, lp_arena, expr_arena, &mut scratch, false);
            scratch.clear();

            #[cfg(feature = "cse")]
            if cse_changed {
                // this must run after cse
                cse::decrement_file_counters_by_cache_hits(
                    lp_top,
                    lp_arena,
                    expr_arena,
                    0,
                    &mut scratch,
                );
            }
        }

        rules.push(Box::new(ReplaceDropNulls {}));

        lp_top = opt.optimize_loop(&mut rules, expr_arena, lp_arena, lp_top);

        // during debug we check if the optimizations have not modified the final schema
        #[cfg(debug_assertions)]
        {
            // only check by names because we may supercast types.
            assert_eq!(
                prev_schema.iter_names().collect::<Vec<_>>(),
                lp_arena
                    .get(lp_top)
                    .schema(lp_arena)
                    .iter_names()
                    .collect::<Vec<_>>()
            );
        };

        Ok(lp_top)
    }

    fn prepare_collect(self) -> PolarsResult<(ExecutionState, Box<dyn Executor>)> {
        let file_caching = self.opt_state.file_caching;
        let mut expr_arena = Arena::with_capacity(256);
        let mut lp_arena = Arena::with_capacity(128);
        let lp_top = self.optimize(&mut lp_arena, &mut expr_arena)?;

        let finger_prints = if file_caching {
            #[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
            {
                let mut fps = Vec::with_capacity(8);
                collect_fingerprints(lp_top, &mut fps, &lp_arena, &expr_arena);
                Some(fps)
            }
            #[cfg(not(any(feature = "ipc", feature = "parquet", feature = "csv-file")))]
            {
                None
            }
        } else {
            None
        };

        let planner = PhysicalPlanner::default();
        let physical_plan = planner.create_physical_plan(lp_top, &mut lp_arena, &mut expr_arena)?;

        let state = ExecutionState::with_finger_prints(finger_prints);
        Ok((state, physical_plan))
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
    /// fn example(df: DataFrame) -> PolarsResult<DataFrame> {
    ///     df.lazy()
    ///       .groupby([col("foo")])
    ///       .agg([col("bar").sum(), col("ham").mean().alias("avg_ham")])
    ///       .collect()
    /// }
    /// ```
    pub fn collect(self) -> PolarsResult<DataFrame> {
        let (mut state, mut physical_plan) = self.prepare_collect()?;
        let out = physical_plan.execute(&mut state);
        #[cfg(debug_assertions)]
        {
            #[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
            state.file_cache.assert_empty();
        }
        out
    }

    //// Profile a LazyFrame.
    ////
    //// This will run the query and return a tuple
    //// containing the materialized DataFrame and a DataFrame that contains profiling information
    //// of each node that is executed.
    ////
    //// The units of the timings are microseconds.
    pub fn profile(self) -> PolarsResult<(DataFrame, DataFrame)> {
        let (mut state, mut physical_plan) = self.prepare_collect()?;
        state.time_nodes();
        let out = physical_plan.execute(&mut state)?;
        let timer_df = state.finish_timer()?;
        Ok((out, timer_df))
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
    ///         .select(&[col("*").exclude(["foo"])])
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
    /// use polars_arrow::prelude::QuantileInterpolOptions;
    ///
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///        .groupby([col("date")])
    ///        .agg([
    ///            col("rain").min(),
    ///            col("rain").sum(),
    ///            col("rain").quantile(0.5, QuantileInterpolOptions::Nearest).alias("median_rain"),
    ///        ])
    /// }
    /// ```
    pub fn groupby<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(self, by: E) -> LazyGroupBy {
        let keys = by
            .as_ref()
            .iter()
            .map(|e| e.clone().into())
            .collect::<Vec<_>>();
        let opt_state = self.get_opt_state();
        LazyGroupBy {
            logical_plan: self.logical_plan,
            opt_state,
            keys,
            maintain_order: false,
            dynamic_options: None,
            rolling_options: None,
        }
    }

    /// Create rolling groups based on a time column.
    ///
    /// Also works for index values of type Int32 or Int64.
    ///
    /// Different from a [`dynamic_groupby`] the windows are now determined by the
    /// individual values and are not of constant intervals. For constant intervals use
    /// *groupby_dynamic*
    pub fn groupby_rolling<E: AsRef<[Expr]>>(
        self,
        by: E,
        options: RollingGroupOptions,
    ) -> LazyGroupBy {
        let opt_state = self.get_opt_state();
        LazyGroupBy {
            logical_plan: self.logical_plan,
            opt_state,
            keys: by.as_ref().to_vec(),
            maintain_order: true,
            dynamic_options: None,
            rolling_options: Some(options),
        }
    }

    /// Group based on a time value (or index value of type Int32, Int64).
    ///
    /// Time windows are calculated and rows are assigned to windows. Different from a
    /// normal groupby is that a row can be member of multiple groups. The time/index
    /// window could be seen as a rolling window, with a window size determined by
    /// dates/times/values instead of slots in the DataFrame.
    ///
    /// A window is defined by:
    ///
    /// - every: interval of the window
    /// - period: length of the window
    /// - offset: offset of the window
    ///
    /// The `by` argument should be empty `[]` if you don't want to combine this
    /// with a ordinary groupby on these keys.
    pub fn groupby_dynamic<E: AsRef<[Expr]>>(
        self,
        by: E,
        options: DynamicGroupOptions,
    ) -> LazyGroupBy {
        let opt_state = self.get_opt_state();
        LazyGroupBy {
            logical_plan: self.logical_plan,
            opt_state,
            keys: by.as_ref().to_vec(),
            maintain_order: true,
            dynamic_options: Some(options),
            rolling_options: None,
        }
    }

    /// Similar to [`groupby`], but order of the DataFrame is maintained.
    pub fn groupby_stable<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(self, by: E) -> LazyGroupBy {
        let keys = by
            .as_ref()
            .iter()
            .map(|e| e.clone().into())
            .collect::<Vec<_>>();
        let opt_state = self.get_opt_state();
        LazyGroupBy {
            logical_plan: self.logical_plan,
            opt_state,
            keys,
            maintain_order: true,
            dynamic_options: None,
            rolling_options: None,
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
    pub fn left_join<E: Into<Expr>>(self, other: LazyFrame, left_on: E, right_on: E) -> LazyFrame {
        self.join(other, [left_on.into()], [right_on.into()], JoinType::Left)
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
    pub fn outer_join<E: Into<Expr>>(self, other: LazyFrame, left_on: E, right_on: E) -> LazyFrame {
        self.join(other, [left_on.into()], [right_on.into()], JoinType::Outer)
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
    pub fn inner_join<E: Into<Expr>>(self, other: LazyFrame, left_on: E, right_on: E) -> LazyFrame {
        self.join(other, [left_on.into()], [right_on.into()], JoinType::Inner)
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
    ///         .join(other, [col("foo"), col("bar")], [col("foo"), col("bar")], JoinType::Inner)
    /// }
    /// ```
    pub fn join<E: AsRef<[Expr]>>(
        mut self,
        other: LazyFrame,
        left_on: E,
        right_on: E,
        how: JoinType,
    ) -> LazyFrame {
        // if any of the nodes reads from files we must activate this this plan as well.
        self.opt_state.file_caching |= other.opt_state.file_caching;

        let left_on = left_on.as_ref().to_vec();
        let right_on = right_on.as_ref().to_vec();
        self.join_builder()
            .with(other)
            .left_on(left_on)
            .right_on(right_on)
            .how(how)
            .finish()
    }

    /// Control more join options with the join builder.
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
    pub fn with_columns<E: AsRef<[Expr]>>(self, exprs: E) -> LazyFrame {
        let exprs = exprs.as_ref().to_vec();
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().with_columns(exprs).build();
        Self::from_logical_plan(lp, opt_state)
    }

    pub fn with_context<C: AsRef<[LazyFrame]>>(self, contexts: C) -> LazyFrame {
        let contexts = contexts
            .as_ref()
            .iter()
            .map(|lf| lf.logical_plan.clone())
            .collect();
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().with_context(contexts).build();
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
    pub fn quantile(self, quantile: f64, interpol: QuantileInterpolOptions) -> LazyFrame {
        self.select_local(vec![col("*").quantile(quantile, interpol)])
    }

    /// Aggregate all the columns as their standard deviation values.
    pub fn std(self, ddof: u8) -> LazyFrame {
        self.select_local(vec![col("*").std(ddof)])
    }

    /// Aggregate all the columns as their variance values.
    pub fn var(self, ddof: u8) -> LazyFrame {
        self.select_local(vec![col("*").var(ddof)])
    }

    /// Apply explode operation. [See eager explode](polars_core::frame::DataFrame::explode).
    pub fn explode<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(self, columns: E) -> LazyFrame {
        let columns = columns
            .as_ref()
            .iter()
            .map(|e| e.clone().into())
            .collect::<Vec<_>>();
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().explode(columns).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Keep unique rows and maintain order
    pub fn unique_stable(
        self,
        subset: Option<Vec<String>>,
        keep_strategy: UniqueKeepStrategy,
    ) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let options = DistinctOptions {
            subset: subset.map(Arc::new),
            maintain_order: true,
            keep_strategy,
        };
        let lp = self.get_plan_builder().distinct(options).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Keep unique rows, do not maintain order
    pub fn unique(
        self,
        subset: Option<Vec<String>>,
        keep_strategy: UniqueKeepStrategy,
    ) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let options = DistinctOptions {
            subset: subset.map(Arc::new),
            maintain_order: false,
            keep_strategy,
        };
        let lp = self.get_plan_builder().distinct(options).build();
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
    pub fn slice(self, offset: i64, len: IdxSize) -> LazyFrame {
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

    /// Get the last `n` rows
    pub fn tail(self, n: IdxSize) -> LazyFrame {
        let neg_tail = -(n as i64);
        self.slice(neg_tail, n)
    }

    /// Melt the DataFrame from wide to long format
    pub fn melt(self, args: MeltArgs) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().melt(Arc::new(args)).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Limit the DataFrame to the first `n` rows. Note if you don't want the rows to be scanned,
    /// use [fetch](LazyFrame::fetch).
    pub fn limit(self, n: IdxSize) -> LazyFrame {
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
        schema: Option<Arc<dyn UdfSchema>>,
        name: Option<&'static str>,
    ) -> LazyFrame
    where
        F: 'static + Fn(DataFrame) -> PolarsResult<DataFrame> + Send + Sync,
    {
        let opt_state = self.get_opt_state();
        let lp = self
            .get_plan_builder()
            .map(
                function,
                optimizations.unwrap_or_default(),
                schema,
                name.unwrap_or("ANONYMOUS UDF"),
            )
            .build();
        Self::from_logical_plan(lp, opt_state)
    }

    pub(crate) fn map_private(self, function: FunctionNode) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().map_private(function).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Add a new column at index 0 that counts the rows.
    ///
    /// # Warning
    /// This can have a negative effect on query performance.
    /// This may for instance block predicate pushdown optimization.
    pub fn with_row_count(mut self, name: &str, offset: Option<IdxSize>) -> LazyFrame {
        let mut add_row_count_in_map = false;
        match &mut self.logical_plan {
            // Do the row count at scan
            #[cfg(feature = "csv-file")]
            LogicalPlan::CsvScan { options, .. } => {
                options.row_count = Some(RowCount {
                    name: name.to_string(),
                    offset: offset.unwrap_or(0),
                });
            }
            #[cfg(feature = "ipc")]
            LogicalPlan::IpcScan { options, .. } => {
                options.row_count = Some(RowCount {
                    name: name.to_string(),
                    offset: offset.unwrap_or(0),
                });
            }
            #[cfg(feature = "parquet")]
            LogicalPlan::ParquetScan { options, .. } => {
                options.row_count = Some(RowCount {
                    name: name.to_string(),
                    offset: offset.unwrap_or(0),
                });
            }
            _ => {
                add_row_count_in_map = true;
            }
        }

        let name2 = name.to_string();
        let udf_schema = move |s: &Schema| {
            let new = s.insert_index(0, name2.clone(), IDX_DTYPE).unwrap();
            Ok(Arc::new(new))
        };

        let name = name.to_owned();

        // if we do the row count at scan we add a dummy map, to update the schema
        let opt = if add_row_count_in_map {
            AllowedOptimizations {
                slice_pushdown: false,
                predicate_pushdown: false,
                ..Default::default()
            }
        } else {
            AllowedOptimizations::default()
        };

        self.map(
            move |df: DataFrame| {
                if add_row_count_in_map {
                    df.with_row_count(&name, offset)
                } else {
                    Ok(df)
                }
            },
            Some(opt),
            Some(Arc::new(udf_schema)),
            Some("WITH ROW COUNT"),
        )
    }

    /// Unnest the given `Struct` columns. This means that the fields of the `Struct` type will be
    /// inserted as columns.
    #[cfg(feature = "dtype-struct")]
    #[cfg_attr(docsrs, doc(cfg(feature = "dtype-struct")))]
    pub fn unnest<I: IntoVec<String>>(self, cols: I) -> Self {
        let cols = cols.into_vec();
        self.unnest_impl(cols.into_iter().collect())
    }

    #[cfg(feature = "dtype-struct")]
    fn unnest_impl(self, cols: PlHashSet<String>) -> Self {
        let cols2 = cols.clone();
        let udf_schema = move |schema: &Schema| {
            let mut new_schema = Schema::with_capacity(schema.len() * 2);
            for (name, dtype) in schema.iter() {
                if cols.contains(name) {
                    if let DataType::Struct(flds) = dtype {
                        for fld in flds {
                            new_schema.with_column(fld.name().clone(), fld.data_type().clone())
                        }
                    } else {
                        // todo: return lazy error here.
                        panic!("expected struct dtype")
                    }
                } else {
                    new_schema.with_column(name.clone(), dtype.clone())
                }
            }
            Ok(Arc::new(new_schema))
        };
        self.map(
            move |df| df.unnest(&cols2),
            Some(AllowedOptimizations::default()),
            Some(Arc::new(udf_schema)),
            Some("unnest"),
        )
    }
}

/// Utility struct for lazy groupby operation.
#[derive(Clone)]
pub struct LazyGroupBy {
    pub logical_plan: LogicalPlan,
    opt_state: OptState,
    keys: Vec<Expr>,
    maintain_order: bool,
    dynamic_options: Option<DynamicGroupOptions>,
    rolling_options: Option<RollingGroupOptions>,
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
    /// use polars_arrow::prelude::QuantileInterpolOptions;
    ///
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///        .groupby_stable([col("date")])
    ///        .agg([
    ///            col("rain").min(),
    ///            col("rain").sum(),
    ///            col("rain").quantile(0.5, QuantileInterpolOptions::Nearest).alias("median_rain"),
    ///        ])
    /// }
    /// ```
    pub fn agg<E: AsRef<[Expr]>>(self, aggs: E) -> LazyFrame {
        let lp = LogicalPlanBuilder::from(self.logical_plan)
            .groupby(
                Arc::new(self.keys),
                aggs,
                None,
                self.maintain_order,
                self.dynamic_options,
                self.rolling_options,
            )
            .build();
        LazyFrame::from_logical_plan(lp, self.opt_state)
    }

    /// Return first n rows of each group
    pub fn head(self, n: Option<usize>) -> LazyFrame {
        let keys = self
            .keys
            .iter()
            .flat_map(|k| expr_to_leaf_column_names(k).into_iter())
            .collect::<Vec<_>>();

        self.agg([col("*").exclude(&keys).head(n).list().keep_name()])
            .explode([col("*").exclude(&keys)])
    }

    /// Return last n rows of each group
    pub fn tail(self, n: Option<usize>) -> LazyFrame {
        let keys = self
            .keys
            .iter()
            .flat_map(|k| expr_to_leaf_column_names(k).into_iter())
            .collect::<Vec<_>>();

        self.agg([col("*").exclude(&keys).tail(n).keep_name()])
            .explode([col("*").exclude(&keys)])
    }

    /// Apply a function over the groups as a new `DataFrame`. It is not recommended that you use
    /// this as materializing the `DataFrame` is very expensive.
    pub fn apply<F>(self, f: F, schema: SchemaRef) -> LazyFrame
    where
        F: 'static + Fn(DataFrame) -> PolarsResult<DataFrame> + Send + Sync,
    {
        let lp = LogicalPlan::Aggregate {
            input: Box::new(self.logical_plan),
            keys: Arc::new(self.keys),
            aggs: vec![],
            schema,
            apply: Some(Arc::new(f)),
            maintain_order: self.maintain_order,
            options: GroupbyOptions {
                dynamic: None,
                rolling: None,
                slice: None,
            },
        };
        LazyFrame::from_logical_plan(lp, self.opt_state)
    }
}

#[must_use]
pub struct JoinBuilder {
    lf: LazyFrame,
    how: JoinType,
    other: Option<LazyFrame>,
    left_on: Vec<Expr>,
    right_on: Vec<Expr>,
    allow_parallel: bool,
    force_parallel: bool,
    suffix: Option<String>,
}
impl JoinBuilder {
    pub fn new(lf: LazyFrame) -> Self {
        Self {
            lf,
            other: None,
            how: JoinType::Inner,
            left_on: vec![],
            right_on: vec![],
            allow_parallel: true,
            force_parallel: false,
            suffix: None,
        }
    }

    /// The table to join with.
    pub fn with(mut self, other: LazyFrame) -> Self {
        self.other = Some(other);
        self
    }

    /// Select the join type.
    pub fn how(mut self, how: JoinType) -> Self {
        self.how = how;
        self
    }

    /// The columns you want to join the left table on.
    pub fn left_on<E: AsRef<[Expr]>>(mut self, on: E) -> Self {
        self.left_on = on.as_ref().to_vec();
        self
    }

    /// The columns you want to join the right table on.
    pub fn right_on<E: AsRef<[Expr]>>(mut self, on: E) -> Self {
        self.right_on = on.as_ref().to_vec();
        self
    }
    /// Allow parallel table evaluation.
    pub fn allow_parallel(mut self, allow: bool) -> Self {
        self.allow_parallel = allow;
        self
    }

    /// Force parallel table evaluation.
    pub fn force_parallel(mut self, allow: bool) -> Self {
        self.allow_parallel = allow;
        self
    }

    /// Suffix to add duplicate column names in join.
    /// Defaults to `"_right"`.
    pub fn suffix<S: AsRef<str>>(mut self, suffix: S) -> Self {
        self.suffix = Some(suffix.as_ref().to_string());
        self
    }

    /// Finish builder
    pub fn finish(self) -> LazyFrame {
        let mut opt_state = self.lf.opt_state;
        let other = self.other.expect("with not set");

        // if any of the nodes reads from files we must activate this this plan as well.
        opt_state.file_caching |= other.opt_state.file_caching;

        let suffix = match self.suffix {
            None => Cow::Borrowed("_right"),
            Some(suffix) => Cow::Owned(suffix),
        };

        let lp = self
            .lf
            .get_plan_builder()
            .join(
                other.logical_plan,
                self.left_on,
                self.right_on,
                JoinOptions {
                    allow_parallel: self.allow_parallel,
                    force_parallel: self.force_parallel,
                    how: self.how,
                    suffix,
                    slice: None,
                },
            )
            .build();
        LazyFrame::from_logical_plan(lp, opt_state)
    }
}
