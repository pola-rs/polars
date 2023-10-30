//! Lazy variant of a [DataFrame].
#[cfg(feature = "python")]
mod python;

mod err;
#[cfg(feature = "pivot")]
pub mod pivot;

use std::borrow::Cow;
#[cfg(any(feature = "parquet", feature = "ipc", feature = "csv"))]
use std::path::PathBuf;
use std::sync::Arc;

pub use anonymous_scan::*;
use arrow::legacy::prelude::QuantileInterpolOptions;
#[cfg(feature = "csv")]
pub use csv::*;
pub use file_list_reader::*;
#[cfg(feature = "ipc")]
pub use ipc::*;
#[cfg(feature = "json")]
pub use ndjson::*;
#[cfg(feature = "parquet")]
pub use parquet::*;
use polars_core::frame::explode::MeltArgs;
use polars_core::prelude::*;
use polars_io::RowCount;
pub use polars_plan::frame::{AllowedOptimizations, OptState};
use polars_plan::global::FETCH_ROWS;
#[cfg(any(feature = "ipc", feature = "parquet", feature = "csv"))]
use polars_plan::logical_plan::collect_fingerprints;
use polars_plan::logical_plan::optimize;
use polars_plan::utils::expr_output_name;
use smartstring::alias::String as SmartString;

use crate::fallible;
use crate::physical_plan::executors::Executor;
use crate::physical_plan::planner::{create_physical_expr, create_physical_plan};
use crate::physical_plan::state::ExecutionState;
#[cfg(feature = "streaming")]
use crate::physical_plan::streaming::insert_streaming_nodes;
use crate::prelude::*;

pub trait IntoLazy {
    fn lazy(self) -> LazyFrame;
}

impl IntoLazy for DataFrame {
    /// Convert the `DataFrame` into a `LazyFrame`
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
/// modify a logical plan until output is requested (via [`collect`](crate::frame::LazyFrame::collect)).
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

impl LazyFrame {
    /// Get a handle to the schema — a map from column names to data types — of the current
    /// `LazyFrame` computation.
    ///
    /// Returns an `Err` if the logical plan has already encountered an error (i.e., if
    /// `self.collect()` would fail), `Ok` otherwise.
    pub fn schema(&self) -> PolarsResult<SchemaRef> {
        self.logical_plan.schema().map(|schema| schema.into_owned())
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

    /// Get current optimizations.
    pub fn get_current_optimizations(&self) -> OptState {
        self.opt_state
    }

    /// Set allowed optimizations.
    pub fn with_optimizations(mut self, opt_state: OptState) -> Self {
        self.opt_state = opt_state;
        self
    }

    /// Turn off all optimizations.
    pub fn without_optimizations(self) -> Self {
        self.with_optimizations(OptState {
            projection_pushdown: false,
            predicate_pushdown: false,
            type_coercion: true,
            simplify_expr: false,
            slice_pushdown: false,
            // will be toggled by a scan operation such as csv scan or parquet scan
            file_caching: false,
            #[cfg(feature = "cse")]
            comm_subplan_elim: false,
            #[cfg(feature = "cse")]
            comm_subexpr_elim: false,
            streaming: false,
            eager: false,
            fast_projection: false,
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

    /// Toggle expression simplification optimization on or off.
    pub fn with_simplify_expr(mut self, toggle: bool) -> Self {
        self.opt_state.simplify_expr = toggle;
        self
    }

    /// Toggle common subplan elimination optimization on or off
    #[cfg(feature = "cse")]
    pub fn with_comm_subplan_elim(mut self, toggle: bool) -> Self {
        self.opt_state.comm_subplan_elim = toggle;
        self
    }

    /// Toggle common subexpression elimination optimization on or off
    #[cfg(feature = "cse")]
    pub fn with_comm_subexpr_elim(mut self, toggle: bool) -> Self {
        self.opt_state.comm_subexpr_elim = toggle;
        self
    }

    /// Toggle slice pushdown optimization.
    pub fn with_slice_pushdown(mut self, toggle: bool) -> Self {
        self.opt_state.slice_pushdown = toggle;
        self
    }

    /// Allow (partial) streaming engine.
    pub fn with_streaming(mut self, toggle: bool) -> Self {
        self.opt_state.streaming = toggle;
        self
    }

    pub fn _with_eager(mut self, toggle: bool) -> Self {
        self.opt_state.eager = toggle;
        self
    }

    /// Return a String describing the naive (un-optimized) logical plan.
    pub fn describe_plan(&self) -> String {
        self.logical_plan.describe()
    }

    /// Return a String describing the optimized logical plan.
    ///
    /// Returns `Err` if optimizing the logical plan fails.
    pub fn describe_optimized_plan(&self) -> PolarsResult<String> {
        let mut expr_arena = Arena::with_capacity(64);
        let mut lp_arena = Arena::with_capacity(64);
        let lp_top = self.clone().optimize_with_scratch(
            &mut lp_arena,
            &mut expr_arena,
            &mut vec![],
            true,
        )?;
        let logical_plan = node_to_lp(lp_top, &expr_arena, &mut lp_arena);
        Ok(logical_plan.describe())
    }

    /// Return a String describing the logical plan.
    ///
    /// If `optimized` is `true`, explains the optimized plan. If `optimized` is `false,
    /// explains the naive, un-optimized plan.
    pub fn explain(&self, optimized: bool) -> PolarsResult<String> {
        if optimized {
            self.describe_optimized_plan()
        } else {
            Ok(self.describe_plan())
        }
    }

    /// Add a sort operation to the logical plan.
    ///
    /// Sorts the LazyFrame by the column name specified using the provided options.
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
        let descending = options.descending;
        let nulls_last = options.nulls_last;
        let maintain_order = options.maintain_order;

        let opt_state = self.get_opt_state();
        let lp = self
            .get_plan_builder()
            .sort(
                vec![col(by_column)],
                vec![descending],
                nulls_last,
                maintain_order,
            )
            .build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Add a sort operation to the logical plan.
    ///
    /// Sorts the LazyFrame by the provided list of expressions, which will be turned into
    /// concrete columns before sorting. `reverse` is a list of `bool`, the same length as
    /// `by_exprs`, that specifies whether each corresponding expression will be sorted
    /// ascending (`false`) or descending (`true`).
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
    ///         .sort_by_exprs(vec![col("sepal.width")], vec![false], false, false)
    /// }
    /// ```
    pub fn sort_by_exprs<E: AsRef<[Expr]>, B: AsRef<[bool]>>(
        self,
        by_exprs: E,
        descending: B,
        nulls_last: bool,
        maintain_order: bool,
    ) -> Self {
        let by_exprs = by_exprs.as_ref().to_vec();
        let descending = descending.as_ref().to_vec();
        if by_exprs.is_empty() {
            self
        } else {
            let opt_state = self.get_opt_state();
            let lp = self
                .get_plan_builder()
                .sort(by_exprs, descending, nulls_last, maintain_order)
                .build();
            Self::from_logical_plan(lp, opt_state)
        }
    }

    pub fn top_k<E: AsRef<[Expr]>, B: AsRef<[bool]>>(
        self,
        k: IdxSize,
        by_exprs: E,
        descending: B,
        nulls_last: bool,
        maintain_order: bool,
    ) -> Self {
        let mut descending = descending.as_ref().to_vec();
        // top-k is reverse from sort
        for v in &mut descending {
            *v = !*v;
        }
        // this will optimize to top-k
        self.sort_by_exprs(by_exprs, descending, nulls_last, maintain_order)
            .slice(0, k)
    }

    pub fn bottom_k<E: AsRef<[Expr]>, B: AsRef<[bool]>>(
        self,
        k: IdxSize,
        by_exprs: E,
        descending: B,
        nulls_last: bool,
        maintain_order: bool,
    ) -> Self {
        let descending = descending.as_ref().to_vec();
        // this will optimize to bottom-k
        self.sort_by_exprs(by_exprs, descending, nulls_last, maintain_order)
            .slice(0, k)
    }

    /// Reverse the `DataFrame` from top to bottom.
    ///
    /// Row `i` becomes row `number_of_rows - i - 1`.
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
        self.select(vec![col("*").reverse()])
    }

    /// Check the if the `names` are available in the `schema`, if not
    /// return a `LogicalPlan` that raises an `Error`.
    fn check_names(&self, names: &[SmartString], schema: Option<&SchemaRef>) -> Option<Self> {
        let schema = schema
            .map(Cow::Borrowed)
            .unwrap_or_else(|| Cow::Owned(self.schema().unwrap()));

        let mut opt_not_found = None;
        names.iter().for_each(|name| {
            let invalid = schema.get(name).is_none();

            if invalid && opt_not_found.is_none() {
                opt_not_found = Some(name)
            }
        });

        if let Some(name) = opt_not_found {
            let lp = self
                .clone()
                .get_plan_builder()
                .add_err(polars_err!(SchemaFieldNotFound: "{}", name))
                .build();
            Some(Self::from_logical_plan(lp, self.opt_state))
        } else {
            None
        }
    }

    /// Rename columns in the DataFrame.
    ///
    /// `existing` and `new` are iterables of the same length containing the old and
    /// corresponding new column names. Renaming happens to all `existing` columns
    /// simultaneously, not iteratively. (In particular, all columns in `existing` must
    /// already exist in the `LazyFrame` when `rename` is called.)
    pub fn rename<I, J, T, S>(self, existing: I, new: J) -> Self
    where
        I: IntoIterator<Item = T>,
        J: IntoIterator<Item = S>,
        T: AsRef<str>,
        S: AsRef<str>,
    {
        let iter = existing.into_iter();
        let cap = iter.size_hint().0;
        let mut existing_vec: Vec<SmartString> = Vec::with_capacity(cap);
        let mut new_vec: Vec<SmartString> = Vec::with_capacity(cap);

        // todo! should this error if `existing` and `new` have different lengths?
        // Currently, the longer of the two is truncated.
        for (existing, new) in iter.zip(new) {
            let existing = existing.as_ref();
            let new = new.as_ref();
            if new != existing {
                existing_vec.push(existing.into());
                new_vec.push(new.into());
            }
        }

        // a column gets swapped
        let schema = &self.schema().unwrap();
        let swapping = new_vec.iter().any(|name| schema.get(name).is_some());

        if let Some(lp) = self.check_names(&existing_vec, Some(schema)) {
            lp
        } else {
            self.map_private(FunctionNode::Rename {
                existing: existing_vec.into(),
                new: new_vec.into(),
                swapping,
            })
        }
    }

    /// Removes columns from the DataFrame.
    /// Note that it's better to only select the columns you need
    /// and let the projection pushdown optimize away the unneeded columns.
    pub fn drop_columns<I, T>(self, columns: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: AsRef<str>,
    {
        let to_drop = columns
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect::<PlHashSet<_>>();

        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().drop_columns(to_drop).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with `Nones`.
    ///
    /// See the method on [Series](polars_core::series::SeriesTrait::shift) for more info on the `shift` operation.
    pub fn shift<E: Into<Expr>>(self, n: E) -> Self {
        self.select(vec![col("*").shift(n.into())])
    }

    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with the result of the `fill_value` expression.
    ///
    /// See the method on [Series](polars_core::series::SeriesTrait::shift) for more info on the `shift` operation.
    pub fn shift_and_fill<E: Into<Expr>>(self, n: E, fill_value: E) -> Self {
        self.select(vec![col("*").shift_and_fill(n.into(), fill_value.into())])
    }

    /// Fill None values in the DataFrame with an expression.
    pub fn fill_null<E: Into<Expr>>(self, fill_value: E) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().fill_null(fill_value.into()).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Fill NaN values in the DataFrame with an expression.
    pub fn fill_nan<E: Into<Expr>>(self, fill_value: E) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().fill_nan(fill_value.into()).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Caches the result into a new LazyFrame.
    ///
    /// This should be used to prevent computations running multiple times.
    pub fn cache(self) -> Self {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().cache().build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Cast named frame columns, resulting in a new LazyFrame with updated dtypes
    pub fn cast(self, dtypes: PlHashMap<&str, DataType>, strict: bool) -> Self {
        let cast_cols: Vec<Expr> = dtypes
            .into_iter()
            .map(|(name, dt)| {
                if strict {
                    col(name).strict_cast(dt)
                } else {
                    col(name).cast(dt)
                }
            })
            .collect();
        self.with_columns(cast_cols)
    }

    /// Cast all frame columns to the given dtype, resulting in a new LazyFrame
    pub fn cast_all(self, dtype: DataType, strict: bool) -> Self {
        self.with_columns(vec![if strict {
            col("*").strict_cast(dtype)
        } else {
            col("*").cast(dtype)
        }])
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
        self.optimize_with_scratch(lp_arena, expr_arena, &mut vec![], false)
    }

    pub fn to_alp_optimized(self) -> PolarsResult<(Node, Arena<ALogicalPlan>, Arena<AExpr>)> {
        let mut lp_arena = Arena::with_capacity(16);
        let mut expr_arena = Arena::with_capacity(16);
        let node =
            self.optimize_with_scratch(&mut lp_arena, &mut expr_arena, &mut vec![], false)?;
        Ok((node, lp_arena, expr_arena))
    }

    pub fn to_alp(self) -> PolarsResult<(Node, Arena<ALogicalPlan>, Arena<AExpr>)> {
        self.logical_plan.to_alp()
    }

    pub(crate) fn optimize_with_scratch(
        self,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
        scratch: &mut Vec<Node>,
        _fmt: bool,
    ) -> PolarsResult<Node> {
        #[allow(unused_mut)]
        let mut opt_state = self.opt_state;
        let streaming = self.opt_state.streaming;
        #[cfg(feature = "cse")]
        if streaming && self.opt_state.comm_subplan_elim {
            polars_warn!(
                "Cannot combine 'streaming' with 'comm_subplan_elim'. CSE will be turned off."
            );
            opt_state.comm_subplan_elim = false;
        }
        let lp_top = optimize(
            self.logical_plan,
            opt_state,
            lp_arena,
            expr_arena,
            scratch,
            Some(&|node, expr_arena| {
                let phys_expr = create_physical_expr(
                    node,
                    Context::Default,
                    expr_arena,
                    None,
                    &mut Default::default(),
                )
                .ok()?;
                let io_expr = phys_expr_to_io_expr(phys_expr);
                Some(io_expr)
            }),
        )?;

        if streaming {
            #[cfg(feature = "streaming")]
            {
                insert_streaming_nodes(lp_top, lp_arena, expr_arena, scratch, _fmt, true)?;
            }
            #[cfg(not(feature = "streaming"))]
            {
                panic!("activate feature 'streaming'")
            }
        }
        Ok(lp_top)
    }

    #[allow(unused_mut)]
    fn prepare_collect(
        mut self,
        check_sink: bool,
    ) -> PolarsResult<(ExecutionState, Box<dyn Executor>, bool)> {
        let file_caching = self.opt_state.file_caching;
        let mut expr_arena = Arena::with_capacity(256);
        let mut lp_arena = Arena::with_capacity(128);
        let mut scratch = vec![];
        let lp_top =
            self.optimize_with_scratch(&mut lp_arena, &mut expr_arena, &mut scratch, false)?;

        let finger_prints = if file_caching {
            #[cfg(any(feature = "ipc", feature = "parquet", feature = "csv"))]
            {
                let mut fps = Vec::with_capacity(8);
                collect_fingerprints(lp_top, &mut fps, &lp_arena, &expr_arena);
                Some(fps)
            }
            #[cfg(not(any(feature = "ipc", feature = "parquet", feature = "csv")))]
            {
                None
            }
        } else {
            None
        };

        // sink should be replaced
        let no_file_sink = if check_sink {
            !matches!(lp_arena.get(lp_top), ALogicalPlan::Sink { .. })
        } else {
            true
        };
        let physical_plan = create_physical_plan(lp_top, &mut lp_arena, &mut expr_arena)?;

        let state = ExecutionState::with_finger_prints(finger_prints);
        Ok((state, physical_plan, no_file_sink))
    }

    /// Execute all the lazy operations and collect them into a [`DataFrame`].
    ///
    /// The query is optimized prior to execution.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    ///
    /// fn example(df: DataFrame) -> PolarsResult<DataFrame> {
    ///     df.lazy()
    ///       .group_by([col("foo")])
    ///       .agg([col("bar").sum(), col("ham").mean().alias("avg_ham")])
    ///       .collect()
    /// }
    /// ```
    pub fn collect(self) -> PolarsResult<DataFrame> {
        let (mut state, mut physical_plan, _) = self.prepare_collect(false)?;
        let out = physical_plan.execute(&mut state);
        #[cfg(debug_assertions)]
        {
            #[cfg(any(feature = "ipc", feature = "parquet", feature = "csv"))]
            state.file_cache.assert_empty();
        }
        out
    }

    /// Profile a LazyFrame.
    ///
    /// This will run the query and return a tuple
    /// containing the materialized DataFrame and a DataFrame that contains profiling information
    /// of each node that is executed.
    ///
    /// The units of the timings are microseconds.
    pub fn profile(self) -> PolarsResult<(DataFrame, DataFrame)> {
        let (mut state, mut physical_plan, _) = self.prepare_collect(false)?;
        state.time_nodes();
        let out = physical_plan.execute(&mut state)?;
        let timer_df = state.finish_timer()?;
        Ok((out, timer_df))
    }

    /// Stream a query result into a parquet file. This is useful if the final result doesn't fit
    /// into memory. This methods will return an error if the query cannot be completely done in a
    /// streaming fashion.
    #[cfg(feature = "parquet")]
    pub fn sink_parquet(mut self, path: PathBuf, options: ParquetWriteOptions) -> PolarsResult<()> {
        self.opt_state.streaming = true;
        self.logical_plan = LogicalPlan::Sink {
            input: Box::new(self.logical_plan),
            payload: SinkType::File {
                path: Arc::new(path),
                file_type: FileType::Parquet(options),
            },
        };
        let (mut state, mut physical_plan, is_streaming) = self.prepare_collect(true)?;
        polars_ensure!(
            is_streaming,
            ComputeError: "cannot run the whole query in a streaming order; \
            use `collect().write_parquet()` instead"
        );
        let _ = physical_plan.execute(&mut state)?;
        Ok(())
    }

    /// Stream a query result into a parquet file on an ObjectStore-compatible cloud service. This is useful if the final result doesn't fit
    /// into memory, and where you do not want to write to a local file but to a location in the cloud.
    /// This method will return an error if the query cannot be completely done in a
    /// streaming fashion.
    #[cfg(all(feature = "cloud_write", feature = "parquet"))]
    pub fn sink_parquet_cloud(
        mut self,
        uri: String,
        cloud_options: Option<polars_io::cloud::CloudOptions>,
        parquet_options: ParquetWriteOptions,
    ) -> PolarsResult<()> {
        self.opt_state.streaming = true;
        self.logical_plan = LogicalPlan::Sink {
            input: Box::new(self.logical_plan),
            payload: SinkType::Cloud {
                uri: Arc::new(uri),
                cloud_options,
                file_type: FileType::Parquet(parquet_options),
            },
        };
        let (mut state, mut physical_plan, is_streaming) = self.prepare_collect(true)?;
        polars_ensure!(
            is_streaming,
            ComputeError: "cannot run the whole query in a streaming order"
        );
        let _ = physical_plan.execute(&mut state)?;
        Ok(())
    }

    /// Stream a query result into an ipc/arrow file. This is useful if the final result doesn't fit
    /// into memory. This methods will return an error if the query cannot be completely done in a
    /// streaming fashion.
    #[cfg(feature = "ipc")]
    pub fn sink_ipc(mut self, path: PathBuf, options: IpcWriterOptions) -> PolarsResult<()> {
        self.opt_state.streaming = true;
        self.logical_plan = LogicalPlan::Sink {
            input: Box::new(self.logical_plan),
            payload: SinkType::File {
                path: Arc::new(path),
                file_type: FileType::Ipc(options),
            },
        };
        let (mut state, mut physical_plan, is_streaming) = self.prepare_collect(true)?;
        polars_ensure!(
            is_streaming,
            ComputeError: "cannot run the whole query in a streaming order; \
            use `collect().write_ipc()` instead"
        );
        let _ = physical_plan.execute(&mut state)?;
        Ok(())
    }

    /// Stream a query result into an csv file. This is useful if the final result doesn't fit
    /// into memory. This methods will return an error if the query cannot be completely done in a
    /// streaming fashion.
    #[cfg(feature = "csv")]
    pub fn sink_csv(mut self, path: PathBuf, options: CsvWriterOptions) -> PolarsResult<()> {
        self.opt_state.streaming = true;
        self.logical_plan = LogicalPlan::Sink {
            input: Box::new(self.logical_plan),
            payload: SinkType::File {
                path: Arc::new(path),
                file_type: FileType::Csv(options),
            },
        };
        let (mut state, mut physical_plan, is_streaming) = self.prepare_collect(true)?;
        polars_ensure!(
            is_streaming,
            ComputeError: "cannot run the whole query in a streaming order; \
            use `collect().write_csv()` instead"
        );
        let _ = physical_plan.execute(&mut state)?;
        Ok(())
    }

    /// Filter by some predicate expression.
    ///
    /// The expression must yield boolean values.
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

    /// Select (and optionally rename, with [`alias`](crate::dsl::Expr::alias)) columns from the query.
    ///
    /// Columns can be selected with [`col`];
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
        let exprs = exprs.as_ref().to_vec();
        self.select_impl(
            exprs,
            ProjectionOptions {
                run_parallel: true,
                duplicate_check: true,
            },
        )
    }

    pub fn select_seq<E: AsRef<[Expr]>>(self, exprs: E) -> Self {
        let exprs = exprs.as_ref().to_vec();
        self.select_impl(
            exprs,
            ProjectionOptions {
                run_parallel: false,
                duplicate_check: true,
            },
        )
    }

    fn select_impl(self, exprs: Vec<Expr>, options: ProjectionOptions) -> Self {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().project(exprs, options).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Performs a "group-by" on a `LazyFrame`, producing a [`LazyGroupBy`], which can subsequently be aggregated.
    ///
    /// Takes a list of expressions to group on.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    /// use arrow::legacy::prelude::QuantileInterpolOptions;
    ///
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///        .group_by([col("date")])
    ///        .agg([
    ///            col("rain").min().alias("min_rain"),
    ///            col("rain").sum().alias("sum_rain"),
    ///            col("rain").quantile(lit(0.5), QuantileInterpolOptions::Nearest).alias("median_rain"),
    ///        ])
    /// }
    /// ```
    pub fn group_by<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(self, by: E) -> LazyGroupBy {
        let keys = by
            .as_ref()
            .iter()
            .map(|e| e.clone().into())
            .collect::<Vec<_>>();
        let opt_state = self.get_opt_state();

        #[cfg(feature = "dynamic_group_by")]
        {
            LazyGroupBy {
                logical_plan: self.logical_plan,
                opt_state,
                keys,
                maintain_order: false,
                dynamic_options: None,
                rolling_options: None,
            }
        }

        #[cfg(not(feature = "dynamic_group_by"))]
        {
            LazyGroupBy {
                logical_plan: self.logical_plan,
                opt_state,
                keys,
                maintain_order: false,
            }
        }
    }

    /// Create rolling groups based on a time column.
    ///
    /// Also works for index values of type Int32 or Int64.
    ///
    /// Different from a [`group_by_dynamic`][`Self::group_by_dynamic`], the windows are now determined by the
    /// individual values and are not of constant intervals. For constant intervals use
    /// *group_by_dynamic*
    #[cfg(feature = "dynamic_group_by")]
    pub fn group_by_rolling<E: AsRef<[Expr]>>(
        self,
        index_column: Expr,
        by: E,
        mut options: RollingGroupOptions,
    ) -> LazyGroupBy {
        if let Expr::Column(name) = index_column {
            options.index_column = name.as_ref().into();
        } else {
            let name = expr_output_name(&index_column).unwrap();
            return self.with_column(index_column).group_by_rolling(
                Expr::Column(name),
                by,
                options,
            );
        }
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
    /// normal group_by is that a row can be member of multiple groups. The time/index
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
    /// with a ordinary group_by on these keys.
    #[cfg(feature = "dynamic_group_by")]
    pub fn group_by_dynamic<E: AsRef<[Expr]>>(
        self,
        index_column: Expr,
        by: E,
        mut options: DynamicGroupOptions,
    ) -> LazyGroupBy {
        if let Expr::Column(name) = index_column {
            options.index_column = name.as_ref().into();
        } else {
            let name = expr_output_name(&index_column).unwrap();
            return self.with_column(index_column).group_by_dynamic(
                Expr::Column(name),
                by,
                options,
            );
        }
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

    /// Similar to [`group_by`][`Self::group_by`], but order of the DataFrame is maintained.
    pub fn group_by_stable<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(self, by: E) -> LazyGroupBy {
        let keys = by
            .as_ref()
            .iter()
            .map(|e| e.clone().into())
            .collect::<Vec<_>>();
        let opt_state = self.get_opt_state();

        #[cfg(feature = "dynamic_group_by")]
        {
            LazyGroupBy {
                logical_plan: self.logical_plan,
                opt_state,
                keys,
                maintain_order: true,
                dynamic_options: None,
                rolling_options: None,
            }
        }

        #[cfg(not(feature = "dynamic_group_by"))]
        {
            LazyGroupBy {
                logical_plan: self.logical_plan,
                opt_state,
                keys,
                maintain_order: true,
            }
        }
    }

    /// Left anti join this query with another lazy query.
    ///
    /// Matches on the values of the expressions `left_on` and `right_on`. For more
    /// flexible join logic, see [`join`](LazyFrame::join) or
    /// [`join_builder`](LazyFrame::join_builder).
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    /// fn anti_join_dataframes(ldf: LazyFrame, other: LazyFrame) -> LazyFrame {
    ///         ldf
    ///         .anti_join(other, col("foo"), col("bar").cast(DataType::Utf8))
    /// }
    /// ```
    #[cfg(feature = "semi_anti_join")]
    pub fn anti_join<E: Into<Expr>>(self, other: LazyFrame, left_on: E, right_on: E) -> LazyFrame {
        self.join(
            other,
            [left_on.into()],
            [right_on.into()],
            JoinArgs::new(JoinType::Anti),
        )
    }

    /// Creates the cartesian product from both frames, preserving the order of the left keys.
    #[cfg(feature = "cross_join")]
    pub fn cross_join(self, other: LazyFrame) -> LazyFrame {
        self.join(other, vec![], vec![], JoinArgs::new(JoinType::Cross))
    }

    /// Left join this query with another lazy query.
    ///
    /// Matches on the values of the expressions `left_on` and `right_on`. For more
    /// flexible join logic, see [`join`](LazyFrame::join) or
    /// [`join_builder`](LazyFrame::join_builder).
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    /// fn left_join_dataframes(ldf: LazyFrame, other: LazyFrame) -> LazyFrame {
    ///         ldf
    ///         .left_join(other, col("foo"), col("bar"))
    /// }
    /// ```
    pub fn left_join<E: Into<Expr>>(self, other: LazyFrame, left_on: E, right_on: E) -> LazyFrame {
        self.join(
            other,
            [left_on.into()],
            [right_on.into()],
            JoinArgs::new(JoinType::Left),
        )
    }

    /// Inner join this query with another lazy query.
    ///
    /// Matches on the values of the expressions `left_on` and `right_on`. For more
    /// flexible join logic, see [`join`](LazyFrame::join) or
    /// [`join_builder`](LazyFrame::join_builder).
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    /// fn inner_join_dataframes(ldf: LazyFrame, other: LazyFrame) -> LazyFrame {
    ///         ldf
    ///         .inner_join(other, col("foo"), col("bar").cast(DataType::Utf8))
    /// }
    /// ```
    pub fn inner_join<E: Into<Expr>>(self, other: LazyFrame, left_on: E, right_on: E) -> LazyFrame {
        self.join(
            other,
            [left_on.into()],
            [right_on.into()],
            JoinArgs::new(JoinType::Inner),
        )
    }

    /// Outer join this query with another lazy query.
    ///
    /// Matches on the values of the expressions `left_on` and `right_on`. For more
    /// flexible join logic, see [`join`](LazyFrame::join) or
    /// [`join_builder`](LazyFrame::join_builder).
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    /// fn outer_join_dataframes(ldf: LazyFrame, other: LazyFrame) -> LazyFrame {
    ///         ldf
    ///         .outer_join(other, col("foo"), col("bar"))
    /// }
    /// ```
    pub fn outer_join<E: Into<Expr>>(self, other: LazyFrame, left_on: E, right_on: E) -> LazyFrame {
        self.join(
            other,
            [left_on.into()],
            [right_on.into()],
            JoinArgs::new(JoinType::Outer),
        )
    }

    /// Left semi join this query with another lazy query.
    ///
    /// Matches on the values of the expressions `left_on` and `right_on`. For more
    /// flexible join logic, see [`join`](LazyFrame::join) or
    /// [`join_builder`](LazyFrame::join_builder).
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    /// fn semi_join_dataframes(ldf: LazyFrame, other: LazyFrame) -> LazyFrame {
    ///         ldf
    ///         .semi_join(other, col("foo"), col("bar").cast(DataType::Utf8))
    /// }
    /// ```
    #[cfg(feature = "semi_anti_join")]
    pub fn semi_join<E: Into<Expr>>(self, other: LazyFrame, left_on: E, right_on: E) -> LazyFrame {
        self.join(
            other,
            [left_on.into()],
            [right_on.into()],
            JoinArgs::new(JoinType::Semi),
        )
    }

    /// Generic function to join two LazyFrames.
    ///
    /// `join` can join on multiple columns, given as two list of expressions, and with a
    /// [`JoinType`] specified by `how`. Non-joined column names in the right DataFrame
    /// that already exist in this DataFrame are suffixed with `"_right"`. For control
    /// over how columns are renamed and parallelization options, use
    /// [`join_builder`](LazyFrame::join_builder).
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    ///
    /// fn example(ldf: LazyFrame, other: LazyFrame) -> LazyFrame {
    ///         ldf
    ///         .join(other, [col("foo"), col("bar")], [col("foo"), col("bar")], JoinArgs::new(JoinType::Inner))
    /// }
    /// ```
    pub fn join<E: AsRef<[Expr]>>(
        mut self,
        other: LazyFrame,
        left_on: E,
        right_on: E,
        args: JoinArgs,
    ) -> LazyFrame {
        // if any of the nodes reads from files we must activate this this plan as well.
        self.opt_state.file_caching |= other.opt_state.file_caching;

        let left_on = left_on.as_ref().to_vec();
        let right_on = right_on.as_ref().to_vec();
        self.join_builder()
            .with(other)
            .left_on(left_on)
            .right_on(right_on)
            .how(args.how)
            .finish()
    }

    /// Consume `self` and return a [`JoinBuilder`] to customize a join on this LazyFrame.
    ///
    /// After the `JoinBuilder` has been created and set up, calling
    /// [`finish()`](JoinBuilder::finish) on it will give back the `LazyFrame`
    /// representing the `join` operation.
    pub fn join_builder(self) -> JoinBuilder {
        JoinBuilder::new(self)
    }

    /// Add or replace a column, given as an expression, to a DataFrame.
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
    ///         )
    /// }
    /// ```
    pub fn with_column(self, expr: Expr) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self
            .get_plan_builder()
            .with_columns(
                vec![expr],
                ProjectionOptions {
                    run_parallel: false,
                    duplicate_check: true,
                },
            )
            .build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Add or replace multiple columns, given as expressions, to a DataFrame.
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
        self.with_columns_impl(
            exprs,
            ProjectionOptions {
                run_parallel: true,
                duplicate_check: true,
            },
        )
    }

    /// Add or replace multiple columns to a DataFrame, but evaluate them sequentially.
    pub fn with_columns_seq<E: AsRef<[Expr]>>(self, exprs: E) -> LazyFrame {
        let exprs = exprs.as_ref().to_vec();
        self.with_columns_impl(
            exprs,
            ProjectionOptions {
                run_parallel: false,
                duplicate_check: true,
            },
        )
    }

    fn with_columns_impl(self, exprs: Vec<Expr>, options: ProjectionOptions) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().with_columns(exprs, options).build();
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
    ///
    /// Aggregated columns will have the same names as the original columns.
    pub fn max(self) -> LazyFrame {
        self.select(vec![col("*").max()])
    }

    /// Aggregate all the columns as their minimum values.
    ///
    /// Aggregated columns will have the same names as the original columns.
    pub fn min(self) -> LazyFrame {
        self.select(vec![col("*").min()])
    }

    /// Aggregate all the columns as their sum values.
    ///
    /// Aggregated columns will have the same names as the original columns.
    ///
    /// - Boolean columns will sum to a `u32` containing the number of `true`s.
    /// - For integer columns, the ordinary checks for overflow are performed: if running
    /// in `debug` mode, overflows will panic, whereas in `release` mode overflows will
    /// silently wrap.
    /// - String columns will sum to None.
    pub fn sum(self) -> LazyFrame {
        self.select(vec![col("*").sum()])
    }

    /// Aggregate all the columns as their mean values.
    ///
    /// - Boolean and integer columns are converted to `f64` before computing the mean.
    /// - String columns will have a mean of None.
    pub fn mean(self) -> LazyFrame {
        self.select(vec![col("*").mean()])
    }

    /// Aggregate all the columns as their median values.
    ///
    /// - Boolean and integer results are converted to `f64`. However, they are still
    ///   susceptible to overflow before this conversion occurs.
    /// - String columns will sum to None.
    pub fn median(self) -> LazyFrame {
        self.select(vec![col("*").median()])
    }

    /// Aggregate all the columns as their quantile values.
    pub fn quantile(self, quantile: Expr, interpol: QuantileInterpolOptions) -> LazyFrame {
        self.select(vec![col("*").quantile(quantile, interpol)])
    }

    /// Aggregate all the columns as their standard deviation values.
    ///
    /// `ddof` is the "Delta Degrees of Freedom"; `N - ddof` will be the denominator when
    /// computing the variance, where `N` is the number of rows.
    /// > In standard statistical practice, `ddof=1` provides an unbiased estimator of the
    /// > variance of a hypothetical infinite population. `ddof=0` provides a maximum
    /// > likelihood estimate of the variance for normally distributed variables. The
    /// > standard deviation computed in this function is the square root of the estimated
    /// > variance, so even with `ddof=1`, it will not be an unbiased estimate of the
    /// > standard deviation per se.
    ///
    /// Source: [Numpy](https://numpy.org/doc/stable/reference/generated/numpy.std.html#)
    pub fn std(self, ddof: u8) -> LazyFrame {
        self.select(vec![col("*").std(ddof)])
    }

    /// Aggregate all the columns as their variance values.
    ///
    /// `ddof` is the "Delta Degrees of Freedom"; `N - ddof` will be the denominator when
    /// computing the variance, where `N` is the number of rows.
    /// > In standard statistical practice, `ddof=1` provides an unbiased estimator of the
    /// > variance of a hypothetical infinite population. `ddof=0` provides a maximum
    /// > likelihood estimate of the variance for normally distributed variables.
    ///
    /// Source: [Numpy](https://numpy.org/doc/stable/reference/generated/numpy.var.html#)
    pub fn var(self, ddof: u8) -> LazyFrame {
        self.select(vec![col("*").var(ddof)])
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

    /// Aggregate all the columns as the sum of their null value count.
    pub fn null_count(self) -> LazyFrame {
        self.select(vec![col("*").null_count()])
    }

    /// Drop non-unique rows and maintain the order of kept rows.
    ///
    /// `subset` is an optional `Vec` of column names to consider for uniqueness; if
    /// `None`, all columns are considered.
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
            ..Default::default()
        };
        let lp = self.get_plan_builder().distinct(options).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Drop non-unique rows without maintaining the order of kept rows.
    ///
    /// The order of the kept rows may change; to maintain the original row order, use
    /// [`unique_stable`](LazyFrame::unique_stable).
    ///
    /// `subset` is an optional `Vec` of column names to consider for uniqueness; if None,
    /// all columns are considered.
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
            ..Default::default()
        };
        let lp = self.get_plan_builder().distinct(options).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Drop rows containing None.
    ///
    /// `subset` is an optional `Vec` of column names to consider for nulls; if None, all
    /// columns are considered.
    pub fn drop_nulls(self, subset: Option<Vec<Expr>>) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().drop_nulls(subset).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Slice the DataFrame using an offset (starting row) and a length.
    ///
    /// If `offset` is negative, it is counted from the end of the DataFrame. For
    /// instance, `lf.slice(-5, 3)` gets three rows, starting at the row fifth from the
    /// end.
    ///
    /// If `offset` and `len` are such that the slice extends beyond the end of the
    /// DataFrame, the portion between `offset` and the end will be returned. In this
    /// case, the number of rows in the returned DataFrame will be less than `len`.
    pub fn slice(self, offset: i64, len: IdxSize) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().slice(offset, len).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Get the first row.
    ///
    /// Equivalent to `self.slice(0, 1)`.
    pub fn first(self) -> LazyFrame {
        self.slice(0, 1)
    }

    /// Get the last row.
    ///
    /// Equivalent to `self.slice(-1, 1)`.
    pub fn last(self) -> LazyFrame {
        self.slice(-1, 1)
    }

    /// Get the last `n` rows.
    ///
    /// Equivalent to `self.slice(-(n as i64), n)`.
    pub fn tail(self, n: IdxSize) -> LazyFrame {
        let neg_tail = -(n as i64);
        self.slice(neg_tail, n)
    }

    /// Melt the DataFrame from wide to long format.
    ///
    /// See [`MeltArgs`] for information on how to melt a DataFrame.
    pub fn melt(self, args: MeltArgs) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().melt(Arc::new(args)).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Limit the DataFrame to the first `n` rows.
    ///
    /// Note if you don't want the rows to be scanned, use [`fetch`](LazyFrame::fetch).
    pub fn limit(self, n: IdxSize) -> LazyFrame {
        self.slice(0, n)
    }

    /// Apply a function/closure once the logical plan get executed.
    ///
    /// The function has access to the whole materialized DataFrame at the time it is
    /// called.
    ///
    /// To apply specific functions to specific columns, use [`Expr::map`] in conjunction
    /// with `LazyFrame::with_column` or `with_columns`.
    ///
    /// ## Warning
    /// This can blow up in your face if the schema is changed due to the operation. The
    /// optimizer relies on a correct schema.
    ///
    /// You can toggle certain optimizations off.
    pub fn map<F>(
        self,
        function: F,
        optimizations: AllowedOptimizations,
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
                optimizations,
                schema,
                name.unwrap_or("ANONYMOUS UDF"),
            )
            .build();
        Self::from_logical_plan(lp, opt_state)
    }

    #[cfg(feature = "python")]
    pub fn map_python(
        self,
        function: polars_plan::prelude::python_udf::PythonFunction,
        optimizations: AllowedOptimizations,
        schema: Option<SchemaRef>,
        validate_output: bool,
    ) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self
            .get_plan_builder()
            .map_python(function, optimizations, schema, validate_output)
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
    /// `name` is the name of the new column. `offset` is where to start counting from; if
    /// `None`, it is set to `0`.
    ///
    /// # Warning
    /// This can have a negative effect on query performance. This may for instance block
    /// predicate pushdown optimization.
    pub fn with_row_count(mut self, name: &str, offset: Option<IdxSize>) -> LazyFrame {
        let add_row_count_in_map = match &mut self.logical_plan {
            LogicalPlan::Scan {
                file_options: options,
                file_info,
                scan_type,
                ..
            } if !matches!(scan_type, FileScan::Anonymous { .. }) => {
                options.row_count = Some(RowCount {
                    name: name.to_string(),
                    offset: offset.unwrap_or(0),
                });
                file_info.schema = Arc::new(
                    file_info
                        .schema
                        .new_inserting_at_index(0, name.into(), IDX_DTYPE)
                        .unwrap(),
                );
                false
            },
            _ => true,
        };

        if add_row_count_in_map {
            let schema = fallible!(self.schema(), &self);
            let schema = schema
                .new_inserting_at_index(0, name.into(), IDX_DTYPE)
                .unwrap();

            self.map_private(FunctionNode::RowCount {
                name: Arc::from(name),
                offset,
                schema: Arc::new(schema),
            })
        } else {
            self
        }
    }

    /// Unnest the given `Struct` columns: the fields of the `Struct` type will be
    /// inserted as columns.
    #[cfg(feature = "dtype-struct")]
    pub fn unnest<I: IntoIterator<Item = S>, S: AsRef<str>>(self, cols: I) -> Self {
        self.map_private(FunctionNode::Unnest {
            columns: cols.into_iter().map(|s| Arc::from(s.as_ref())).collect(),
        })
    }

    #[cfg(feature = "merge_sorted")]
    pub fn merge_sorted(self, other: LazyFrame, key: &str) -> PolarsResult<LazyFrame> {
        // The two DataFrames are temporary concatenated
        // this indicates until which chunk the data is from the left df
        // this trick allows us to reuse the `Union` architecture to get map over
        // two DataFrames
        let left = self.map_private(FunctionNode::Rechunk);
        let q = concat(
            &[left, other],
            UnionArgs {
                rechunk: false,
                parallel: true,
                ..Default::default()
            },
        )?;
        Ok(q.map_private(FunctionNode::MergeSorted {
            column: Arc::from(key),
        }))
    }
}

/// Utility struct for lazy group_by operation.
#[derive(Clone)]
pub struct LazyGroupBy {
    pub logical_plan: LogicalPlan,
    opt_state: OptState,
    keys: Vec<Expr>,
    maintain_order: bool,
    #[cfg(feature = "dynamic_group_by")]
    dynamic_options: Option<DynamicGroupOptions>,
    #[cfg(feature = "dynamic_group_by")]
    rolling_options: Option<RollingGroupOptions>,
}

impl LazyGroupBy {
    /// Group by and aggregate.
    ///
    /// Select a column with [col] and choose an aggregation.
    /// If you want to aggregate all columns use `col("*")`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    /// use arrow::legacy::prelude::QuantileInterpolOptions;
    ///
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///        .group_by_stable([col("date")])
    ///        .agg([
    ///            col("rain").min().alias("min_rain"),
    ///            col("rain").sum().alias("sum_rain"),
    ///            col("rain").quantile(lit(0.5), QuantileInterpolOptions::Nearest).alias("median_rain"),
    ///        ])
    /// }
    /// ```
    pub fn agg<E: AsRef<[Expr]>>(self, aggs: E) -> LazyFrame {
        #[cfg(feature = "dynamic_group_by")]
        let lp = LogicalPlanBuilder::from(self.logical_plan)
            .group_by(
                self.keys,
                aggs,
                None,
                self.maintain_order,
                self.dynamic_options,
                self.rolling_options,
            )
            .build();

        #[cfg(not(feature = "dynamic_group_by"))]
        let lp = LogicalPlanBuilder::from(self.logical_plan)
            .group_by(self.keys, aggs, None, self.maintain_order)
            .build();
        LazyFrame::from_logical_plan(lp, self.opt_state)
    }

    /// Return first n rows of each group
    pub fn head(self, n: Option<usize>) -> LazyFrame {
        let keys = self
            .keys
            .iter()
            .filter_map(|expr| expr_output_name(expr).ok())
            .collect::<Vec<_>>();

        self.agg([col("*").exclude(&keys).head(n)])
            .explode([col("*").exclude(&keys)])
    }

    /// Return last n rows of each group
    pub fn tail(self, n: Option<usize>) -> LazyFrame {
        let keys = self
            .keys
            .iter()
            .filter_map(|expr| expr_output_name(expr).ok())
            .collect::<Vec<_>>();

        self.agg([col("*").exclude(&keys).tail(n)])
            .explode([col("*").exclude(&keys)])
    }

    /// Apply a function over the groups as a new DataFrame.
    ///
    /// **It is not recommended that you use this as materializing the DataFrame is very
    /// expensive.**
    pub fn apply<F>(self, f: F, schema: SchemaRef) -> LazyFrame
    where
        F: 'static + Fn(DataFrame) -> PolarsResult<DataFrame> + Send + Sync,
    {
        #[cfg(feature = "dynamic_group_by")]
        let options = GroupbyOptions {
            dynamic: self.dynamic_options,
            rolling: self.rolling_options,
            slice: None,
        };

        #[cfg(not(feature = "dynamic_group_by"))]
        let options = GroupbyOptions { slice: None };

        let lp = LogicalPlan::Aggregate {
            input: Box::new(self.logical_plan),
            keys: Arc::new(self.keys),
            aggs: vec![],
            schema,
            apply: Some(Arc::new(f)),
            maintain_order: self.maintain_order,
            options: Arc::new(options),
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
    validation: JoinValidation,
}
impl JoinBuilder {
    /// Create the `JoinBuilder` with the provided `LazyFrame` as the left table.
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
            validation: Default::default(),
        }
    }

    /// The right table in the join.
    pub fn with(mut self, other: LazyFrame) -> Self {
        self.other = Some(other);
        self
    }

    /// Select the join type.
    pub fn how(mut self, how: JoinType) -> Self {
        self.how = how;
        self
    }

    pub fn validate(mut self, validation: JoinValidation) -> Self {
        self.validation = validation;
        self
    }

    /// The expressions you want to join both tables on.
    ///
    /// The passed expressions must be valid in both `LazyFrame`s in the join.
    pub fn on<E: AsRef<[Expr]>>(mut self, on: E) -> Self {
        let on = on.as_ref().to_vec();
        self.left_on = on.clone();
        self.right_on = on;
        self
    }

    /// The expressions you want to join the left table on.
    ///
    /// The passed expressions must be valid in the left table.
    pub fn left_on<E: AsRef<[Expr]>>(mut self, on: E) -> Self {
        self.left_on = on.as_ref().to_vec();
        self
    }

    /// The expressions you want to join the right table on.
    ///
    /// The passed expressions must be valid in the right table.
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
    pub fn force_parallel(mut self, force: bool) -> Self {
        self.force_parallel = force;
        self
    }

    /// Suffix to add duplicate column names in join.
    /// Defaults to `"_right"` if this method is never called.
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

        let args = JoinArgs {
            how: self.how,
            validation: self.validation,
            suffix: self.suffix,
            slice: None,
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
                    args,
                    ..Default::default()
                }
                .into(),
            )
            .build();
        LazyFrame::from_logical_plan(lp, opt_state)
    }
}
