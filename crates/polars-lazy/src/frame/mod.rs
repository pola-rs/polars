//! Lazy variant of a [DataFrame].
#[cfg(feature = "python")]
mod python;

mod cached_arenas;
mod err;
#[cfg(not(target_arch = "wasm32"))]
mod exitable;
#[cfg(feature = "pivot")]
pub mod pivot;

use std::sync::{Arc, Mutex};

pub use anonymous_scan::*;
#[cfg(feature = "csv")]
pub use csv::*;
#[cfg(not(target_arch = "wasm32"))]
pub use exitable::*;
pub use file_list_reader::*;
#[cfg(feature = "ipc")]
pub use ipc::*;
#[cfg(feature = "json")]
pub use ndjson::*;
#[cfg(feature = "parquet")]
pub use parquet::*;
use polars_compute::rolling::QuantileMethod;
use polars_core::POOL;
use polars_core::error::feature_gated;
use polars_core::prelude::*;
use polars_expr::{ExpressionConversionState, create_physical_expr};
use polars_io::RowIndex;
use polars_mem_engine::{Executor, create_multiple_physical_plans, create_physical_plan};
use polars_ops::frame::{JoinCoalesce, MaintainOrderJoin};
#[cfg(feature = "is_between")]
use polars_ops::prelude::ClosedInterval;
pub use polars_plan::frame::{AllowedOptimizations, OptFlags};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::plpath::PlPath;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::frame::cached_arenas::CachedArena;
use crate::prelude::*;

pub trait IntoLazy {
    fn lazy(self) -> LazyFrame;
}

impl IntoLazy for DataFrame {
    /// Convert the `DataFrame` into a `LazyFrame`
    fn lazy(self) -> LazyFrame {
        let lp = DslBuilder::from_existing_df(self).build();
        LazyFrame {
            logical_plan: lp,
            opt_state: Default::default(),
            cached_arena: Default::default(),
        }
    }
}

impl IntoLazy for LazyFrame {
    fn lazy(self) -> LazyFrame {
        self
    }
}

/// Lazy abstraction over an eager `DataFrame`.
///
/// It really is an abstraction over a logical plan. The methods of this struct will incrementally
/// modify a logical plan until output is requested (via [`collect`](crate::frame::LazyFrame::collect)).
#[derive(Clone, Default)]
#[must_use]
pub struct LazyFrame {
    pub logical_plan: DslPlan,
    pub(crate) opt_state: OptFlags,
    pub(crate) cached_arena: Arc<Mutex<Option<CachedArena>>>,
}

impl From<DslPlan> for LazyFrame {
    fn from(plan: DslPlan) -> Self {
        Self {
            logical_plan: plan,
            opt_state: OptFlags::default(),
            cached_arena: Default::default(),
        }
    }
}

impl LazyFrame {
    pub(crate) fn from_inner(
        logical_plan: DslPlan,
        opt_state: OptFlags,
        cached_arena: Arc<Mutex<Option<CachedArena>>>,
    ) -> Self {
        Self {
            logical_plan,
            opt_state,
            cached_arena,
        }
    }

    pub(crate) fn get_plan_builder(self) -> DslBuilder {
        DslBuilder::from(self.logical_plan)
    }

    fn get_opt_state(&self) -> OptFlags {
        self.opt_state
    }

    fn from_logical_plan(logical_plan: DslPlan, opt_state: OptFlags) -> Self {
        LazyFrame {
            logical_plan,
            opt_state,
            cached_arena: Default::default(),
        }
    }

    /// Get current optimizations.
    pub fn get_current_optimizations(&self) -> OptFlags {
        self.opt_state
    }

    /// Set allowed optimizations.
    pub fn with_optimizations(mut self, opt_state: OptFlags) -> Self {
        self.opt_state = opt_state;
        self
    }

    /// Turn off all optimizations.
    pub fn without_optimizations(self) -> Self {
        self.with_optimizations(OptFlags::from_bits_truncate(0) | OptFlags::TYPE_COERCION)
    }

    /// Toggle projection pushdown optimization.
    pub fn with_projection_pushdown(mut self, toggle: bool) -> Self {
        self.opt_state.set(OptFlags::PROJECTION_PUSHDOWN, toggle);
        self
    }

    /// Toggle cluster with columns optimization.
    pub fn with_cluster_with_columns(mut self, toggle: bool) -> Self {
        self.opt_state.set(OptFlags::CLUSTER_WITH_COLUMNS, toggle);
        self
    }

    /// Toggle collapse joins optimization.
    pub fn with_collapse_joins(mut self, toggle: bool) -> Self {
        self.opt_state.set(OptFlags::COLLAPSE_JOINS, toggle);
        self
    }

    /// Check if operations are order dependent and unset maintaining_order if
    /// the order would not be observed.
    pub fn with_check_order(mut self, toggle: bool) -> Self {
        self.opt_state.set(OptFlags::CHECK_ORDER_OBSERVE, toggle);
        self
    }

    /// Toggle predicate pushdown optimization.
    pub fn with_predicate_pushdown(mut self, toggle: bool) -> Self {
        self.opt_state.set(OptFlags::PREDICATE_PUSHDOWN, toggle);
        self
    }

    /// Toggle type coercion optimization.
    pub fn with_type_coercion(mut self, toggle: bool) -> Self {
        self.opt_state.set(OptFlags::TYPE_COERCION, toggle);
        self
    }

    /// Toggle type check optimization.
    pub fn with_type_check(mut self, toggle: bool) -> Self {
        self.opt_state.set(OptFlags::TYPE_CHECK, toggle);
        self
    }

    /// Toggle expression simplification optimization on or off.
    pub fn with_simplify_expr(mut self, toggle: bool) -> Self {
        self.opt_state.set(OptFlags::SIMPLIFY_EXPR, toggle);
        self
    }

    /// Toggle common subplan elimination optimization on or off
    #[cfg(feature = "cse")]
    pub fn with_comm_subplan_elim(mut self, toggle: bool) -> Self {
        self.opt_state.set(OptFlags::COMM_SUBPLAN_ELIM, toggle);
        self
    }

    /// Toggle common subexpression elimination optimization on or off
    #[cfg(feature = "cse")]
    pub fn with_comm_subexpr_elim(mut self, toggle: bool) -> Self {
        self.opt_state.set(OptFlags::COMM_SUBEXPR_ELIM, toggle);
        self
    }

    /// Toggle slice pushdown optimization.
    pub fn with_slice_pushdown(mut self, toggle: bool) -> Self {
        self.opt_state.set(OptFlags::SLICE_PUSHDOWN, toggle);
        self
    }

    #[cfg(feature = "new_streaming")]
    pub fn with_new_streaming(mut self, toggle: bool) -> Self {
        self.opt_state.set(OptFlags::NEW_STREAMING, toggle);
        self
    }

    /// Try to estimate the number of rows so that joins can determine which side to keep in memory.
    pub fn with_row_estimate(mut self, toggle: bool) -> Self {
        self.opt_state.set(OptFlags::ROW_ESTIMATE, toggle);
        self
    }

    /// Run every node eagerly. This turns off multi-node optimizations.
    pub fn _with_eager(mut self, toggle: bool) -> Self {
        self.opt_state.set(OptFlags::EAGER, toggle);
        self
    }

    /// Return a String describing the naive (un-optimized) logical plan.
    pub fn describe_plan(&self) -> PolarsResult<String> {
        Ok(self.clone().to_alp()?.describe())
    }

    /// Return a String describing the naive (un-optimized) logical plan in tree format.
    pub fn describe_plan_tree(&self) -> PolarsResult<String> {
        Ok(self.clone().to_alp()?.describe_tree_format())
    }

    /// Return a String describing the optimized logical plan.
    ///
    /// Returns `Err` if optimizing the logical plan fails.
    pub fn describe_optimized_plan(&self) -> PolarsResult<String> {
        Ok(self.clone().to_alp_optimized()?.describe())
    }

    /// Return a String describing the optimized logical plan in tree format.
    ///
    /// Returns `Err` if optimizing the logical plan fails.
    pub fn describe_optimized_plan_tree(&self) -> PolarsResult<String> {
        Ok(self.clone().to_alp_optimized()?.describe_tree_format())
    }

    /// Return a String describing the logical plan.
    ///
    /// If `optimized` is `true`, explains the optimized plan. If `optimized` is `false`,
    /// explains the naive, un-optimized plan.
    pub fn explain(&self, optimized: bool) -> PolarsResult<String> {
        if optimized {
            self.describe_optimized_plan()
        } else {
            self.describe_plan()
        }
    }

    /// Add a sort operation to the logical plan.
    ///
    /// Sorts the LazyFrame by the column name specified using the provided options.
    ///
    /// # Example
    ///
    /// Sort DataFrame by 'sepal_width' column:
    /// ```rust
    /// # use polars_core::prelude::*;
    /// # use polars_lazy::prelude::*;
    /// fn sort_by_a(df: DataFrame) -> LazyFrame {
    ///     df.lazy().sort(["sepal_width"], Default::default())
    /// }
    /// ```
    /// Sort by a single column with specific order:
    /// ```
    /// # use polars_core::prelude::*;
    /// # use polars_lazy::prelude::*;
    /// fn sort_with_specific_order(df: DataFrame, descending: bool) -> LazyFrame {
    ///     df.lazy().sort(
    ///         ["sepal_width"],
    ///         SortMultipleOptions::new()
    ///             .with_order_descending(descending)
    ///     )
    /// }
    /// ```
    /// Sort by multiple columns with specifying order for each column:
    /// ```
    /// # use polars_core::prelude::*;
    /// # use polars_lazy::prelude::*;
    /// fn sort_by_multiple_columns_with_specific_order(df: DataFrame) -> LazyFrame {
    ///     df.lazy().sort(
    ///         ["sepal_width", "sepal_length"],
    ///         SortMultipleOptions::new()
    ///             .with_order_descending_multi([false, true])
    ///     )
    /// }
    /// ```
    /// See [`SortMultipleOptions`] for more options.
    pub fn sort(self, by: impl IntoVec<PlSmallStr>, sort_options: SortMultipleOptions) -> Self {
        let opt_state = self.get_opt_state();
        let lp = self
            .get_plan_builder()
            .sort(by.into_vec().into_iter().map(col).collect(), sort_options)
            .build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Add a sort operation to the logical plan.
    ///
    /// Sorts the LazyFrame by the provided list of expressions, which will be turned into
    /// concrete columns before sorting.
    ///
    /// See [`SortMultipleOptions`] for more options.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    ///
    /// /// Sort DataFrame by 'sepal_width' column
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///         .sort_by_exprs(vec![col("sepal_width")], Default::default())
    /// }
    /// ```
    pub fn sort_by_exprs<E: AsRef<[Expr]>>(
        self,
        by_exprs: E,
        sort_options: SortMultipleOptions,
    ) -> Self {
        let by_exprs = by_exprs.as_ref().to_vec();
        if by_exprs.is_empty() {
            self
        } else {
            let opt_state = self.get_opt_state();
            let lp = self.get_plan_builder().sort(by_exprs, sort_options).build();
            Self::from_logical_plan(lp, opt_state)
        }
    }

    pub fn top_k<E: AsRef<[Expr]>>(
        self,
        k: IdxSize,
        by_exprs: E,
        sort_options: SortMultipleOptions,
    ) -> Self {
        // this will optimize to top-k
        self.sort_by_exprs(
            by_exprs,
            sort_options.with_order_reversed().with_nulls_last(true),
        )
        .slice(0, k)
    }

    pub fn bottom_k<E: AsRef<[Expr]>>(
        self,
        k: IdxSize,
        by_exprs: E,
        sort_options: SortMultipleOptions,
    ) -> Self {
        // this will optimize to bottom-k
        self.sort_by_exprs(by_exprs, sort_options.with_nulls_last(true))
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
        self.select(vec![col(PlSmallStr::from_static("*")).reverse()])
    }

    /// Rename columns in the DataFrame.
    ///
    /// `existing` and `new` are iterables of the same length containing the old and
    /// corresponding new column names. Renaming happens to all `existing` columns
    /// simultaneously, not iteratively. If `strict` is true, all columns in `existing`
    /// must be present in the `LazyFrame` when `rename` is called; otherwise, only
    /// those columns that are actually found will be renamed (others will be ignored).
    pub fn rename<I, J, T, S>(self, existing: I, new: J, strict: bool) -> Self
    where
        I: IntoIterator<Item = T>,
        J: IntoIterator<Item = S>,
        T: AsRef<str>,
        S: AsRef<str>,
    {
        let iter = existing.into_iter();
        let cap = iter.size_hint().0;
        let mut existing_vec: Vec<PlSmallStr> = Vec::with_capacity(cap);
        let mut new_vec: Vec<PlSmallStr> = Vec::with_capacity(cap);

        // TODO! should this error if `existing` and `new` have different lengths?
        // Currently, the longer of the two is truncated.
        for (existing, new) in iter.zip(new) {
            let existing = existing.as_ref();
            let new = new.as_ref();
            if new != existing {
                existing_vec.push(existing.into());
                new_vec.push(new.into());
            }
        }

        self.map_private(DslFunction::Rename {
            existing: existing_vec.into(),
            new: new_vec.into(),
            strict,
        })
    }

    /// Removes columns from the DataFrame.
    /// Note that it's better to only select the columns you need
    /// and let the projection pushdown optimize away the unneeded columns.
    ///
    /// Any given columns that are not in the schema will give a [`PolarsError::ColumnNotFound`]
    /// error while materializing the [`LazyFrame`].
    pub fn drop(self, columns: Selector) -> Self {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().drop(columns).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with `Nones`.
    ///
    /// See the method on [Series](polars_core::series::SeriesTrait::shift) for more info on the `shift` operation.
    pub fn shift<E: Into<Expr>>(self, n: E) -> Self {
        self.select(vec![col(PlSmallStr::from_static("*")).shift(n.into())])
    }

    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with the result of the `fill_value` expression.
    ///
    /// See the method on [Series](polars_core::series::SeriesTrait::shift) for more info on the `shift` operation.
    pub fn shift_and_fill<E: Into<Expr>, IE: Into<Expr>>(self, n: E, fill_value: IE) -> Self {
        self.select(vec![
            col(PlSmallStr::from_static("*")).shift_and_fill(n.into(), fill_value.into()),
        ])
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
                let name = PlSmallStr::from_str(name);

                if strict {
                    col(name).strict_cast(dt)
                } else {
                    col(name).cast(dt)
                }
            })
            .collect();

        if cast_cols.is_empty() {
            self
        } else {
            self.with_columns(cast_cols)
        }
    }

    /// Cast all frame columns to the given dtype, resulting in a new LazyFrame
    pub fn cast_all(self, dtype: impl Into<DataTypeExpr>, strict: bool) -> Self {
        self.with_columns(vec![if strict {
            col(PlSmallStr::from_static("*")).strict_cast(dtype)
        } else {
            col(PlSmallStr::from_static("*")).cast(dtype)
        }])
    }

    pub fn optimize(
        self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<Node> {
        self.optimize_with_scratch(lp_arena, expr_arena, &mut vec![])
    }

    pub fn to_alp_optimized(mut self) -> PolarsResult<IRPlan> {
        let (mut lp_arena, mut expr_arena) = self.get_arenas();
        let node = self.optimize_with_scratch(&mut lp_arena, &mut expr_arena, &mut vec![])?;

        Ok(IRPlan::new(node, lp_arena, expr_arena))
    }

    pub fn to_alp(mut self) -> PolarsResult<IRPlan> {
        let (mut lp_arena, mut expr_arena) = self.get_arenas();
        let node = to_alp(
            self.logical_plan,
            &mut expr_arena,
            &mut lp_arena,
            &mut self.opt_state,
        )?;
        let plan = IRPlan::new(node, lp_arena, expr_arena);
        Ok(plan)
    }

    pub(crate) fn optimize_with_scratch(
        self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        scratch: &mut Vec<Node>,
    ) -> PolarsResult<Node> {
        #[allow(unused_mut)]
        let mut opt_state = self.opt_state;
        let new_streaming = self.opt_state.contains(OptFlags::NEW_STREAMING);

        #[cfg(feature = "cse")]
        if new_streaming {
            // The new streaming engine can't deal with the way the common
            // subexpression elimination adds length-incorrect with_columns.
            opt_state &= !OptFlags::COMM_SUBEXPR_ELIM;
        }

        let lp_top = optimize(
            self.logical_plan,
            opt_state,
            lp_arena,
            expr_arena,
            scratch,
            Some(&|expr, expr_arena, schema| {
                let phys_expr = create_physical_expr(
                    expr,
                    Context::Default,
                    expr_arena,
                    schema,
                    &mut ExpressionConversionState::new(true),
                )
                .ok()?;
                let io_expr = phys_expr_to_io_expr(phys_expr);
                Some(io_expr)
            }),
        )?;

        Ok(lp_top)
    }

    fn prepare_collect_post_opt<P>(
        mut self,
        check_sink: bool,
        query_start: Option<std::time::Instant>,
        post_opt: P,
    ) -> PolarsResult<(ExecutionState, Box<dyn Executor>, bool)>
    where
        P: FnOnce(
            Node,
            &mut Arena<IR>,
            &mut Arena<AExpr>,
            Option<std::time::Duration>,
        ) -> PolarsResult<()>,
    {
        let (mut lp_arena, mut expr_arena) = self.get_arenas();

        let mut scratch = vec![];
        let lp_top = self.optimize_with_scratch(&mut lp_arena, &mut expr_arena, &mut scratch)?;

        post_opt(
            lp_top,
            &mut lp_arena,
            &mut expr_arena,
            // Post optimization callback gets the time since the
            // query was started as its "base" timepoint.
            query_start.map(|s| s.elapsed()),
        )?;

        // sink should be replaced
        let no_file_sink = if check_sink {
            !matches!(
                lp_arena.get(lp_top),
                IR::Sink {
                    payload: SinkTypeIR::File { .. } | SinkTypeIR::Partition { .. },
                    ..
                }
            )
        } else {
            true
        };
        let physical_plan = create_physical_plan(
            lp_top,
            &mut lp_arena,
            &mut expr_arena,
            BUILD_STREAMING_EXECUTOR,
        )?;

        let state = ExecutionState::new();
        Ok((state, physical_plan, no_file_sink))
    }

    // post_opt: A function that is called after optimization. This can be used to modify the IR jit.
    pub fn _collect_post_opt<P>(self, post_opt: P) -> PolarsResult<DataFrame>
    where
        P: FnOnce(
            Node,
            &mut Arena<IR>,
            &mut Arena<AExpr>,
            Option<std::time::Duration>,
        ) -> PolarsResult<()>,
    {
        let (mut state, mut physical_plan, _) =
            self.prepare_collect_post_opt(false, None, post_opt)?;
        physical_plan.execute(&mut state)
    }

    #[allow(unused_mut)]
    fn prepare_collect(
        self,
        check_sink: bool,
        query_start: Option<std::time::Instant>,
    ) -> PolarsResult<(ExecutionState, Box<dyn Executor>, bool)> {
        self.prepare_collect_post_opt(check_sink, query_start, |_, _, _, _| Ok(()))
    }

    /// Execute all the lazy operations and collect them into a [`DataFrame`] using a specified
    /// `engine`.
    ///
    /// The query is optimized prior to execution.
    pub fn collect_with_engine(mut self, mut engine: Engine) -> PolarsResult<DataFrame> {
        let payload = if let DslPlan::Sink { payload, .. } = &self.logical_plan {
            payload.clone()
        } else {
            self.logical_plan = DslPlan::Sink {
                input: Arc::new(self.logical_plan),
                payload: SinkType::Memory,
            };
            SinkType::Memory
        };

        // Default engine for collect is InMemory, sink_* is Streaming
        if engine == Engine::Auto {
            engine = match payload {
                #[cfg(feature = "new_streaming")]
                SinkType::File { .. } | SinkType::Partition { .. } => Engine::Streaming,
                _ => Engine::InMemory,
            };
        }
        // Gpu uses some hacks to dispatch.
        if engine == Engine::Gpu {
            engine = Engine::InMemory;
        }

        #[cfg(feature = "new_streaming")]
        {
            if let Some(result) = self.try_new_streaming_if_requested() {
                return result.map(|v| v.unwrap_single());
            }
        }

        match engine {
            Engine::Auto => unreachable!(),
            Engine::Streaming => {
                feature_gated!("new_streaming", self = self.with_new_streaming(true))
            },
            _ => {},
        }
        let mut alp_plan = self.clone().to_alp_optimized()?;

        match engine {
            Engine::Auto | Engine::Streaming => feature_gated!("new_streaming", {
                let result = polars_stream::run_query(
                    alp_plan.lp_top,
                    &mut alp_plan.lp_arena,
                    &mut alp_plan.expr_arena,
                );
                result.map(|v| v.unwrap_single())
            }),
            Engine::Gpu => {
                Err(polars_err!(InvalidOperation: "sink is not supported for the gpu engine"))
            },
            Engine::InMemory => {
                let mut physical_plan = create_physical_plan(
                    alp_plan.lp_top,
                    &mut alp_plan.lp_arena,
                    &mut alp_plan.expr_arena,
                    BUILD_STREAMING_EXECUTOR,
                )?;
                let mut state = ExecutionState::new();
                physical_plan.execute(&mut state)
            },
        }
    }

    pub fn explain_all(plans: Vec<DslPlan>, opt_state: OptFlags) -> PolarsResult<String> {
        let sink_multiple = LazyFrame {
            logical_plan: DslPlan::SinkMultiple { inputs: plans },
            opt_state,
            cached_arena: Default::default(),
        };
        sink_multiple.explain(true)
    }

    pub fn collect_all_with_engine(
        plans: Vec<DslPlan>,
        mut engine: Engine,
        opt_state: OptFlags,
    ) -> PolarsResult<Vec<DataFrame>> {
        if plans.is_empty() {
            return Ok(Vec::new());
        }

        // Default engine for collect_all is InMemory
        if engine == Engine::Auto {
            engine = Engine::InMemory;
        }
        // Gpu uses some hacks to dispatch.
        if engine == Engine::Gpu {
            engine = Engine::InMemory;
        }

        let mut sink_multiple = LazyFrame {
            logical_plan: DslPlan::SinkMultiple { inputs: plans },
            opt_state,
            cached_arena: Default::default(),
        };

        #[cfg(feature = "new_streaming")]
        {
            if let Some(result) = sink_multiple.try_new_streaming_if_requested() {
                return result.map(|v| v.unwrap_multiple());
            }
        }

        match engine {
            Engine::Auto => unreachable!(),
            Engine::Streaming => {
                feature_gated!(
                    "new_streaming",
                    sink_multiple = sink_multiple.with_new_streaming(true)
                )
            },
            _ => {},
        }
        let mut alp_plan = sink_multiple.to_alp_optimized()?;

        if engine == Engine::Streaming {
            feature_gated!("new_streaming", {
                let result = polars_stream::run_query(
                    alp_plan.lp_top,
                    &mut alp_plan.lp_arena,
                    &mut alp_plan.expr_arena,
                );
                return result.map(|v| v.unwrap_multiple());
            });
        }

        let IR::SinkMultiple { inputs } = alp_plan.root() else {
            unreachable!()
        };

        let mut multiplan = create_multiple_physical_plans(
            inputs.clone().as_slice(),
            &mut alp_plan.lp_arena,
            &mut alp_plan.expr_arena,
            BUILD_STREAMING_EXECUTOR,
        )?;

        match engine {
            Engine::Gpu => polars_bail!(
                InvalidOperation: "collect_all is not supported for the gpu engine"
            ),
            Engine::InMemory => {
                // We don't use par_iter directly because the LP may also start threads for every LP (for instance scan_csv)
                // this might then lead to a rayon SO. So we take a multitude of the threads to keep work stealing
                // within bounds
                let mut state = ExecutionState::new();
                if let Some(mut cache_prefiller) = multiplan.cache_prefiller {
                    cache_prefiller.execute(&mut state)?;
                }
                let out = POOL.install(|| {
                    multiplan
                        .physical_plans
                        .chunks_mut(POOL.current_num_threads() * 3)
                        .map(|chunk| {
                            chunk
                                .into_par_iter()
                                .enumerate()
                                .map(|(idx, input)| {
                                    let mut input = std::mem::take(input);
                                    let mut state = state.split();
                                    state.branch_idx += idx;

                                    let df = input.execute(&mut state)?;
                                    Ok(df)
                                })
                                .collect::<PolarsResult<Vec<_>>>()
                        })
                        .collect::<PolarsResult<Vec<_>>>()
                });
                Ok(out?.into_iter().flatten().collect())
            },
            _ => unreachable!(),
        }
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
        self.collect_with_engine(Engine::InMemory)
    }

    // post_opt: A function that is called after optimization. This can be used to modify the IR jit.
    // This version does profiling of the node execution.
    pub fn _profile_post_opt<P>(self, post_opt: P) -> PolarsResult<(DataFrame, DataFrame)>
    where
        P: FnOnce(
            Node,
            &mut Arena<IR>,
            &mut Arena<AExpr>,
            Option<std::time::Duration>,
        ) -> PolarsResult<()>,
    {
        let query_start = std::time::Instant::now();
        let (mut state, mut physical_plan, _) =
            self.prepare_collect_post_opt(false, Some(query_start), post_opt)?;
        state.time_nodes(query_start);
        let out = physical_plan.execute(&mut state)?;
        let timer_df = state.finish_timer()?;
        Ok((out, timer_df))
    }

    /// Profile a LazyFrame.
    ///
    /// This will run the query and return a tuple
    /// containing the materialized DataFrame and a DataFrame that contains profiling information
    /// of each node that is executed.
    ///
    /// The units of the timings are microseconds.
    pub fn profile(self) -> PolarsResult<(DataFrame, DataFrame)> {
        self._profile_post_opt(|_, _, _, _| Ok(()))
    }

    /// Stream a query result into a parquet file. This is useful if the final result doesn't fit
    /// into memory. This methods will return an error if the query cannot be completely done in a
    /// streaming fashion.
    #[cfg(feature = "parquet")]
    pub fn sink_parquet(
        self,
        target: SinkTarget,
        options: ParquetWriteOptions,
        cloud_options: Option<polars_io::cloud::CloudOptions>,
        sink_options: SinkOptions,
    ) -> PolarsResult<Self> {
        self.sink(SinkType::File(FileSinkType {
            target,
            sink_options,
            file_type: FileType::Parquet(options),
            cloud_options,
        }))
    }

    /// Stream a query result into an ipc/arrow file. This is useful if the final result doesn't fit
    /// into memory. This methods will return an error if the query cannot be completely done in a
    /// streaming fashion.
    #[cfg(feature = "ipc")]
    pub fn sink_ipc(
        self,
        target: SinkTarget,
        options: IpcWriterOptions,
        cloud_options: Option<polars_io::cloud::CloudOptions>,
        sink_options: SinkOptions,
    ) -> PolarsResult<Self> {
        self.sink(SinkType::File(FileSinkType {
            target,
            sink_options,
            file_type: FileType::Ipc(options),
            cloud_options,
        }))
    }

    /// Stream a query result into an csv file. This is useful if the final result doesn't fit
    /// into memory. This methods will return an error if the query cannot be completely done in a
    /// streaming fashion.
    #[cfg(feature = "csv")]
    pub fn sink_csv(
        self,
        target: SinkTarget,
        options: CsvWriterOptions,
        cloud_options: Option<polars_io::cloud::CloudOptions>,
        sink_options: SinkOptions,
    ) -> PolarsResult<Self> {
        self.sink(SinkType::File(FileSinkType {
            target,
            sink_options,
            file_type: FileType::Csv(options),
            cloud_options,
        }))
    }

    /// Stream a query result into a JSON file. This is useful if the final result doesn't fit
    /// into memory. This methods will return an error if the query cannot be completely done in a
    /// streaming fashion.
    #[cfg(feature = "json")]
    pub fn sink_json(
        self,
        target: SinkTarget,
        options: JsonWriterOptions,
        cloud_options: Option<polars_io::cloud::CloudOptions>,
        sink_options: SinkOptions,
    ) -> PolarsResult<Self> {
        self.sink(SinkType::File(FileSinkType {
            target,
            sink_options,
            file_type: FileType::Json(options),
            cloud_options,
        }))
    }

    /// Stream a query result into a parquet file in a partitioned manner. This is useful if the
    /// final result doesn't fit into memory. This methods will return an error if the query cannot
    /// be completely done in a streaming fashion.
    #[cfg(feature = "parquet")]
    #[allow(clippy::too_many_arguments)]
    pub fn sink_parquet_partitioned(
        self,
        base_path: Arc<PlPath>,
        file_path_cb: Option<PartitionTargetCallback>,
        variant: PartitionVariant,
        options: ParquetWriteOptions,
        cloud_options: Option<polars_io::cloud::CloudOptions>,
        sink_options: SinkOptions,
        per_partition_sort_by: Option<Vec<SortColumn>>,
        finish_callback: Option<SinkFinishCallback>,
    ) -> PolarsResult<Self> {
        self.sink(SinkType::Partition(PartitionSinkType {
            base_path,
            file_path_cb,
            sink_options,
            variant,
            file_type: FileType::Parquet(options),
            cloud_options,
            per_partition_sort_by,
            finish_callback,
        }))
    }

    /// Stream a query result into an ipc/arrow file in a partitioned manner. This is useful if the
    /// final result doesn't fit into memory. This methods will return an error if the query cannot
    /// be completely done in a streaming fashion.
    #[cfg(feature = "ipc")]
    #[allow(clippy::too_many_arguments)]
    pub fn sink_ipc_partitioned(
        self,
        base_path: Arc<PlPath>,
        file_path_cb: Option<PartitionTargetCallback>,
        variant: PartitionVariant,
        options: IpcWriterOptions,
        cloud_options: Option<polars_io::cloud::CloudOptions>,
        sink_options: SinkOptions,
        per_partition_sort_by: Option<Vec<SortColumn>>,
        finish_callback: Option<SinkFinishCallback>,
    ) -> PolarsResult<Self> {
        self.sink(SinkType::Partition(PartitionSinkType {
            base_path,
            file_path_cb,
            sink_options,
            variant,
            file_type: FileType::Ipc(options),
            cloud_options,
            per_partition_sort_by,
            finish_callback,
        }))
    }

    /// Stream a query result into an csv file in a partitioned manner. This is useful if the final
    /// result doesn't fit into memory. This methods will return an error if the query cannot be
    /// completely done in a streaming fashion.
    #[cfg(feature = "csv")]
    #[allow(clippy::too_many_arguments)]
    pub fn sink_csv_partitioned(
        self,
        base_path: Arc<PlPath>,
        file_path_cb: Option<PartitionTargetCallback>,
        variant: PartitionVariant,
        options: CsvWriterOptions,
        cloud_options: Option<polars_io::cloud::CloudOptions>,
        sink_options: SinkOptions,
        per_partition_sort_by: Option<Vec<SortColumn>>,
        finish_callback: Option<SinkFinishCallback>,
    ) -> PolarsResult<Self> {
        self.sink(SinkType::Partition(PartitionSinkType {
            base_path,
            file_path_cb,
            sink_options,
            variant,
            file_type: FileType::Csv(options),
            cloud_options,
            per_partition_sort_by,
            finish_callback,
        }))
    }

    /// Stream a query result into a JSON file in a partitioned manner. This is useful if the final
    /// result doesn't fit into memory. This methods will return an error if the query cannot be
    /// completely done in a streaming fashion.
    #[cfg(feature = "json")]
    #[allow(clippy::too_many_arguments)]
    pub fn sink_json_partitioned(
        self,
        base_path: Arc<PlPath>,
        file_path_cb: Option<PartitionTargetCallback>,
        variant: PartitionVariant,
        options: JsonWriterOptions,
        cloud_options: Option<polars_io::cloud::CloudOptions>,
        sink_options: SinkOptions,
        per_partition_sort_by: Option<Vec<SortColumn>>,
        finish_callback: Option<SinkFinishCallback>,
    ) -> PolarsResult<Self> {
        self.sink(SinkType::Partition(PartitionSinkType {
            base_path,
            file_path_cb,
            sink_options,
            variant,
            file_type: FileType::Json(options),
            cloud_options,
            per_partition_sort_by,
            finish_callback,
        }))
    }

    #[cfg(feature = "new_streaming")]
    pub fn try_new_streaming_if_requested(
        &mut self,
    ) -> Option<PolarsResult<polars_stream::QueryResult>> {
        let auto_new_streaming = std::env::var("POLARS_AUTO_NEW_STREAMING").as_deref() == Ok("1");
        let force_new_streaming = std::env::var("POLARS_FORCE_NEW_STREAMING").as_deref() == Ok("1");

        if auto_new_streaming || force_new_streaming {
            // Try to run using the new streaming engine, falling back
            // if it fails in a todo!() error if auto_new_streaming is set.
            let mut new_stream_lazy = self.clone();
            new_stream_lazy.opt_state |= OptFlags::NEW_STREAMING;
            let mut alp_plan = match new_stream_lazy.to_alp_optimized() {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
            };

            let f = || {
                polars_stream::run_query(
                    alp_plan.lp_top,
                    &mut alp_plan.lp_arena,
                    &mut alp_plan.expr_arena,
                )
            };

            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
                Ok(v) => return Some(v),
                Err(e) => {
                    // Fallback to normal engine if error is due to not being implemented
                    // and auto_new_streaming is set, otherwise propagate error.
                    if !force_new_streaming
                        && auto_new_streaming
                        && e.downcast_ref::<&str>()
                            .map(|s| s.starts_with("not yet implemented"))
                            .unwrap_or(false)
                    {
                        if polars_core::config::verbose() {
                            eprintln!(
                                "caught unimplemented error in new streaming engine, falling back to normal engine"
                            );
                        }
                    } else {
                        std::panic::resume_unwind(e);
                    }
                },
            }
        }

        None
    }

    fn sink(mut self, payload: SinkType) -> Result<LazyFrame, PolarsError> {
        polars_ensure!(
            !matches!(self.logical_plan, DslPlan::Sink { .. }),
            InvalidOperation: "cannot create a sink on top of another sink"
        );
        self.logical_plan = DslPlan::Sink {
            input: Arc::new(self.logical_plan),
            payload,
        };
        Ok(self)
    }

    /// Filter frame rows that match a predicate expression.
    ///
    /// The expression must yield boolean values (note that rows where the
    /// predicate resolves to `null` are *not* included in the resulting frame).
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    ///
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///         .filter(col("sepal_width").is_not_null())
    ///         .select([col("sepal_width"), col("sepal_length")])
    /// }
    /// ```
    pub fn filter(self, predicate: Expr) -> Self {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().filter(predicate).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Remove frame rows that match a predicate expression.
    ///
    /// The expression must yield boolean values (note that rows where the
    /// predicate resolves to `null` are *not* removed from the resulting frame).
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    ///
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///         .remove(col("sepal_width").is_null())
    ///         .select([col("sepal_width"), col("sepal_length")])
    /// }
    /// ```
    pub fn remove(self, predicate: Expr) -> Self {
        self.filter(predicate.neq_missing(lit(true)))
    }

    /// Select (and optionally rename, with [`alias`](crate::dsl::Expr::alias)) columns from the query.
    ///
    /// Columns can be selected with [`col`];
    /// If you want to select all columns use `col(PlSmallStr::from_static("*"))`.
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
    ///         .select([col("foo"),
    ///                   col("bar").alias("ham")])
    /// }
    ///
    /// /// This function selects all columns except "foo"
    /// fn exclude_a_column(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///         .select([all().exclude_cols(["foo"]).as_expr()])
    /// }
    /// ```
    pub fn select<E: AsRef<[Expr]>>(self, exprs: E) -> Self {
        let exprs = exprs.as_ref().to_vec();
        self.select_impl(
            exprs,
            ProjectionOptions {
                run_parallel: true,
                duplicate_check: true,
                should_broadcast: true,
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
                should_broadcast: true,
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
    ///
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///        .group_by([col("date")])
    ///        .agg([
    ///            col("rain").min().alias("min_rain"),
    ///            col("rain").sum().alias("sum_rain"),
    ///            col("rain").quantile(lit(0.5), QuantileMethod::Nearest).alias("median_rain"),
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
    /// Also works for index values of type UInt32, UInt64, Int32, or Int64.
    ///
    /// Different from a [`group_by_dynamic`][`Self::group_by_dynamic`], the windows are now determined by the
    /// individual values and are not of constant intervals. For constant intervals use
    /// *group_by_dynamic*
    #[cfg(feature = "dynamic_group_by")]
    pub fn rolling<E: AsRef<[Expr]>>(
        mut self,
        index_column: Expr,
        group_by: E,
        mut options: RollingGroupOptions,
    ) -> LazyGroupBy {
        if let Expr::Column(name) = index_column {
            options.index_column = name;
        } else {
            let output_field = index_column
                .to_field(&self.collect_schema().unwrap(), Context::Default)
                .unwrap();
            return self.with_column(index_column).rolling(
                Expr::Column(output_field.name().clone()),
                group_by,
                options,
            );
        }
        let opt_state = self.get_opt_state();
        LazyGroupBy {
            logical_plan: self.logical_plan,
            opt_state,
            keys: group_by.as_ref().to_vec(),
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
    /// The `group_by` argument should be empty `[]` if you don't want to combine this
    /// with a ordinary group_by on these keys.
    #[cfg(feature = "dynamic_group_by")]
    pub fn group_by_dynamic<E: AsRef<[Expr]>>(
        mut self,
        index_column: Expr,
        group_by: E,
        mut options: DynamicGroupOptions,
    ) -> LazyGroupBy {
        if let Expr::Column(name) = index_column {
            options.index_column = name;
        } else {
            let output_field = index_column
                .to_field(&self.collect_schema().unwrap(), Context::Default)
                .unwrap();
            return self.with_column(index_column).group_by_dynamic(
                Expr::Column(output_field.name().clone()),
                group_by,
                options,
            );
        }
        let opt_state = self.get_opt_state();
        LazyGroupBy {
            logical_plan: self.logical_plan,
            opt_state,
            keys: group_by.as_ref().to_vec(),
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
    ///         .anti_join(other, col("foo"), col("bar").cast(DataType::String))
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

    /// Creates the Cartesian product from both frames, preserving the order of the left keys.
    #[cfg(feature = "cross_join")]
    pub fn cross_join(self, other: LazyFrame, suffix: Option<PlSmallStr>) -> LazyFrame {
        self.join(
            other,
            vec![],
            vec![],
            JoinArgs::new(JoinType::Cross).with_suffix(suffix),
        )
    }

    /// Left outer join this query with another lazy query.
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
    ///         .inner_join(other, col("foo"), col("bar").cast(DataType::String))
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

    /// Full outer join this query with another lazy query.
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
    /// fn full_join_dataframes(ldf: LazyFrame, other: LazyFrame) -> LazyFrame {
    ///         ldf
    ///         .full_join(other, col("foo"), col("bar"))
    /// }
    /// ```
    pub fn full_join<E: Into<Expr>>(self, other: LazyFrame, left_on: E, right_on: E) -> LazyFrame {
        self.join(
            other,
            [left_on.into()],
            [right_on.into()],
            JoinArgs::new(JoinType::Full),
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
    ///         .semi_join(other, col("foo"), col("bar").cast(DataType::String))
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
    /// Any provided `args.slice` parameter is not considered, but set by the internal optimizer.
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
        self,
        other: LazyFrame,
        left_on: E,
        right_on: E,
        args: JoinArgs,
    ) -> LazyFrame {
        let left_on = left_on.as_ref().to_vec();
        let right_on = right_on.as_ref().to_vec();

        self._join_impl(other, left_on, right_on, args)
    }

    fn _join_impl(
        self,
        other: LazyFrame,
        left_on: Vec<Expr>,
        right_on: Vec<Expr>,
        args: JoinArgs,
    ) -> LazyFrame {
        let JoinArgs {
            how,
            validation,
            suffix,
            slice,
            nulls_equal,
            coalesce,
            maintain_order,
        } = args;

        if slice.is_some() {
            panic!("impl error: slice is not handled")
        }

        let mut builder = self
            .join_builder()
            .with(other)
            .left_on(left_on)
            .right_on(right_on)
            .how(how)
            .validate(validation)
            .join_nulls(nulls_equal)
            .coalesce(coalesce)
            .maintain_order(maintain_order);

        if let Some(suffix) = suffix {
            builder = builder.suffix(suffix);
        }

        // Note: args.slice is set by the optimizer
        builder.finish()
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
    ///             when(col("sepal_length").lt(lit(5.0)))
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
                    should_broadcast: true,
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
                should_broadcast: true,
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
                should_broadcast: true,
            },
        )
    }

    /// Match or evolve to a certain schema.
    pub fn match_to_schema(
        self,
        schema: SchemaRef,
        per_column: Arc<[MatchToSchemaPerColumn]>,
        extra_columns: ExtraColumnsPolicy,
    ) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self
            .get_plan_builder()
            .match_to_schema(schema, per_column, extra_columns)
            .build();
        Self::from_logical_plan(lp, opt_state)
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
    pub fn max(self) -> Self {
        self.map_private(DslFunction::Stats(StatsFunction::Max))
    }

    /// Aggregate all the columns as their minimum values.
    ///
    /// Aggregated columns will have the same names as the original columns.
    pub fn min(self) -> Self {
        self.map_private(DslFunction::Stats(StatsFunction::Min))
    }

    /// Aggregate all the columns as their sum values.
    ///
    /// Aggregated columns will have the same names as the original columns.
    ///
    /// - Boolean columns will sum to a `u32` containing the number of `true`s.
    /// - For integer columns, the ordinary checks for overflow are performed:
    ///   if running in `debug` mode, overflows will panic, whereas in `release` mode overflows will
    ///   silently wrap.
    /// - String columns will sum to None.
    pub fn sum(self) -> Self {
        self.map_private(DslFunction::Stats(StatsFunction::Sum))
    }

    /// Aggregate all the columns as their mean values.
    ///
    /// - Boolean and integer columns are converted to `f64` before computing the mean.
    /// - String columns will have a mean of None.
    pub fn mean(self) -> Self {
        self.map_private(DslFunction::Stats(StatsFunction::Mean))
    }

    /// Aggregate all the columns as their median values.
    ///
    /// - Boolean and integer results are converted to `f64`. However, they are still
    ///   susceptible to overflow before this conversion occurs.
    /// - String columns will sum to None.
    pub fn median(self) -> Self {
        self.map_private(DslFunction::Stats(StatsFunction::Median))
    }

    /// Aggregate all the columns as their quantile values.
    pub fn quantile(self, quantile: Expr, method: QuantileMethod) -> Self {
        self.map_private(DslFunction::Stats(StatsFunction::Quantile {
            quantile,
            method,
        }))
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
    pub fn std(self, ddof: u8) -> Self {
        self.map_private(DslFunction::Stats(StatsFunction::Std { ddof }))
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
    pub fn var(self, ddof: u8) -> Self {
        self.map_private(DslFunction::Stats(StatsFunction::Var { ddof }))
    }

    /// Apply explode operation. [See eager explode](polars_core::frame::DataFrame::explode).
    pub fn explode(self, columns: Selector) -> LazyFrame {
        self.explode_impl(columns, false)
    }

    /// Apply explode operation. [See eager explode](polars_core::frame::DataFrame::explode).
    fn explode_impl(self, columns: Selector, allow_empty: bool) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self
            .get_plan_builder()
            .explode(columns, allow_empty)
            .build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Aggregate all the columns as the sum of their null value count.
    pub fn null_count(self) -> LazyFrame {
        self.select(vec![col(PlSmallStr::from_static("*")).null_count()])
    }

    /// Drop non-unique rows and maintain the order of kept rows.
    ///
    /// `subset` is an optional `Vec` of column names to consider for uniqueness; if
    /// `None`, all columns are considered.
    pub fn unique_stable(
        self,
        subset: Option<Selector>,
        keep_strategy: UniqueKeepStrategy,
    ) -> LazyFrame {
        self.unique_stable_generic(subset, keep_strategy)
    }

    pub fn unique_stable_generic(
        self,
        subset: Option<Selector>,
        keep_strategy: UniqueKeepStrategy,
    ) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let options = DistinctOptionsDSL {
            subset,
            maintain_order: true,
            keep_strategy,
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
    pub fn unique(self, subset: Option<Selector>, keep_strategy: UniqueKeepStrategy) -> LazyFrame {
        self.unique_generic(subset, keep_strategy)
    }

    pub fn unique_generic(
        self,
        subset: Option<Selector>,
        keep_strategy: UniqueKeepStrategy,
    ) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let options = DistinctOptionsDSL {
            subset,
            maintain_order: false,
            keep_strategy,
        };
        let lp = self.get_plan_builder().distinct(options).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Drop rows containing one or more NaN values.
    ///
    /// `subset` is an optional `Vec` of column names to consider for NaNs; if None, all
    /// floating point columns are considered.
    pub fn drop_nans(self, subset: Option<Selector>) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().drop_nans(subset).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Drop rows containing one or more None values.
    ///
    /// `subset` is an optional `Vec` of column names to consider for nulls; if None, all
    /// columns are considered.
    pub fn drop_nulls(self, subset: Option<Selector>) -> LazyFrame {
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

    /// Unpivot the DataFrame from wide to long format.
    ///
    /// See [`UnpivotArgsIR`] for information on how to unpivot a DataFrame.
    #[cfg(feature = "pivot")]
    pub fn unpivot(self, args: UnpivotArgsDSL) -> LazyFrame {
        let opt_state = self.get_opt_state();
        let lp = self.get_plan_builder().unpivot(args).build();
        Self::from_logical_plan(lp, opt_state)
    }

    /// Limit the DataFrame to the first `n` rows.
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
                PlSmallStr::from_static(name.unwrap_or("ANONYMOUS UDF")),
            )
            .build();
        Self::from_logical_plan(lp, opt_state)
    }

    #[cfg(feature = "python")]
    pub fn map_python(
        self,
        function: polars_utils::python_function::PythonFunction,
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

    pub(crate) fn map_private(self, function: DslFunction) -> LazyFrame {
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
    pub fn with_row_index<S>(self, name: S, offset: Option<IdxSize>) -> LazyFrame
    where
        S: Into<PlSmallStr>,
    {
        let name = name.into();

        match &self.logical_plan {
            v @ DslPlan::Scan { scan_type, .. }
                if !matches!(&**scan_type, FileScanDsl::Anonymous { .. }) =>
            {
                let DslPlan::Scan {
                    sources,
                    mut unified_scan_args,
                    scan_type,
                    cached_ir: _,
                } = v.clone()
                else {
                    unreachable!()
                };

                unified_scan_args.row_index = Some(RowIndex {
                    name,
                    offset: offset.unwrap_or(0),
                });

                DslPlan::Scan {
                    sources,
                    unified_scan_args,
                    scan_type,
                    cached_ir: Default::default(),
                }
                .into()
            },
            _ => self.map_private(DslFunction::RowIndex { name, offset }),
        }
    }

    /// Return the number of non-null elements for each column.
    pub fn count(self) -> LazyFrame {
        self.select(vec![col(PlSmallStr::from_static("*")).count()])
    }

    /// Unnest the given `Struct` columns: the fields of the `Struct` type will be
    /// inserted as columns.
    #[cfg(feature = "dtype-struct")]
    pub fn unnest(self, cols: Selector) -> Self {
        self.map_private(DslFunction::Unnest(cols))
    }

    #[cfg(feature = "merge_sorted")]
    pub fn merge_sorted<S>(self, other: LazyFrame, key: S) -> PolarsResult<LazyFrame>
    where
        S: Into<PlSmallStr>,
    {
        let key = key.into();

        let lp = DslPlan::MergeSorted {
            input_left: Arc::new(self.logical_plan),
            input_right: Arc::new(other.logical_plan),
            key,
        };
        Ok(LazyFrame::from_logical_plan(lp, self.opt_state))
    }
}

/// Utility struct for lazy group_by operation.
#[derive(Clone)]
pub struct LazyGroupBy {
    pub logical_plan: DslPlan,
    opt_state: OptFlags,
    keys: Vec<Expr>,
    maintain_order: bool,
    #[cfg(feature = "dynamic_group_by")]
    dynamic_options: Option<DynamicGroupOptions>,
    #[cfg(feature = "dynamic_group_by")]
    rolling_options: Option<RollingGroupOptions>,
}

impl From<LazyGroupBy> for LazyFrame {
    fn from(lgb: LazyGroupBy) -> Self {
        Self {
            logical_plan: lgb.logical_plan,
            opt_state: lgb.opt_state,
            cached_arena: Default::default(),
        }
    }
}

impl LazyGroupBy {
    /// Group by and aggregate.
    ///
    /// Select a column with [col] and choose an aggregation.
    /// If you want to aggregate all columns use `col(PlSmallStr::from_static("*"))`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    ///
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///        .group_by_stable([col("date")])
    ///        .agg([
    ///            col("rain").min().alias("min_rain"),
    ///            col("rain").sum().alias("sum_rain"),
    ///            col("rain").quantile(lit(0.5), QuantileMethod::Nearest).alias("median_rain"),
    ///        ])
    /// }
    /// ```
    pub fn agg<E: AsRef<[Expr]>>(self, aggs: E) -> LazyFrame {
        #[cfg(feature = "dynamic_group_by")]
        let lp = DslBuilder::from(self.logical_plan)
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
        let lp = DslBuilder::from(self.logical_plan)
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

        self.agg([all().as_expr().head(n).explode()])
            .explode_impl(all() - by_name(keys.iter().cloned(), false), true)
    }

    /// Return last n rows of each group
    pub fn tail(self, n: Option<usize>) -> LazyFrame {
        let keys = self
            .keys
            .iter()
            .filter_map(|expr| expr_output_name(expr).ok())
            .collect::<Vec<_>>();

        self.agg([all().as_expr().tail(n).explode()])
            .explode_impl(all() - by_name(keys.iter().cloned(), false), true)
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

        let lp = DslPlan::GroupBy {
            input: Arc::new(self.logical_plan),
            keys: self.keys,
            aggs: vec![],
            apply: Some((Arc::new(f), schema)),
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
    suffix: Option<PlSmallStr>,
    validation: JoinValidation,
    nulls_equal: bool,
    coalesce: JoinCoalesce,
    maintain_order: MaintainOrderJoin,
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
            nulls_equal: false,
            coalesce: Default::default(),
            maintain_order: Default::default(),
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
        self.left_on.clone_from(&on);
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

    /// Join on null values. By default null values will never produce matches.
    pub fn join_nulls(mut self, nulls_equal: bool) -> Self {
        self.nulls_equal = nulls_equal;
        self
    }

    /// Suffix to add duplicate column names in join.
    /// Defaults to `"_right"` if this method is never called.
    pub fn suffix<S>(mut self, suffix: S) -> Self
    where
        S: Into<PlSmallStr>,
    {
        self.suffix = Some(suffix.into());
        self
    }

    /// Whether to coalesce join columns.
    pub fn coalesce(mut self, coalesce: JoinCoalesce) -> Self {
        self.coalesce = coalesce;
        self
    }

    /// Whether to preserve the row order.
    pub fn maintain_order(mut self, maintain_order: MaintainOrderJoin) -> Self {
        self.maintain_order = maintain_order;
        self
    }

    /// Finish builder
    pub fn finish(self) -> LazyFrame {
        let opt_state = self.lf.opt_state;
        let other = self.other.expect("'with' not set in join builder");

        let args = JoinArgs {
            how: self.how,
            validation: self.validation,
            suffix: self.suffix,
            slice: None,
            nulls_equal: self.nulls_equal,
            coalesce: self.coalesce,
            maintain_order: self.maintain_order,
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
                }
                .into(),
            )
            .build();
        LazyFrame::from_logical_plan(lp, opt_state)
    }

    // Finish with join predicates
    pub fn join_where(self, predicates: Vec<Expr>) -> LazyFrame {
        let opt_state = self.lf.opt_state;
        let other = self.other.expect("with not set");

        // Decompose `And` conjunctions into their component expressions
        fn decompose_and(predicate: Expr, expanded_predicates: &mut Vec<Expr>) {
            if let Expr::BinaryExpr {
                op: Operator::And,
                left,
                right,
            } = predicate
            {
                decompose_and((*left).clone(), expanded_predicates);
                decompose_and((*right).clone(), expanded_predicates);
            } else {
                expanded_predicates.push(predicate);
            }
        }
        let mut expanded_predicates = Vec::with_capacity(predicates.len() * 2);
        for predicate in predicates {
            decompose_and(predicate, &mut expanded_predicates);
        }
        let predicates: Vec<Expr> = expanded_predicates;

        // Decompose `is_between` predicates to allow for cleaner expression of range joins
        #[cfg(feature = "is_between")]
        let predicates: Vec<Expr> = {
            let mut expanded_predicates = Vec::with_capacity(predicates.len() * 2);
            for predicate in predicates {
                if let Expr::Function {
                    function: FunctionExpr::Boolean(BooleanFunction::IsBetween { closed }),
                    input,
                    ..
                } = &predicate
                {
                    if let [expr, lower, upper] = input.as_slice() {
                        match closed {
                            ClosedInterval::Both => {
                                expanded_predicates.push(expr.clone().gt_eq(lower.clone()));
                                expanded_predicates.push(expr.clone().lt_eq(upper.clone()));
                            },
                            ClosedInterval::Right => {
                                expanded_predicates.push(expr.clone().gt(lower.clone()));
                                expanded_predicates.push(expr.clone().lt_eq(upper.clone()));
                            },
                            ClosedInterval::Left => {
                                expanded_predicates.push(expr.clone().gt_eq(lower.clone()));
                                expanded_predicates.push(expr.clone().lt(upper.clone()));
                            },
                            ClosedInterval::None => {
                                expanded_predicates.push(expr.clone().gt(lower.clone()));
                                expanded_predicates.push(expr.clone().lt(upper.clone()));
                            },
                        }
                        continue;
                    }
                }
                expanded_predicates.push(predicate);
            }
            expanded_predicates
        };

        let args = JoinArgs {
            how: self.how,
            validation: self.validation,
            suffix: self.suffix,
            slice: None,
            nulls_equal: self.nulls_equal,
            coalesce: self.coalesce,
            maintain_order: self.maintain_order,
        };
        let options = JoinOptions {
            allow_parallel: self.allow_parallel,
            force_parallel: self.force_parallel,
            args,
        };

        let lp = DslPlan::Join {
            input_left: Arc::new(self.lf.logical_plan),
            input_right: Arc::new(other.logical_plan),
            left_on: Default::default(),
            right_on: Default::default(),
            predicates,
            options: Arc::from(options),
        };

        LazyFrame::from_logical_plan(lp, opt_state)
    }
}

pub const BUILD_STREAMING_EXECUTOR: Option<polars_mem_engine::StreamingExecutorBuilder> = {
    #[cfg(not(feature = "new_streaming"))]
    {
        None
    }
    #[cfg(feature = "new_streaming")]
    {
        Some(streaming_dispatch::build_streaming_query_executor)
    }
};
#[cfg(feature = "new_streaming")]
pub use streaming_dispatch::build_streaming_query_executor;

#[cfg(feature = "new_streaming")]
mod streaming_dispatch {
    use std::sync::{Arc, Mutex};

    use polars_core::POOL;
    use polars_core::error::PolarsResult;
    use polars_core::frame::DataFrame;
    use polars_expr::state::ExecutionState;
    use polars_mem_engine::Executor;
    use polars_plan::dsl::SinkTypeIR;
    use polars_plan::plans::{AExpr, IR};
    use polars_utils::arena::{Arena, Node};

    pub fn build_streaming_query_executor(
        node: Node,
        ir_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<Box<dyn Executor>> {
        let rechunk = match ir_arena.get(node) {
            IR::Scan {
                unified_scan_args, ..
            } => unified_scan_args.rechunk,
            _ => false,
        };

        let node = match ir_arena.get(node) {
            IR::SinkMultiple { .. } => panic!("SinkMultiple not supported"),
            IR::Sink { .. } => node,
            _ => ir_arena.add(IR::Sink {
                input: node,
                payload: SinkTypeIR::Memory,
            }),
        };

        polars_stream::StreamingQuery::build(node, ir_arena, expr_arena)
            .map(Some)
            .map(Mutex::new)
            .map(Arc::new)
            .map(|x| StreamingQueryExecutor {
                executor: x,
                rechunk,
            })
            .map(|x| Box::new(x) as Box<dyn Executor>)
    }

    // Note: Arc/Mutex is because Executor requires Sync, but SlotMap is not Sync.
    struct StreamingQueryExecutor {
        executor: Arc<Mutex<Option<polars_stream::StreamingQuery>>>,
        rechunk: bool,
    }

    impl Executor for StreamingQueryExecutor {
        fn execute(&mut self, _cache: &mut ExecutionState) -> PolarsResult<DataFrame> {
            // Must not block rayon thread on pending new-streaming future.
            assert!(POOL.current_thread_index().is_none());

            let mut df = { self.executor.try_lock().unwrap().take() }
                .expect("unhandled: execute() more than once")
                .execute()
                .map(|x| x.unwrap_single())?;

            if self.rechunk {
                df.as_single_chunk_par();
            }

            Ok(df)
        }
    }
}
