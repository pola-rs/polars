mod aggregation;
mod alias;
mod apply;
mod binary;
mod cast;
mod column;
mod count;
mod filter;
mod group_iter;
mod literal;
#[cfg(feature = "dynamic_group_by")]
mod rolling;
mod slice;
mod sort;
mod sortby;
mod take;
mod ternary;
mod window;

use std::borrow::Cow;
use std::fmt::{Display, Formatter};

pub(crate) use aggregation::*;
pub(crate) use alias::*;
pub(crate) use apply::*;
use arrow::legacy::utils::CustomIterTools;
pub(crate) use binary::*;
pub(crate) use cast::*;
pub(crate) use column::*;
pub(crate) use count::*;
pub(crate) use filter::*;
pub(crate) use literal::*;
use polars_core::frame::group_by::GroupsProxy;
use polars_core::prelude::*;
use polars_io::predicates::PhysicalIoExpr;
#[cfg(feature = "dynamic_group_by")]
pub(crate) use rolling::RollingExpr;
pub(crate) use slice::*;
pub(crate) use sort::*;
pub(crate) use sortby::*;
pub(crate) use take::*;
pub(crate) use ternary::*;
pub(crate) use window::*;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

#[derive(Clone, Debug)]
pub(crate) enum AggState {
    /// Already aggregated: `.agg_list(group_tuples`) is called
    /// and produced a `Series` of dtype `List`
    AggregatedList(Series),
    /// Already aggregated: `.agg` is called on an aggregation
    /// that produces a scalar.
    /// think of `sum`, `mean`, `variance` like aggregations.
    AggregatedScalar(Series),
    /// Not yet aggregated: `agg_list` still has to be called.
    NotAggregated(Series),
    Literal(Series),
}

impl AggState {
    // Literal series are not safe to aggregate
    fn safe_to_agg(&self, groups: &GroupsProxy) -> bool {
        match self {
            AggState::NotAggregated(s) => {
                !(s.len() == 1
                    // or more then one group
                    && (groups.len() > 1
                    // or single groups with more than one index
                    || !groups.is_empty()
                    && groups.get(0).len() > 1))
            },
            _ => true,
        }
    }

    fn try_map<F>(&self, func: F) -> PolarsResult<Self>
    where
        F: FnOnce(&Series) -> PolarsResult<Series>,
    {
        Ok(match self {
            AggState::AggregatedList(s) => AggState::AggregatedList(func(s)?),
            AggState::AggregatedScalar(s) => AggState::AggregatedScalar(func(s)?),
            AggState::Literal(s) => AggState::Literal(func(s)?),
            AggState::NotAggregated(s) => AggState::NotAggregated(func(s)?),
        })
    }

    fn map<F>(&self, func: F) -> Self
    where
        F: FnOnce(&Series) -> Series,
    {
        self.try_map(|s| Ok(func(s))).unwrap()
    }
}

// lazy update strategy
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(PartialEq, Clone, Copy)]
pub(crate) enum UpdateGroups {
    /// don't update groups
    No,
    /// use the length of the current groups to determine new sorted indexes, preferred
    /// for performance
    WithGroupsLen,
    /// use the series list offsets to determine the new group lengths
    /// this one should be used when the length has changed. Note that
    /// the series should be aggregated state or else it will panic.
    WithSeriesLen,
}

#[cfg_attr(debug_assertions, derive(Debug))]
pub struct AggregationContext<'a> {
    /// Can be in one of two states
    /// 1. already aggregated as list
    /// 2. flat (still needs the grouptuples to aggregate)
    state: AggState,
    /// group tuples for AggState
    groups: Cow<'a, GroupsProxy>,
    /// if the group tuples are already used in a level above
    /// and the series is exploded, the group tuples are sorted
    /// e.g. the exploded Series is grouped per group.
    sorted: bool,
    /// This is used to determined if we need to update the groups
    /// into a sorted groups. We do this lazily, so that this work only is
    /// done when the groups are needed
    update_groups: UpdateGroups,
    /// This is true when the Series and GroupsProxy still have all
    /// their original values. Not the case when filtered
    original_len: bool,
}

impl<'a> AggregationContext<'a> {
    pub(crate) fn groups(&mut self) -> &Cow<'a, GroupsProxy> {
        match self.update_groups {
            UpdateGroups::No => {},
            UpdateGroups::WithGroupsLen => {
                // the groups are unordered
                // and the series is aggregated with this groups
                // so we need to recreate new grouptuples that
                // match the exploded Series
                let mut offset = 0 as IdxSize;

                match self.groups.as_ref() {
                    GroupsProxy::Idx(groups) => {
                        let groups = groups
                            .iter()
                            .map(|g| {
                                let len = g.1.len() as IdxSize;
                                let new_offset = offset + len;
                                let out = [offset, len];
                                offset = new_offset;
                                out
                            })
                            .collect();
                        self.groups = Cow::Owned(GroupsProxy::Slice {
                            groups,
                            rolling: false,
                        })
                    },
                    // sliced groups are already in correct order
                    GroupsProxy::Slice { .. } => {},
                }
                self.update_groups = UpdateGroups::No;
            },
            UpdateGroups::WithSeriesLen => {
                let s = self.series().clone();
                self.det_groups_from_list(&s);
            },
        }
        &self.groups
    }

    pub(crate) fn series(&self) -> &Series {
        match &self.state {
            AggState::NotAggregated(s)
            | AggState::AggregatedScalar(s)
            | AggState::AggregatedList(s) => s,
            AggState::Literal(s) => s,
        }
    }

    pub(crate) fn agg_state(&self) -> &AggState {
        &self.state
    }

    pub(crate) fn is_not_aggregated(&self) -> bool {
        matches!(
            &self.state,
            AggState::NotAggregated(_) | AggState::Literal(_)
        )
    }

    pub(crate) fn is_aggregated(&self) -> bool {
        !self.is_not_aggregated()
    }

    pub(crate) fn is_literal(&self) -> bool {
        matches!(self.state, AggState::Literal(_))
    }

    /// # Arguments
    /// - `aggregated` sets if the Series is a list due to aggregation (could also be a list because its
    /// the columns dtype)
    fn new(
        series: Series,
        groups: Cow<'a, GroupsProxy>,
        aggregated: bool,
    ) -> AggregationContext<'a> {
        let series = match (aggregated, series.dtype()) {
            (true, &DataType::List(_)) => {
                assert_eq!(series.len(), groups.len());
                AggState::AggregatedList(series)
            },
            (true, _) => {
                assert_eq!(series.len(), groups.len());
                AggState::AggregatedScalar(series)
            },
            _ => AggState::NotAggregated(series),
        };

        Self {
            state: series,
            groups,
            sorted: false,
            update_groups: UpdateGroups::No,
            original_len: true,
        }
    }

    fn with_agg_state(&mut self, agg_state: AggState) {
        self.state = agg_state;
    }

    fn from_agg_state(agg_state: AggState, groups: Cow<'a, GroupsProxy>) -> AggregationContext<'a> {
        Self {
            state: agg_state,
            groups,
            sorted: false,
            update_groups: UpdateGroups::No,
            original_len: true,
        }
    }

    fn from_literal(lit: Series, groups: Cow<'a, GroupsProxy>) -> AggregationContext<'a> {
        Self {
            state: AggState::Literal(lit),
            groups,
            sorted: false,
            update_groups: UpdateGroups::No,
            original_len: true,
        }
    }

    pub(crate) fn set_original_len(&mut self, original_len: bool) -> &mut Self {
        self.original_len = original_len;
        self
    }

    pub(crate) fn with_update_groups(&mut self, update: UpdateGroups) -> &mut Self {
        self.update_groups = update;
        self
    }

    pub(crate) fn det_groups_from_list(&mut self, s: &Series) {
        let mut offset = 0 as IdxSize;
        let list = s
            .list()
            .expect("impl error, should be a list at this point");

        match list.chunks().len() {
            1 => {
                let arr = list.downcast_iter().next().unwrap();
                let offsets = arr.offsets().as_slice();

                let mut previous = 0i64;
                let groups = offsets[1..]
                    .iter()
                    .map(|&o| {
                        let len = (o - previous) as IdxSize;
                        // explode will fill empty rows with null, so we must increment the group
                        // offset accordingly
                        let new_offset = offset + len + (len == 0) as IdxSize;

                        previous = o;
                        let out = [offset, len];
                        offset = new_offset;
                        out
                    })
                    .collect_trusted();
                self.groups = Cow::Owned(GroupsProxy::Slice {
                    groups,
                    rolling: false,
                });
            },
            _ => {
                // SAFETY: unstable series never lives longer than the iterator.
                let groups = unsafe {
                    self.series()
                        .list()
                        .expect("impl error, should be a list at this point")
                        .amortized_iter()
                        .map(|s| {
                            if let Some(s) = s {
                                let len = s.as_ref().len() as IdxSize;
                                let new_offset = offset + len;
                                let out = [offset, len];
                                offset = new_offset;
                                out
                            } else {
                                [offset, 0]
                            }
                        })
                        .collect_trusted()
                };
                self.groups = Cow::Owned(GroupsProxy::Slice {
                    groups,
                    rolling: false,
                });
            },
        }
        self.update_groups = UpdateGroups::No;
    }

    /// In a binary expression one state can be aggregated and the other not.
    /// If both would be flattened naively one would be sorted and the other not.
    /// Calling this function will ensure both are sorted. This will be a no-op
    /// if already aggregated.
    pub(crate) fn sort_by_groups(&mut self) {
        // make sure that the groups are updated before we use them to sort.
        self.groups();
        match &self.state {
            AggState::NotAggregated(s) => {
                // We should not aggregate literals!!
                if self.state.safe_to_agg(&self.groups) {
                    // safety:
                    // groups are in bounds
                    let agg = unsafe { s.agg_list(&self.groups) };
                    self.update_groups = UpdateGroups::WithGroupsLen;
                    self.state = AggState::AggregatedList(agg);
                }
            },
            AggState::AggregatedScalar(_) => {},
            AggState::AggregatedList(_) => {},
            AggState::Literal(_) => {},
        }
    }

    /// # Arguments
    /// - `aggregated` sets if the Series is a list due to aggregation (could also be a list because its
    /// the columns dtype)
    pub(crate) fn with_series(
        &mut self,
        series: Series,
        aggregated: bool,
        expr: Option<&Expr>,
    ) -> PolarsResult<&mut Self> {
        self.with_series_and_args(series, aggregated, expr, false)
    }

    pub(crate) fn with_series_and_args(
        &mut self,
        series: Series,
        aggregated: bool,
        expr: Option<&Expr>,
        // if the applied function was a `map` instead of an `apply`
        // this will keep functions applied over literals as literals: F(lit) = lit
        mapped: bool,
    ) -> PolarsResult<&mut Self> {
        self.state = match (aggregated, series.dtype()) {
            (true, &DataType::List(_)) => {
                if series.len() != self.groups.len() {
                    let fmt_expr = if let Some(e) = expr {
                        format!("'{e}' ")
                    } else {
                        String::new()
                    };
                    polars_bail!(
                        ComputeError:
                        "aggregation expression '{}' produced a different number of elements: {} \
                        than the number of groups: {} (this is likely invalid)",
                        fmt_expr, series.len(), self.groups.len(),
                    );
                }
                AggState::AggregatedList(series)
            },
            (true, _) => AggState::AggregatedScalar(series),
            _ => {
                match self.state {
                    // already aggregated to sum, min even this series was flattened it never could
                    // retrieve the length before grouping, so it stays  in this state.
                    AggState::AggregatedScalar(_) => AggState::AggregatedScalar(series),
                    // applying a function on a literal, keeps the literal state
                    AggState::Literal(_) if series.len() == 1 && mapped => {
                        AggState::Literal(series)
                    },
                    _ => AggState::NotAggregated(series),
                }
            },
        };
        Ok(self)
    }

    pub(crate) fn with_literal(&mut self, series: Series) -> &mut Self {
        self.state = AggState::Literal(series);
        self
    }

    /// Update the group tuples
    pub(crate) fn with_groups(&mut self, groups: GroupsProxy) -> &mut Self {
        // In case of new groups, a series always needs to be flattened
        self.with_series(self.flat_naive().into_owned(), false, None)
            .unwrap();
        self.groups = Cow::Owned(groups);
        // make sure that previous setting is not used
        self.update_groups = UpdateGroups::No;
        self
    }

    /// Get the aggregated version of the series.
    pub(crate) fn aggregated(&mut self) -> Series {
        // we clone, because we only want to call `self.groups()` if needed.
        // self groups may instantiate new groups and thus can be expensive.
        match self.state.clone() {
            AggState::NotAggregated(s) => {
                // The groups are determined lazily and in case of a flat/non-aggregated
                // series we use the groups to aggregate the list
                // because this is lazy, we first must to update the groups
                // by calling .groups()
                self.groups();
                #[cfg(debug_assertions)]
                {
                    if self.groups.len() > s.len() {
                        polars_warn!("groups may be out of bounds; more groups than elements in a series is only possible in dynamic group_by")
                    }
                }

                // safety:
                // groups are in bounds
                let out = unsafe { s.agg_list(&self.groups) };
                self.state = AggState::AggregatedList(out.clone());

                self.sorted = true;
                self.update_groups = UpdateGroups::WithGroupsLen;
                out
            },
            AggState::AggregatedList(s) | AggState::AggregatedScalar(s) => s,
            AggState::Literal(s) => {
                self.groups();
                let rows = self.groups.len();
                let s = s.new_from_index(0, rows);
                s.reshape(&[rows as i64, -1]).unwrap()
            },
        }
    }

    /// Get the final aggregated version of the series.
    pub(crate) fn finalize(&mut self) -> Series {
        // we clone, because we only want to call `self.groups()` if needed.
        // self groups may instantiate new groups and thus can be expensive.
        match &self.state {
            AggState::Literal(s) => {
                let s = s.clone();
                self.groups();
                let rows = self.groups.len();
                s.new_from_index(0, rows)
            },
            _ => self.aggregated(),
        }
    }

    // If a binary or ternary function has both of these branches true, it should
    // flatten the list
    fn arity_should_explode(&self) -> bool {
        use AggState::*;
        match self.agg_state() {
            Literal(s) => s.len() == 1,
            AggregatedScalar(_) => true,
            _ => false,
        }
    }

    pub(crate) fn get_final_aggregation(mut self) -> (Series, Cow<'a, GroupsProxy>) {
        let _ = self.groups();
        let groups = self.groups;
        match self.state {
            AggState::NotAggregated(s) => (s, groups),
            AggState::AggregatedScalar(s) => (s, groups),
            AggState::Literal(s) => (s, groups),
            AggState::AggregatedList(s) => {
                let flattened = s.explode().unwrap();
                let groups = groups.into_owned();
                // unroll the possible flattened state
                // say we have groups with overlapping windows:
                //
                // offset, len
                // 0, 1
                // 0, 2
                // 0, 4
                //
                // gets aggregation
                //
                // [0]
                // [0, 1],
                // [0, 1, 2, 3]
                //
                // before aggregation the column was
                // [0, 1, 2, 3]
                // but explode on this list yields
                // [0, 0, 1, 0, 1, 2, 3]
                //
                // so we unroll the groups as
                //
                // [0, 1]
                // [1, 2]
                // [3, 4]
                let groups = groups.unroll();
                (flattened, Cow::Owned(groups))
            },
        }
    }

    /// Get the not-aggregated version of the series.
    /// Note that we call it naive, because if a previous expr
    /// has filtered or sorted this, this information is in the
    /// group tuples not the flattened series.
    pub(crate) fn flat_naive(&self) -> Cow<'_, Series> {
        match &self.state {
            AggState::NotAggregated(s) => Cow::Borrowed(s),
            AggState::AggregatedList(s) => {
                #[cfg(debug_assertions)]
                {
                    // panic so we find cases where we accidentally explode overlapping groups
                    // we don't want this as this can create a lot of data
                    if let GroupsProxy::Slice { rolling: true, .. } = self.groups.as_ref() {
                        panic!("implementation error, polars should not hit this branch for overlapping groups")
                    }
                }

                Cow::Owned(s.explode().unwrap())
            },
            AggState::AggregatedScalar(s) => Cow::Borrowed(s),
            AggState::Literal(s) => Cow::Borrowed(s),
        }
    }

    /// Take the series.
    pub(crate) fn take(&mut self) -> Series {
        let s = match &mut self.state {
            AggState::NotAggregated(s)
            | AggState::AggregatedScalar(s)
            | AggState::AggregatedList(s) => s,
            AggState::Literal(s) => s,
        };
        std::mem::take(s)
    }
}

/// Take a DataFrame and evaluate the expressions.
/// Implement this for Column, lt, eq, etc
pub trait PhysicalExpr: Send + Sync {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    /// Take a DataFrame and evaluate the expression.
    fn evaluate(&self, df: &DataFrame, _state: &ExecutionState) -> PolarsResult<Series>;

    /// Some expression that are not aggregations can be done per group
    /// Think of sort, slice, filter, shift, etc.
    /// defaults to ignoring the group
    ///
    /// This method is called by an aggregation function.
    ///
    /// In case of a simple expr, like 'column', the groups are ignored and the column is returned.
    /// In case of an expr where group behavior makes sense, this method is called.
    /// For a filter operation for instance, a Series is created per groups and filtered.
    ///
    /// An implementation of this method may apply an aggregation on the groups only. For instance
    /// on a shift, the groups are first aggregated to a `ListChunked` and the shift is applied per
    /// group. The implementation then has to return the `Series` exploded (because a later aggregation
    /// will use the group tuples to aggregate). The group tuples also have to be updated, because
    /// aggregation to a list sorts the exploded `Series` by group.
    ///
    /// This has some gotcha's. An implementation may also change the group tuples instead of
    /// the `Series`.
    ///
    // we allow this because we pass the vec to the Cow
    // Note to self: Don't be smart and dispatch to evaluate as default implementation
    // this means filters will be incorrect and lead to invalid results down the line
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>>;

    /// Get the output field of this expr
    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field>;

    /// Convert to a partitioned aggregator.
    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        None
    }

    /// Can take &dyn Statistics and determine of a file should be
    /// read -> `true`
    /// or not -> `false`
    fn as_stats_evaluator(&self) -> Option<&dyn polars_io::predicates::StatsEvaluator> {
        None
    }
    fn is_literal(&self) -> bool {
        false
    }
}

impl Display for &dyn PhysicalExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.as_expression() {
            None => Ok(()),
            Some(e) => write!(f, "{e}"),
        }
    }
}

/// Wrapper struct that allow us to use a PhysicalExpr in polars-io.
///
/// This is used to filter rows during the scan of file.
pub struct PhysicalIoHelper {
    pub expr: Arc<dyn PhysicalExpr>,
    pub has_window_function: bool,
}

impl PhysicalIoExpr for PhysicalIoHelper {
    fn evaluate_io(&self, df: &DataFrame) -> PolarsResult<Series> {
        let mut state: ExecutionState = Default::default();
        if self.has_window_function {
            state.insert_has_window_function_flag();
        }
        self.expr.evaluate(df, &state)
    }

    #[cfg(feature = "parquet")]
    fn as_stats_evaluator(&self) -> Option<&dyn polars_io::predicates::StatsEvaluator> {
        self.expr.as_stats_evaluator()
    }
}

pub(crate) fn phys_expr_to_io_expr(expr: Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalIoExpr> {
    let has_window_function = if let Some(expr) = expr.as_expression() {
        expr.into_iter()
            .any(|expr| matches!(expr, Expr::Window { .. }))
    } else {
        false
    };
    Arc::new(PhysicalIoHelper {
        expr,
        has_window_function,
    }) as Arc<dyn PhysicalIoExpr>
}

pub trait PartitionedAggregation: Send + Sync + PhysicalExpr {
    /// This is called in partitioned aggregation.
    /// Partitioned results may differ from aggregation results.
    /// For instance, for a `mean` operation a partitioned result
    /// needs to return the `sum` and the `valid_count` (length - null count).
    ///
    /// A final aggregation can then take the sum of sums and sum of valid_counts
    /// to produce a final mean.
    #[allow(clippy::ptr_arg)]
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series>;

    /// Called to merge all the partitioned results in a final aggregate.
    #[allow(clippy::ptr_arg)]
    fn finalize(
        &self,
        partitioned: Series,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series>;
}
