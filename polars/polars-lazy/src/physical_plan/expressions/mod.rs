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
pub(crate) use binary::*;
pub(crate) use cast::*;
pub(crate) use column::*;
pub(crate) use count::*;
pub(crate) use filter::*;
pub(crate) use literal::*;
use polars_arrow::export::arrow::array::ListArray;
use polars_arrow::export::arrow::offset::Offsets;
use polars_arrow::trusted_len::PushUnchecked;
use polars_arrow::utils::CustomIterTools;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use polars_io::predicates::PhysicalIoExpr;
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
    /// Already aggregated: `.agg_list(group_tuples`) is called
    /// and produced a `Series` of any dtype that is not nested.
    /// think of `sum`, `mean`, `variance` like aggregations.
    AggregatedFlat(Series),
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
            }
            _ => true,
        }
    }
}

// lazy update strategy
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(PartialEq)]
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
    // Same as WithSeriesLen, but now take a series given by the caller
    WithSeriesLenOwned(Series),
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
    // special state that just should propagate nulls on aggregations.
    // this is needed as (expr - expr.mean()) could leave nulls but is
    // not really a final aggregation as left is still a list, but right only
    // contains null and thus propagates that.
    null_propagated: bool,
}

impl<'a> AggregationContext<'a> {
    pub(crate) fn groups(&mut self) -> &Cow<'a, GroupsProxy> {
        match self.update_groups {
            UpdateGroups::No => {}
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
                    }
                    // sliced groups are already in correct order
                    GroupsProxy::Slice { .. } => {}
                }
                self.update_groups = UpdateGroups::No;
            }
            UpdateGroups::WithSeriesLen => {
                let s = self.series().clone();
                self.det_groups_from_list(&s);
            }
            UpdateGroups::WithSeriesLenOwned(ref s) => {
                let s = s.clone();
                self.det_groups_from_list(&s);
            }
        }
        &self.groups
    }

    pub(crate) fn series(&self) -> &Series {
        match &self.state {
            AggState::NotAggregated(s)
            | AggState::AggregatedFlat(s)
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

    pub(crate) fn combine_groups(&mut self, other: AggregationContext) -> &mut Self {
        if let (Cow::Borrowed(_), Cow::Owned(a)) = (&self.groups, other.groups) {
            self.groups = Cow::Owned(a);
        };
        self
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
            }
            (true, _) => {
                assert_eq!(series.len(), groups.len());
                AggState::AggregatedFlat(series)
            }
            _ => AggState::NotAggregated(series),
        };

        Self {
            state: series,
            groups,
            sorted: false,
            update_groups: UpdateGroups::No,
            original_len: true,
            null_propagated: false,
        }
    }

    fn from_literal(lit: Series, groups: Cow<'a, GroupsProxy>) -> AggregationContext<'a> {
        Self {
            state: AggState::Literal(lit),
            groups,
            sorted: false,
            update_groups: UpdateGroups::No,
            original_len: true,
            null_propagated: false,
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
            }
            _ => {
                let groups = self
                    .series()
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
                    .collect_trusted();
                self.groups = Cow::Owned(GroupsProxy::Slice {
                    groups,
                    rolling: false,
                });
            }
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
            }
            AggState::AggregatedFlat(_) => {}
            AggState::AggregatedList(_) => {}
            AggState::Literal(_) => {}
        }
    }

    /// # Arguments
    /// - `aggregated` sets if the Series is a list due to aggregation (could also be a list because its
    /// the columns dtype)
    pub(crate) fn with_series(&mut self, series: Series, aggregated: bool) -> &mut Self {
        self.state = match (aggregated, series.dtype()) {
            (true, &DataType::List(_)) => {
                assert_eq!(series.len(), self.groups.len());
                AggState::AggregatedList(series)
            }
            (true, _) => AggState::AggregatedFlat(series),
            _ => {
                // already aggregated to sum, min even this series was flattened it never could
                // retrieve the length before grouping, so it stays  in this state.
                if let AggState::AggregatedFlat(_) = self.state {
                    AggState::AggregatedFlat(series)
                } else {
                    AggState::NotAggregated(series)
                }
            }
        };
        self
    }

    pub(crate) fn with_literal(&mut self, series: Series) -> &mut Self {
        self.state = AggState::Literal(series);
        self
    }

    /// Update the group tuples
    pub(crate) fn with_groups(&mut self, groups: GroupsProxy) -> &mut Self {
        // In case of new groups, a series always needs to be flattened
        self.with_series(self.flat_naive().into_owned(), false);
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
                        eprintln!("groups may be out of bounds; more groups than elements in a series is only possible in dynamic groupby")
                    }
                }

                // safety:
                // groups are in bounds
                let out = unsafe { s.agg_list(&self.groups) };
                self.state = AggState::AggregatedList(out.clone());

                self.sorted = true;
                self.update_groups = UpdateGroups::WithGroupsLen;
                out
            }
            AggState::AggregatedList(s) | AggState::AggregatedFlat(s) => s,
            AggState::Literal(s) => {
                self.groups();
                let rows = self.groups.len();
                let s = s.new_from_index(0, rows);
                s.reshape(&[rows as i64, -1]).unwrap()
            }
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
            }
            _ => self.aggregated(),
        }
    }

    /// Different from aggregated, in arity operations we expect literals to expand to the size of the
    /// group
    /// eg:
    ///
    /// lit(9) in groups [[1, 1], [2, 2, 2]]
    /// becomes: [[9, 9], [9, 9, 9]]
    ///
    /// where in [`Self::aggregated`] this becomes [9, 9]
    ///
    /// this is because comparisons need to create mask that have a correct length.
    fn aggregated_arity_operation(&mut self) -> Series {
        if let AggState::Literal(s) = self.agg_state() {
            // stop borrow;
            let s = s.clone();
            let groups = self.groups();

            let mut offsets = Vec::with_capacity(groups.len() + 1);

            let mut last_offset = 0i64;
            offsets.push(last_offset);
            for g in groups.iter() {
                last_offset += g.len() as i64;
                // safety:
                // we allocated enough
                unsafe { offsets.push_unchecked(last_offset) };
            }
            let values = s.new_from_index(0, last_offset as usize);
            let values = values.array_ref(0).clone();
            // Safety:
            // offsets are monotonically increasing
            let arr = unsafe {
                ListArray::<i64>::new(
                    DataType::List(Box::new(s.dtype().clone())).to_arrow(),
                    Offsets::new_unchecked(offsets).into(),
                    values,
                    None,
                )
            };
            Series::try_from((s.name(), Box::new(arr) as ArrayRef)).unwrap()
        } else {
            self.aggregated()
        }
    }

    // If a binary or ternary function has both of these branches true, it should
    // flatten the list
    fn arity_should_explode(&self) -> bool {
        use AggState::*;
        match self.agg_state() {
            Literal(s) => s.len() == 1,
            AggregatedFlat(_) => true,
            _ => false,
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
            }
            AggState::AggregatedFlat(s) => Cow::Borrowed(s),
            AggState::Literal(s) => Cow::Borrowed(s),
        }
    }

    /// Take the series.
    pub(crate) fn take(&mut self) -> Series {
        let s = match &mut self.state {
            AggState::NotAggregated(s)
            | AggState::AggregatedFlat(s)
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
    #[cfg(feature = "parquet")]
    fn as_stats_evaluator(&self) -> Option<&dyn polars_io::predicates::StatsEvaluator> {
        None
    }

    //
    fn is_valid_aggregation(&self) -> bool;

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
}

impl PhysicalIoExpr for PhysicalIoHelper {
    fn evaluate(&self, df: &DataFrame) -> PolarsResult<Series> {
        self.expr.evaluate(df, &Default::default())
    }

    #[cfg(feature = "parquet")]
    fn as_stats_evaluator(&self) -> Option<&dyn polars_io::predicates::StatsEvaluator> {
        self.expr.as_stats_evaluator()
    }
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
