pub(crate) mod aggregation;
pub(crate) mod alias;
pub(crate) mod apply;
pub(crate) mod binary;
pub(crate) mod cast;
pub(crate) mod column;
pub(crate) mod count;
pub(crate) mod filter;
pub(crate) mod is_not_null;
pub(crate) mod is_null;
pub(crate) mod literal;
pub(crate) mod not;
pub(crate) mod shift;
pub(crate) mod slice;
pub(crate) mod sort;
pub(crate) mod sortby;
pub(crate) mod take;
pub(crate) mod ternary;
pub(crate) mod utils;
pub(crate) mod window;
// pub(crate) mod unique;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_arrow::utils::CustomIterTools;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use polars_io::predicates::PhysicalIoExpr;
use std::borrow::Cow;

#[derive(Clone, Debug)]
pub(crate) enum AggState {
    /// Already aggregated: `.agg_list(group_tuples` is called
    /// and produced a `Series` of dtype `List`
    AggregatedList(Series),
    /// Already aggregated: `.agg_list(group_tuples` is called
    /// and produced a `Series` of any dtype that is not nested.
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
                        self.groups = Cow::Owned(GroupsProxy::Slice(groups))
                    }
                    // sliced groups are already in correct order
                    GroupsProxy::Slice(_) => {}
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

    /// Check if this contexts group tuples can be combined with that of other.
    pub(crate) fn can_combine(&self, other: &AggregationContext) -> bool {
        match (
            &self.groups,
            self.sorted,
            self.is_original_len(),
            &other.groups,
            other.sorted,
            other.original_len,
        ) {
            (Cow::Borrowed(_), _, _, Cow::Borrowed(_), _, _) => true,
            (Cow::Owned(_), _, _, Cow::Borrowed(_), _, _) => true,
            (Cow::Borrowed(_), _, _, Cow::Owned(_), _, _) => true,
            (Cow::Owned(_), true, true, Cow::Owned(_), true, true) => true,
            (Cow::Owned(_), true, false, Cow::Owned(_), true, true) => false,
            (Cow::Owned(_), true, true, Cow::Owned(_), true, false) => false,
            (Cow::Owned(_), true, _, Cow::Owned(_), true, _) => {
                self.groups.len() == other.groups.len()
            }
            _ => false,
        }
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

    pub(crate) fn is_original_len(&self) -> bool {
        self.original_len
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
                self.groups = Cow::Owned(GroupsProxy::Slice(groups));
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
                self.groups = Cow::Owned(GroupsProxy::Slice(groups));
            }
        }
        self.update_groups = UpdateGroups::No;
    }

    /// In a binary expression one state can be aggregated and the other not.
    /// If both would be flattened naively one would be sorted and the other not.
    /// Calling this function will ensure both are sortened. This will be a no-op
    /// if already aggregated.
    pub(crate) fn sort_by_groups(&mut self) {
        match &self.state {
            AggState::NotAggregated(s) => {
                // We should not aggregate literals!!
                if self.state.safe_to_agg(&self.groups) {
                    let agg = s.agg_list(&self.groups).unwrap();
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

                let out = s
                    .agg_list(&self.groups)
                    .expect("should be able to aggregate this to list");

                if !self.sorted {
                    self.sorted = true;
                    self.update_groups = UpdateGroups::WithGroupsLen;
                };
                out
            }
            AggState::AggregatedList(s) | AggState::AggregatedFlat(s) => s,
            AggState::Literal(s) => {
                self.groups();
                let rows = self.groups.len();
                let s = s.expand_at_index(0, rows);
                s.reshape(&[rows as i64, -1]).unwrap()
            }
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
            let s = s.clone();
            // // todo! optimize this, we don't have to call agg_list, create the list directly.
            let s = s.expand_at_index(0, self.groups.iter().map(|g| g.len()).sum());
            s.agg_list(&self.groups).unwrap()
        } else {
            self.aggregated()
        }
    }

    /// Get the not-aggregated version of the series.
    /// Note that we call it naive, because if a previous expr
    /// has filtered or sorted this, this information is in the
    /// group tuples not the flattened series.
    pub(crate) fn flat_naive(&self) -> Cow<'_, Series> {
        match &self.state {
            AggState::NotAggregated(s) => Cow::Borrowed(s),
            AggState::AggregatedList(s) => Cow::Owned(s.explode().unwrap()),
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
    fn as_expression(&self) -> &Expr;

    /// Take a DataFrame and evaluate the expression.
    fn evaluate(&self, df: &DataFrame, _state: &ExecutionState) -> Result<Series>;

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
    ) -> Result<AggregationContext<'a>>;

    /// Get the output field of this expr
    fn to_field(&self, input_schema: &Schema) -> Result<Field>;

    /// Convert to a aggregation expression.
    /// This can only be done for the final expressions that produce an aggregated result.
    ///
    /// The expression sum, min, max etc can be called as `evaluate` in the standard context,
    /// or during a groupby execution, this method is called to convert them to an AggPhysicalExpr
    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        let e = self.as_expression();
        Err(PolarsError::InvalidOperation(
            format!("{:?} is not an agg expression", e).into(),
        ))
    }

    /// Can take &dyn Statistics and determine of a file should be
    /// read -> `true`
    /// or not -> `false`
    #[cfg(feature = "parquet")]
    fn as_stats_evaluator(&self) -> Option<&dyn polars_io::predicates::StatsEvaluator> {
        None
    }
}

/// Wrapper struct that allow us to use a PhysicalExpr in polars-io.
///
/// This is used to filter rows during the scan of file.
pub struct PhysicalIoHelper {
    pub expr: Arc<dyn PhysicalExpr>,
}

impl PhysicalIoExpr for PhysicalIoHelper {
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        self.expr.evaluate(df, &Default::default())
    }

    #[cfg(feature = "parquet")]
    fn as_stats_evaluator(&self) -> Option<&dyn polars_io::predicates::StatsEvaluator> {
        self.expr.as_stats_evaluator()
    }
}

pub trait PhysicalAggregation: Send + Sync {
    #[allow(clippy::ptr_arg)]
    /// Should be called on the final aggregation node like sum, min, max, etc.
    /// When called on a tail, slice, sort, etc. it should return a list-array
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Option<Series>>;

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
    ) -> Result<Option<Vec<Series>>> {
        // we return a vec, such that an implementor can return more information, such as a sum and count.
        self.aggregate(df, groups, state)
            .map(|opt| opt.map(|s| vec![s]))
    }

    /// Called to merge all the partitioned results in a final aggregate.
    #[allow(clippy::ptr_arg)]
    fn evaluate_partitioned_final(
        &self,
        final_df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        self.aggregate(final_df, groups, state)
    }
}
