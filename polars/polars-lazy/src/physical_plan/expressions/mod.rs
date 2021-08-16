pub(crate) mod aggregation;
pub(crate) mod alias;
pub(crate) mod apply;
pub(crate) mod binary;
pub(crate) mod binary_function;
pub(crate) mod cast;
pub(crate) mod column;
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
pub(crate) mod window;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use polars_io::PhysicalIoExpr;
use std::borrow::Cow;

pub(crate) enum AggState {
    List(Series),
    Flat(Series),
    None,
}

impl Default for AggState {
    fn default() -> Self {
        AggState::None
    }
}

pub struct AggregationContext<'a> {
    /// Can be in one of two states
    /// 1. already aggregated as list
    /// 2. flat (still needs the grouptuples to aggregate)
    series: AggState,
    /// group tuples for AggState
    groups: Cow<'a, GroupTuples>,
    /// if the group tuples are already used in a level above
    /// and the series is exploded, the group tuples are sorted
    /// e.g. the exploded Series is grouped per group.
    sorted: bool,
    /// This is used to determined if we need to update the groups
    /// into a sorted groups. We do this lazily, so that this work only is
    /// done when the groups are needed
    update_groups: bool,
    /// This is true when the Series and GroupTuples still have all
    /// their original values. Not the case when filtered
    original_len: bool,
}

impl<'a> AggregationContext<'a> {
    pub(crate) fn groups(&mut self) -> &Cow<'a, GroupTuples> {
        if self.update_groups {
            // the groups are unordered
            // and the series is aggregated with this groups
            // so we need to recreate new grouptuples that
            // match the exploded Series
            let mut count = 0u32;
            let groups: GroupTuples = self
                .groups
                .iter()
                .map(|g| {
                    let add = g.1.len() as u32;
                    let new_count = count + add;
                    let out = (count, (count..new_count).collect::<Vec<_>>());
                    count = new_count;
                    out
                })
                .collect();
            self.groups = Cow::Owned(groups);
            self.update_groups = false;
        }
        &self.groups
    }

    /// Check if this contexts group tuples can be combined with that of other.
    pub(crate) fn can_combine(&self, other: &AggregationContext) -> bool {
        match (
            &self.groups,
            self.sorted,
            self.original_len,
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
        match &self.series {
            AggState::List(s) => s,
            AggState::Flat(s) => s,
            _ => unreachable!(),
        }
    }

    pub(crate) fn combine_groups(&mut self, other: AggregationContext) -> &mut Self {
        if let (Cow::Borrowed(_), Cow::Owned(a)) = (&self.groups, other.groups) {
            self.groups = Cow::Owned(a);
        };
        self
    }

    pub(crate) fn new(series: Series, groups: Cow<'a, GroupTuples>) -> AggregationContext<'a> {
        let series = match series.list() {
            Ok(_) => AggState::List(series),
            Err(_) => AggState::Flat(series),
        };

        Self {
            series,
            groups,
            sorted: false,
            update_groups: false,
            original_len: true,
        }
    }

    pub(crate) fn set_original_len(&mut self, original_len: bool) -> &mut Self {
        self.original_len = original_len;
        self
    }

    pub(crate) fn with_series(&mut self, series: Series) -> &mut Self {
        self.series = match series.list() {
            Ok(_) => AggState::List(series),
            Err(_) => AggState::Flat(series),
        };
        self
    }

    pub(crate) fn with_groups(&mut self, groups: GroupTuples) -> &mut Self {
        self.groups = Cow::Owned(groups);
        self
    }

    pub(crate) fn aggregated(&mut self) -> Cow<'_, Series> {
        // we do this here because of mutable borrow overlaps.
        // The groups are determined lazily and in case of a flat
        // series we use the groups to aggregate the list
        // because this is lazy, we first must to update the groups
        // by calling .groups()
        if let AggState::Flat(_) = self.series {
            self.groups();
        }
        match &self.series {
            AggState::Flat(s) => {
                let out = Cow::Owned(
                    s.agg_list(&self.groups)
                        .expect("should be able to aggregate this to list"),
                );

                if !self.sorted {
                    self.sorted = true;
                    self.update_groups = true;
                };
                out
            }
            AggState::List(s) => Cow::Borrowed(s),
            AggState::None => unreachable!(),
        }
    }

    pub(crate) fn flat(&self) -> Cow<'_, Series> {
        match &self.series {
            AggState::Flat(s) => Cow::Borrowed(s),
            AggState::List(s) => Cow::Owned(s.explode().unwrap()),
            AggState::None => unreachable!(),
        }
    }

    pub(crate) fn take(&mut self) -> Series {
        match std::mem::take(&mut self.series) {
            AggState::Flat(s) => s,
            AggState::List(s) => s,
            AggState::None => panic!("implementation error"),
        }
    }
}

/// Take a DataFrame and evaluate the expressions.
/// Implement this for Column, lt, eq, etc
pub trait PhysicalExpr: Send + Sync {
    fn as_expression(&self) -> &Expr {
        // for instance not needed for aggregations (for now)
        unimplemented!()
    }

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
        groups: &'a GroupTuples,
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
}

pub trait PhysicalAggregation: Send + Sync {
    #[allow(clippy::ptr_arg)]
    /// Should be called on the final aggregation node like sum, min, max, etc.
    /// When called on a tail, slice, sort, etc. it should return a list-array
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
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
        groups: &GroupTuples,
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
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        self.aggregate(final_df, groups, state)
    }
}
