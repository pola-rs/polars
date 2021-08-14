use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;

pub struct ShiftExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) periods: i64,
}

impl ShiftExpr {
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups_core<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<(Series, Cow<'a, GroupTuples>)> {
        let (s, groups) = self.input.evaluate_on_groups(df, groups, state)?;
        let out = s.agg_list(&groups).ok_or_else(|| {
            PolarsError::Other(
                "could not aggregate into a list in the groupby context. \
        Make sure your aggregation is not of dtype object/list?"
                    .into(),
            )
        })?;
        Ok((
            out.list()?.apply(|s| s.shift(self.periods)).into_series(),
            groups,
        ))
    }
}

impl PhysicalExpr for ShiftExpr {
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.input.evaluate(df, state)?;
        Ok(series.shift(self.periods))
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<(Series, Cow<'a, GroupTuples>)> {
        // The Series are aggregate per group, then the shift is applied.
        // Because an aggregation on a next level, e.g. sum will use the group tuples to aggregate
        // and sum we must explode the Series and update the group tuples to match the new Series.
        // Because we aggregate with the current group tuples, the Series is ordered by group.
        let (out, groups) = self.evaluate_on_groups_core(df, groups, state)?;

        // the groups are unordered
        // and the series is aggregated with this groups
        // so we need to recreate new grouptuples that
        // match the exploded Series
        let mut count = 0u32;
        let groups = groups
            .iter()
            .map(|g| {
                let add = g.1.len() as u32;
                let new_count = count + add;
                let out = (count, (count..new_count).collect::<Vec<_>>());
                count = new_count;
                out
            })
            .collect();

        // we explode again, because the final aggregation needs the group tuples to aggregate
        Ok((out.explode()?, Cow::Owned(groups)))
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.input.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}
impl PhysicalAggregation for ShiftExpr {
    // As a final aggregation a Shift returns a list array.
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let (s, _) = self.evaluate_on_groups_core(df, groups, state)?;
        Ok(Some(s))
    }
}
