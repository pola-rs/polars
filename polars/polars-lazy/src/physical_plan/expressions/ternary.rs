use crate::physical_plan::state::{ExecutionState, StateFlags};
use crate::prelude::*;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use polars_core::POOL;
use std::sync::Arc;

pub struct TernaryExpr {
    predicate: Arc<dyn PhysicalExpr>,
    truthy: Arc<dyn PhysicalExpr>,
    falsy: Arc<dyn PhysicalExpr>,
    expr: Expr,
}

impl TernaryExpr {
    pub fn new(
        predicate: Arc<dyn PhysicalExpr>,
        truthy: Arc<dyn PhysicalExpr>,
        falsy: Arc<dyn PhysicalExpr>,
        expr: Expr,
    ) -> Self {
        Self {
            predicate,
            truthy,
            falsy,
            expr,
        }
    }
}

fn expand_lengths(truthy: &mut Series, falsy: &mut Series, mask: &mut BooleanChunked) {
    let len = std::cmp::max(std::cmp::max(truthy.len(), falsy.len()), mask.len());
    if len > 1 {
        if falsy.len() == 1 {
            *falsy = falsy.expand_at_index(0, len);
        }
        if truthy.len() == 1 {
            *truthy = truthy.expand_at_index(0, len);
        }
        if mask.len() == 1 {
            *mask = mask.expand_at_index(0, len);
        }
    }
}

fn finish_as_iters<'a>(
    mut ac_truthy: AggregationContext<'a>,
    mut ac_falsy: AggregationContext<'a>,
    mut ac_mask: AggregationContext<'a>,
) -> Result<AggregationContext<'a>> {
    let mut ca: ListChunked = ac_truthy
        .iter_groups()
        .zip(ac_falsy.iter_groups())
        .zip(ac_mask.iter_groups())
        .map(|((truthy, falsy), mask)| {
            match (truthy, falsy, mask) {
                (Some(truthy), Some(falsy), Some(mask)) => Some(
                    truthy
                        .as_ref()
                        .zip_with(mask.as_ref().bool()?, falsy.as_ref()),
                ),
                _ => None,
            }
            .transpose()
        })
        .collect::<Result<_>>()?;

    ca.rename(ac_truthy.series().name());
    // aggregation leaves only a single chunks
    let arr = ca.downcast_iter().next().unwrap();
    let list_vals_len = arr.values().len();
    let mut out = ca.into_series();

    if ac_truthy.arity_should_explode() && ac_falsy.arity_should_explode() && ac_mask.arity_should_explode() &&
        // exploded list should be equal to groups length
        list_vals_len == ac_truthy.groups.len()
    {
        out = out.explode()?
    }

    ac_truthy.with_series(out, true);
    Ok(ac_truthy)
}

impl PhysicalExpr for TernaryExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let mut state = state.split();
        // don't cache window functions as they run in parallel
        state.flags.remove(StateFlags::CACHE_WINDOW_EXPR);
        let mask_series = self.predicate.evaluate(df, &state)?;
        let mut mask = mask_series.bool()?.clone();

        let op_truthy = || self.truthy.evaluate(df, &state);
        let op_falsy = || self.falsy.evaluate(df, &state);

        let (truthy, falsy) = POOL.install(|| rayon::join(op_truthy, op_falsy));
        let mut truthy = truthy?;
        let mut falsy = falsy?;

        if truthy.is_empty() {
            return Ok(truthy);
        }
        if falsy.is_empty() {
            return Ok(falsy);
        }

        expand_lengths(&mut truthy, &mut falsy, &mut mask);

        truthy.zip_with(&mask, &falsy)
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.truthy.to_field(input_schema)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let op_mask = || self.predicate.evaluate_on_groups(df, groups, state);
        let op_truthy = || self.truthy.evaluate_on_groups(df, groups, state);
        let op_falsy = || self.falsy.evaluate_on_groups(df, groups, state);

        let (ac_mask, (ac_truthy, ac_falsy)) =
            POOL.install(|| rayon::join(op_mask, || rayon::join(op_truthy, op_falsy)));
        let ac_mask = ac_mask?;
        let mut ac_truthy = ac_truthy?;
        let ac_falsy = ac_falsy?;

        let mask_s = ac_mask.flat_naive();

        assert!(
            ac_truthy.can_combine(&ac_falsy),
            "cannot combine this ternary expression, the groups do not match"
        );

        dbg!(&ac_mask);
        use AggState::*;
        match dbg!((ac_truthy.agg_state(), ac_falsy.agg_state())) {
            // all branches are aggregated-flat or literal
            // mask -> aggregated-flat
            // truthy -> aggregated-flat | literal
            // falsy -> aggregated-flat | literal
            // simply align lengths and zip
            (
                Literal(truthy) | AggregatedFlat(truthy),
                AggregatedFlat(falsy) | Literal(falsy),
            )
            | (AggregatedList(truthy), AggregatedList(falsy))
            if matches!(ac_mask.agg_state(), AggState::AggregatedFlat(_))
            =>
                {
                    let mut truthy = truthy.clone();
                    let mut falsy = falsy.clone();
                    let mut mask = ac_mask.series().bool()?.clone();
                    expand_lengths(&mut truthy, &mut falsy, &mut mask);
                    let mut out = truthy.zip_with(&mask, &falsy).unwrap();
                    out.rename(truthy.name());
                    ac_truthy.with_series(out, true);
                    Ok(ac_truthy)
                }

            // Fallthrough from previous branch. That means the mask is not aggregated
            // and may be longer than branches, so we iterate over the groups
            (AggregatedFlat(agg), Literal(_)) |
            (Literal(_), AggregatedFlat(agg))
            // todo! check if we need to take this branch for those
            //| (AggregatedFlat(agg), NotAggregated(_)) |
            // (NotAggregated(_), AggregatedFlat(agg))

            // if the groups_len == df.len we can just apply all flat.
            // make sure that the order of the aggregation does not matter!
            // this is the case when we have a flat aggregation combined with a literal:
            // `sum(..) - lit(..)`.
            if agg.len() != df.height()
            => {
                finish_as_iters(ac_truthy, ac_falsy, ac_mask)
            }

            // Same branch as above, but without escape hatch for `num_groups == df.height()`

            // we cannot flatten a list because that changes the order, so we apply over groups
            (AggregatedList(_), NotAggregated(_)) |
            (NotAggregated(_), AggregatedList(_)) => {
                finish_as_iters(ac_truthy, ac_falsy, ac_mask)
            }

            // Both are or a flat series or aggregated into a list
            // so we can flatten the Series an apply the operators
            _ => {
                let mut mask = mask_s.bool()?.clone();
                let mut truthy = ac_truthy.flat_naive().into_owned();
                let mut falsy = ac_falsy.flat_naive().into_owned();
                expand_lengths(&mut truthy, &mut falsy, &mut mask);
                let out = truthy.zip_with(&mask, &falsy)?;

                ac_truthy.with_series(out, false);

                Ok(ac_truthy)
            }
        }
    }
    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }
}

impl PartitionedAggregation for TernaryExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Series> {
        let truthy = self.truthy.as_partitioned_aggregator().unwrap();
        let falsy = self.falsy.as_partitioned_aggregator().unwrap();
        let mask = self.predicate.as_partitioned_aggregator().unwrap();

        let mut truthy = truthy.evaluate_partitioned(df, groups, state)?;
        let mut falsy = falsy.evaluate_partitioned(df, groups, state)?;
        let mask = mask.evaluate_partitioned(df, groups, state)?;
        let mut mask = mask.bool()?.clone();

        expand_lengths(&mut truthy, &mut falsy, &mut mask);
        truthy.zip_with(&mask, &falsy)
    }

    fn finalize(
        &self,
        partitioned: Series,
        _groups: &GroupsProxy,
        _state: &ExecutionState,
    ) -> Result<Series> {
        Ok(partitioned)
    }
}
