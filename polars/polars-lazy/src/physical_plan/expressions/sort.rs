use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use std::sync::Arc;

pub struct SortExpr {
    pub(crate) physical_expr: Arc<dyn PhysicalExpr>,
    pub(crate) reverse: bool,
    expr: Expr,
}

impl SortExpr {
    pub fn new(physical_expr: Arc<dyn PhysicalExpr>, reverse: bool, expr: Expr) -> Self {
        Self {
            physical_expr,
            reverse,
            expr,
        }
    }
}

impl PhysicalExpr for SortExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.physical_expr.evaluate(df, state)?;
        Ok(series.sort(self.reverse))
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let mut ac = self.physical_expr.evaluate_on_groups(df, groups, state)?;
        let series = ac.flat().into_owned();

        let groups = ac
            .groups
            .iter()
            .map(|(_first, idx)| {
                // Safety:
                // Group tuples are always in bounds
                let group =
                    unsafe { series.take_iter_unchecked(&mut idx.iter().map(|i| *i as usize)) };

                let sorted_idx = group.argsort(self.reverse);

                let new_idx: Vec<_> = sorted_idx
                    .cont_slice()
                    .unwrap()
                    .iter()
                    .map(|&i| {
                        debug_assert!(idx.get(i as usize).is_some());
                        unsafe { *idx.get_unchecked(i as usize) }
                    })
                    .collect();
                (new_idx[0], new_idx)
            })
            .collect();

        ac.with_groups(groups);

        Ok(ac)
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.physical_expr.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}
impl PhysicalAggregation for SortExpr {
    // As a final aggregation a Sort returns a list array.
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let mut ac = self.physical_expr.evaluate_on_groups(df, groups, state)?;
        let agg_s = ac.aggregated();
        let agg_s = agg_s
            .list()
            .unwrap()
            .apply_amortized(|s| s.as_ref().sort(self.reverse))
            .into_series();
        Ok(Some(agg_s))
    }
}
