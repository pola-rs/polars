use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use std::sync::Arc;

pub struct UniqueExpr {
    pub(crate) physical_expr: Arc<dyn PhysicalExpr>,
    expr: Expr,
}

impl PhysicalExpr for UniqueExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.physical_expr.evaluate(df, state)?;
        series.unique()
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let mut ac = self.physical_expr.evaluate_on_groups(df, groups, state)?;
        let series = ac.flat_naive().into_owned();

        let groups = ac
            .groups
            .iter()
            .map(|(_first, idx)| {
                // Safety:
                // Group tuples are always in bounds
                let group =
                    unsafe { series.take_iter_unchecked(&mut idx.iter().map(|i| *i as usize)) };

                let unique_idx = group.arg_unique()?;

                let new_idx: Vec<_> = unique_idx
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
impl PhysicalAggregation for UniqueExpr {
}
