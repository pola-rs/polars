use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use std::sync::Arc;

pub struct TernaryExpr {
    pub predicate: Arc<dyn PhysicalExpr>,
    pub truthy: Arc<dyn PhysicalExpr>,
    pub falsy: Arc<dyn PhysicalExpr>,
    pub expr: Expr,
}

impl PhysicalExpr for TernaryExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let mask_series = self.predicate.evaluate(df, state)?;
        let mask = mask_series.bool()?;
        let truthy = self.truthy.evaluate(df, state)?;
        let falsy = self.falsy.evaluate(df, state)?;
        truthy.zip_with(mask, &falsy)
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.truthy.to_field(input_schema)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let ac_mask = self.predicate.evaluate_on_groups(df, groups, state)?;
        let mask_s = ac_mask.flat();
        let mask = mask_s.bool()?;
        let mut ac_truthy = self.truthy.evaluate_on_groups(df, groups, state)?;
        let ac_falsy = self.falsy.evaluate_on_groups(df, groups, state)?;

        if !ac_truthy.can_combine(&ac_falsy) {
            return Err(PolarsError::InvalidOperation(
                "\
            cannot combine this ternary expression, the groups do not match"
                    .into(),
            ));
        }

        let out = ac_truthy.flat().zip_with(mask, ac_falsy.flat().as_ref())?;
        ac_truthy.with_series(out);

        Ok(ac_truthy)
    }
}
