use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use std::borrow::Cow;
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
    ) -> Result<(Series, Cow<'a, GroupTuples>)> {
        let (mask_series, mask_groups) = self.predicate.evaluate_on_groups(df, groups, state)?;
        let mask = mask_series.bool()?;
        let (truthy, truthy_groups) = self.truthy.evaluate_on_groups(df, groups, state)?;
        let (falsy, falsy_groups) = self.falsy.evaluate_on_groups(df, groups, state)?;

        match (&mask_groups, &truthy_groups, &falsy_groups) {
            (Cow::Borrowed(_), Cow::Borrowed(_), Cow::Borrowed(_)) => {
                // all groups are equal does not matter wich one we return
                Ok((truthy.zip_with(mask, &falsy)?, mask_groups))
            }
            _ => Err(PolarsError::InvalidOperation(
                "no filters allowed in ternary expressions".into(),
            )),
        }
    }
}
