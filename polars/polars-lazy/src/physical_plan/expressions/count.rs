use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_arrow::utils::CustomIterTools;
use polars_core::prelude::*;
use polars_core::utils::NoNull;
use std::borrow::Cow;

pub struct CountExpr {
    expr: Expr,
}

impl CountExpr {
    pub(crate) fn new() -> Self {
        Self { expr: Expr::Count }
    }
}

impl PhysicalExpr for CountExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, _state: &ExecutionState) -> Result<Series> {
        Ok(Series::new("count", [df.height() as IdxSize]))
    }

    fn evaluate_on_groups<'a>(
        &self,
        _df: &DataFrame,
        groups: &'a GroupsProxy,
        _state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let mut ca = match groups {
            GroupsProxy::Idx(groups) => {
                let ca: NoNull<IdxCa> = groups
                    .all()
                    .iter()
                    .map(|g| g.len() as IdxSize)
                    .collect_trusted();
                ca.into_inner()
            }
            GroupsProxy::Slice(groups) => {
                let ca: NoNull<IdxCa> = groups.iter().map(|g| g[1]).collect_trusted();
                ca.into_inner()
            }
        };
        ca.rename("count");
        let s = ca.into_series();

        Ok(AggregationContext::new(s, Cow::Borrowed(groups), true))
    }
    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        Ok(Field::new("count", DataType::UInt32))
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

impl PhysicalAggregation for CountExpr {
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let mut ac = self.evaluate_on_groups(df, groups, state)?;
        let s = ac.aggregated();
        Ok(Some(s))
    }
}
