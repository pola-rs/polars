use crate::physical_plan::expressions::utils::as_aggregated;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use std::sync::Arc;

pub struct AliasExpr {
    pub(crate) physical_expr: Arc<dyn PhysicalExpr>,
    pub(crate) name: Arc<str>,
    expr: Expr,
}

impl AliasExpr {
    pub fn new(physical_expr: Arc<dyn PhysicalExpr>, name: Arc<str>, expr: Expr) -> Self {
        Self {
            physical_expr,
            name,
            expr,
        }
    }
    fn finish(&self, mut input: Series) -> Result<Series> {
        input.rename(&self.name);
        Ok(input)
    }
}

impl PhysicalExpr for AliasExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.physical_expr.evaluate(df, state)?;
        self.finish(series)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let mut ac = self.physical_expr.evaluate_on_groups(df, groups, state)?;
        let mut s = ac.take();
        s.rename(&self.name);

        ac.with_series(self.finish(s)?, ac.is_aggregated());
        Ok(ac)
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        Ok(Field::new(
            &self.name,
            self.physical_expr
                .to_field(input_schema)?
                .data_type()
                .clone(),
        ))
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

impl PhysicalAggregation for AliasExpr {
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let opt_agg = as_aggregated(self.physical_expr.as_ref(), df, groups, state)?;
        Ok(opt_agg.map(|mut agg| {
            agg.rename(&self.name);
            agg
        }))
    }
}
