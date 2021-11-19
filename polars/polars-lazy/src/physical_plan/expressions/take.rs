use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use std::sync::Arc;

pub struct TakeExpr {
    pub(crate) phys_expr: Arc<dyn PhysicalExpr>,
    pub(crate) idx: Arc<dyn PhysicalExpr>,
    pub(crate) expr: Expr,
}

impl TakeExpr {
    fn finish(&self, df: &DataFrame, state: &ExecutionState, series: Series) -> Result<Series> {
        let idx = self.idx.evaluate(df, state)?.cast(&DataType::UInt32)?;
        let idx_ca = idx.u32()?;

        series.take(idx_ca)
    }
}

impl PhysicalExpr for TakeExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.phys_expr.evaluate(df, state)?;
        self.finish(df, state, series)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let mut ac = self.phys_expr.evaluate_on_groups(df, groups, state)?;
        let idx = self.idx.evaluate(df, state)?.cast(&DataType::UInt32)?;
        let idx_ca = idx.u32()?;

        let taken = ac
            .aggregated()
            .list()
            .unwrap()
            .try_apply_amortized(|s| s.as_ref().take(idx_ca))?;

        ac.with_update_groups(UpdateGroups::WithSeriesLen)
            .with_series(taken.into_series(), true);
        Ok(ac)
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.phys_expr.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

impl PhysicalAggregation for TakeExpr {
    // As a final aggregation a Sort returns a list array.
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let mut ac = self.phys_expr.evaluate_on_groups(df, groups, state)?;
        let idx = self.idx.evaluate_on_groups(df, groups, state)?;
        let idx = idx.series();

        let mut all_unit_length = true;
        let mut taken = if let Ok(idx) = idx.list() {
            // cast the indices up front.
            let idx = idx.cast(&DataType::List(Box::new(DataType::UInt32)))?;

            let idx = idx.list().unwrap();
            let ca: ListChunked = ac
                .aggregated()
                .list()?
                .into_iter()
                .zip(idx.into_iter())
                .map(|(opt_s, opt_idx)| {
                    if let (Some(s), Some(idx)) = (opt_s, opt_idx) {
                        let idx = idx.u32()?;
                        let s = s.take(idx)?;
                        if s.len() != 1 {
                            all_unit_length = false;
                        }
                        Ok(Some(s))
                    } else {
                        Ok(None)
                    }
                })
                .collect::<Result<_>>()?;
            ca
        } else {
            let idx = idx.cast(&DataType::UInt32)?;
            let idx_ca = idx.u32()?;

            ac.aggregated().list().unwrap().try_apply_amortized(|s| {
                match s.as_ref().take(idx_ca) {
                    Ok(s) => {
                        if s.len() != 1 {
                            all_unit_length = false;
                        }
                        Ok(s)
                    }
                    e => e,
                }
            })?
        };

        taken.rename(ac.series().name());

        if all_unit_length {
            let s = taken.explode()?;
            Ok(Some(s))
        } else {
            Ok(Some(taken.into_series()))
        }
    }
}
