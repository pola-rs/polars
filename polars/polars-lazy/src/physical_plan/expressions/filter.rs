use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::{prelude::*, POOL};
use rayon::prelude::*;
use std::sync::Arc;

pub struct FilterExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) by: Arc<dyn PhysicalExpr>,
    expr: Expr,
}

impl FilterExpr {
    pub fn new(input: Arc<dyn PhysicalExpr>, by: Arc<dyn PhysicalExpr>, expr: Expr) -> Self {
        Self { input, by, expr }
    }
}

impl PhysicalExpr for FilterExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.input.evaluate(df, state)?;
        let predicate = self.by.evaluate(df, state)?;
        series.filter(predicate.bool()?)
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let mut ac_s = self.input.evaluate_on_groups(df, groups, state)?;
        let ac_predicate = self.by.evaluate_on_groups(df, groups, state)?;
        let groups = ac_s.groups();
        let predicate_s = ac_predicate.flat();
        let predicate = predicate_s.bool()?;

        let groups = POOL.install(|| {
            groups
                .par_iter()
                .map(|(first, idx)| {
                    let idx: Vec<u32> = idx
                        .iter()
                        .filter_map(|i| match predicate.get(*i as usize) {
                            Some(true) => Some(*i),
                            _ => None,
                        })
                        .collect();

                    (*idx.get(0).unwrap_or(first), idx)
                })
                .collect()
        });

        ac_s.with_groups(groups).set_original_len(false);
        Ok(ac_s)
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.input.to_field(input_schema)
    }
}
