use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;

pub struct SortByExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) by: Arc<dyn PhysicalExpr>,
    pub(crate) reverse: bool,
    pub(crate) expr: Expr,
}

impl SortByExpr {
    pub fn new(
        input: Arc<dyn PhysicalExpr>,
        by: Arc<dyn PhysicalExpr>,
        reverse: bool,
        expr: Expr,
    ) -> Self {
        Self {
            input,
            by,
            reverse,
            expr,
        }
    }
}

impl PhysicalExpr for SortByExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.input.evaluate(df, state)?;
        let series_sort_by = self.by.evaluate(df, state)?;
        let sorted_idx = series_sort_by.argsort(self.reverse);

        // Safety:
        // sorted index are within bounds
        unsafe { series.take_unchecked(&sorted_idx) }
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<(Series, Cow<'a, GroupTuples>)> {
        let (series, _) = self.input.evaluate_on_groups(df, groups, state)?;
        let (series_sort_by, groups) = self.by.evaluate_on_groups(df, groups, state)?;

        let groups = groups
            .iter()
            .map(|(_first, idx)| {
                // Safety:
                // Group tuples are always in bounds
                let group = unsafe {
                    series_sort_by.take_iter_unchecked(&mut idx.iter().map(|i| *i as usize))
                };

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

        Ok((series, Cow::Owned(groups)))
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.input.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}
impl PhysicalAggregation for SortByExpr {
    // As a final aggregation a Sort returns a list array.
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let s = self.input.evaluate(df, state)?;
        let s_sort_by = self.by.evaluate(df, state)?;

        let s_sort_by = s_sort_by.agg_list(groups).ok_or_else(|| {
            PolarsError::Other(format!("cannot aggregate {:?} as list array", self.expr).into())
        })?;

        let agg_s = s.agg_list(groups);
        let out = agg_s.map(|s| {
            s.list()
                .unwrap()
                .into_iter()
                .zip(s_sort_by.list().unwrap())
                .map(|(opt_s, opt_sort_by)| {
                    match (opt_s, opt_sort_by) {
                        (Some(s), Some(sort_by)) => {
                            let sorted_idx = sort_by.argsort(self.reverse);
                            // Safety:
                            // sorted index are within bounds
                            unsafe { s.take_unchecked(&sorted_idx) }.ok()
                        }
                        _ => None,
                    }
                })
                .collect::<ListChunked>()
                .into_series()
        });
        Ok(out)
    }
}
