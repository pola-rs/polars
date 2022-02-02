use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use polars_core::utils::{slice_offsets, CustomIterTools};
use std::sync::Arc;

pub struct SliceExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) offset: i64,
    pub(crate) len: usize,
    pub(crate) expr: Expr,
}

impl PhysicalExpr for SliceExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.input.evaluate(df, state)?;
        Ok(series.slice(self.offset, self.len))
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;
        let groups = ac.groups();

        let groups = match groups.as_ref() {
            GroupsProxy::Idx(groups) => {
                let groups = groups
                    .iter()
                    .map(|(first, idx)| {
                        let (offset, len) = slice_offsets(self.offset as i64, self.len, idx.len());
                        (first + offset as u32, idx[offset..offset + len].to_vec())
                    })
                    .collect();
                GroupsProxy::Idx(groups)
            }
            GroupsProxy::Slice(groups) => {
                let groups = groups
                    .iter()
                    .map(|&[first, len]| {
                        let (offset, len) =
                            slice_offsets(self.offset as i64, self.len, len as usize);
                        [first + offset as u32, len as u32]
                    })
                    .collect_trusted();
                GroupsProxy::Slice(groups)
            }
        };

        ac.with_groups(groups);

        Ok(ac)
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.input.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}
impl PhysicalAggregation for SliceExpr {
    // As a final aggregation a Slice returns a list array.
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
