use std::borrow::Cow;

use polars_core::prelude::*;
use polars_plan::constants::get_literal_name;

use super::*;
use crate::expressions::{AggregationContext, PhysicalExpr};

pub struct LiteralExpr(pub LiteralValue, Expr);

impl LiteralExpr {
    pub fn new(value: LiteralValue, expr: Expr) -> Self {
        Self(value, expr)
    }

    fn as_column(&self) -> PolarsResult<Column> {
        self.0.to_column(get_literal_name())
    }
}

impl PhysicalExpr for LiteralExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.1)
    }

    fn evaluate_impl(&self, _df: &DataFrame, _state: &ExecutionState) -> PolarsResult<Column> {
        self.as_column()
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups_impl<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let s = self.evaluate(df, state)?;

        if self.0.is_scalar() {
            Ok(AggregationContext::from_agg_state(
                AggState::LiteralScalar(s),
                Cow::Borrowed(groups),
            ))
        } else {
            // A non-scalar literal value expands to those values for every group.

            let lit_length = s.len() as IdxSize;
            polars_ensure!(
                (groups.len() as IdxSize).checked_mul(lit_length).is_some(),
                bigidx,
                ctx = "group_by",
                size = groups.len() as u64 * lit_length as u64
            );
            let groups = (0..groups.len() as IdxSize)
                .map(|i| [i * lit_length, lit_length])
                .collect();
            let groups = GroupsType::new_slice(groups, false, true);

            let agg_state = AggState::AggregatedList(Column::new_scalar(
                s.name().clone(),
                Scalar::new_list(s.take_materialized_series()),
                groups.len(),
            ));

            let groups = groups.into_sliceable();
            Ok(AggregationContext::from_agg_state(
                agg_state,
                Cow::Owned(groups),
            ))
        }
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        let dtype = self.0.get_datatype();
        Ok(Field::new(get_literal_name(), dtype))
    }
    fn is_literal(&self) -> bool {
        true
    }

    fn is_scalar(&self) -> bool {
        self.0.is_scalar()
    }
}
