use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::arity::unary_elementwise_values;
use polars_core::prelude::*;
use polars_ops::prelude::lst_get;
use polars_ops::series::convert_to_unsigned_index;
use polars_utils::index::ToIdx;

use super::*;
use crate::expressions::{AggState, AggregationContext, PhysicalExpr, UpdateGroups};

pub struct GatherExpr {
    pub(crate) phys_expr: Arc<dyn PhysicalExpr>,
    pub(crate) idx: Arc<dyn PhysicalExpr>,
    pub(crate) expr: Expr,
    pub(crate) returns_scalar: bool,
}

impl PhysicalExpr for GatherExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let series = self.phys_expr.evaluate(df, state)?;
        let idx = self.idx.evaluate(df, state)?;
        let idx = convert_to_unsigned_index(idx.as_materialized_series(), series.len())?;
        series.take(&idx)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.phys_expr.evaluate_on_groups(df, groups, state)?;
        let mut idx = self.idx.evaluate_on_groups(df, groups, state)?;

        let ac_list = ac.aggregated_as_list();

        if self.returns_scalar {
            polars_ensure!(
                !matches!(idx.agg_state(), AggState::AggregatedList(_) | AggState::NotAggregated(_)),
                ComputeError: "expected single index"
            );

            // For returns_scalar=true, we can dispatch to `list.get`.
            let idx = idx.flat_naive();
            let idx = idx.cast(&DataType::Int64)?;
            let idx = idx.i64().unwrap();
            let taken = lst_get(ac_list.as_ref(), idx, true)?;

            ac.with_values_and_args(taken, true, Some(&self.expr), false, true)?;
            ac.with_update_groups(UpdateGroups::No);
            return Ok(ac);
        }

        // Cast the indices to
        // - IdxSize, if the idx only contains positive integers.
        // - Int64,   if the idx contains negative numbers.
        // This may give false positives if there are masked out elements.
        let idx = idx.aggregated_as_list();
        let idx = idx.apply_to_inner(&|s| match s.dtype() {
            dtype if dtype == &IDX_DTYPE => Ok(s),
            dtype if dtype.is_unsigned_integer() => {
                s.cast_with_options(&IDX_DTYPE, CastOptions::Strict)
            },

            dtype if dtype.is_signed_integer() => {
                let has_negative_integers = s.lt(0)?.any();
                if has_negative_integers && dtype == &DataType::Int64 {
                    Ok(s)
                } else if has_negative_integers {
                    s.cast_with_options(&DataType::Int64, CastOptions::Strict)
                } else {
                    s.cast_with_options(&IDX_DTYPE, CastOptions::Overflowing)
                }
            },
            _ => polars_bail!(
                op = "gather/get",
                got = s.dtype(),
                expected = "integer type"
            ),
        })?;

        let taken = if idx.inner_dtype() == &IDX_DTYPE {
            // Fast path: all indices are positive.

            ac_list
                .amortized_iter()
                .zip(idx.amortized_iter())
                .map(|(s, idx)| Some(s?.as_ref().take(idx?.as_ref().idx().unwrap())))
                .map(|opt_res| opt_res.transpose())
                .collect::<PolarsResult<ListChunked>>()?
                .with_name(ac.get_values().name().clone())
        } else {
            // Slower path: some indices may be negative.
            assert!(idx.inner_dtype() == &DataType::Int64);

            ac_list
                .amortized_iter()
                .zip(idx.amortized_iter())
                .map(|(s, idx)| {
                    let s = s?;
                    let idx = idx?;
                    let idx = idx.as_ref().i64().unwrap();
                    let target_len = s.as_ref().len() as u64;
                    let idx = unary_elementwise_values(idx, |v| v.to_idx(target_len));
                    Some(s.as_ref().take(&idx))
                })
                .map(|opt_res| opt_res.transpose())
                .collect::<PolarsResult<ListChunked>>()?
                .with_name(ac.get_values().name().clone())
        };

        ac.with_values(taken.into_column(), true, Some(&self.expr))?;
        ac.with_update_groups(UpdateGroups::WithSeriesLen);
        Ok(ac)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.phys_expr.to_field(input_schema)
    }

    fn is_scalar(&self) -> bool {
        self.returns_scalar
    }
}
