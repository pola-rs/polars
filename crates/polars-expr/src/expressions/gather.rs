use num_traits::Zero;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::arity::unary_elementwise_values;
use polars_core::prelude::*;
use polars_core::with_match_physical_integer_polars_type;
use polars_ops::prelude::lst_get;
use polars_ops::series::convert_and_bound_index;

use super::*;
use crate::expressions::{AggState, AggregationContext, PhysicalExpr, UpdateGroups};

pub struct GatherExpr {
    pub(crate) phys_expr: Arc<dyn PhysicalExpr>,
    pub(crate) idx: Arc<dyn PhysicalExpr>,
    pub(crate) expr: Expr,
    pub(crate) returns_scalar: bool,
    pub(crate) null_on_oob: bool,
}

impl PhysicalExpr for GatherExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate_impl(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let series = self.phys_expr.evaluate(df, state)?;
        let idx = self.idx.evaluate(df, state)?;
        let idx =
            convert_and_bound_index(idx.as_materialized_series(), series.len(), self.null_on_oob)?;
        series.take(&idx)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups_impl<'a>(
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
            let taken = lst_get(ac_list.as_ref(), idx, self.null_on_oob)?;

            ac.with_values_and_args(taken, true, Some(&self.expr), false, true)?;
            ac.with_update_groups(UpdateGroups::No);
            return Ok(ac);
        }

        // Cast the indices to
        // - IdxSize, if they all fit.
        // - Int64,   if they all fit, e.g. if some are negative.
        // Else they keep their type. Masked out elements may give a slower path.
        let idx = idx.aggregated_as_list();
        let idx = if self.null_on_oob {
            idx.into_owned()
        } else {
            idx.apply_to_inner(&|s| {
                let dtype = s.dtype();
                polars_ensure!(
                    dtype.is_integer(),
                    op = "gather/get",
                    got = dtype,
                    expected = "integer type"
                );
                if dtype == &IDX_DTYPE {
                    return Ok(s);
                }
                // The range is checked first, so the casts don't wrap. Only scan for the
                // bounds that the dtype doesn't already give.
                let dtype_max = dtype.max()?.value().extract::<u128>().unwrap();
                let has_negative = dtype.is_signed_integer()
                    && with_match_physical_integer_polars_type!(dtype, |$T| {
                        has_negative::<$T>(s.as_ref().as_ref())
                    });
                if !has_negative {
                    let max = if dtype_max <= IdxSize::MAX as u128 {
                        dtype_max
                    } else {
                        s.max::<u128>()?.unwrap_or(0)
                    };
                    return if max <= IdxSize::MAX as u128 {
                        s.cast_with_options(&IDX_DTYPE, CastOptions::Overflowing)
                    } else if max <= i64::MAX as u128 {
                        s.cast_with_options(&DataType::Int64, CastOptions::Overflowing)
                    } else {
                        Ok(s)
                    };
                }
                let fits_i64 = dtype_max <= i64::MAX as u128
                    || with_match_physical_integer_polars_type!(dtype, |$T| {
                        let ca: &ChunkedArray<$T> = s.as_ref().as_ref();
                        ca.min_max().is_some_and(|(min, max)| {
                            i64::try_from(min).is_ok() && i64::try_from(max).is_ok()
                        })
                    });
                if fits_i64 {
                    s.cast_with_options(&DataType::Int64, CastOptions::Overflowing)
                } else {
                    Ok(s)
                }
            })?
        };

        let taken = if !self.null_on_oob && idx.inner_dtype() == &IDX_DTYPE {
            // Fast path: all indices are positive.
            ac_list
                .amortized_iter()
                .zip(idx.amortized_iter())
                .map(|(s, idx)| Some(s?.as_ref().take(idx?.as_ref().idx().unwrap())))
                .map(|opt_res| opt_res.transpose())
                .collect::<PolarsResult<ListChunked>>()?
                .with_name(ac.get_values().name().clone())
        } else if !self.null_on_oob && idx.inner_dtype() == &DataType::Int64 {
            // Slower path: some indices are negative.
            ac_list
                .amortized_iter()
                .zip(idx.amortized_iter())
                .map(|(s, idx)| {
                    let s = s?;
                    let idx = idx?;
                    let idx = idx.as_ref().i64().unwrap();
                    let len = s.as_ref().len() as i64;
                    // An index that is out of bounds becomes `len`, so `take` raises.
                    let idx = unary_elementwise_values(idx, |v| {
                        let v = if v < 0 { v + len } else { v };
                        (if (0..len).contains(&v) { v } else { len }) as IdxSize
                    });
                    Some(s.as_ref().take(&idx))
                })
                .map(|opt_res| opt_res.transpose())
                .collect::<PolarsResult<ListChunked>>()?
                .with_name(ac.get_values().name().clone())
        } else {
            ac_list
                .amortized_iter()
                .zip(idx.amortized_iter())
                .map(|(s, idx)| {
                    let s = s?;
                    let idx =
                        convert_and_bound_index(idx?.as_ref(), s.as_ref().len(), self.null_on_oob);
                    Some(idx.and_then(|idx| s.as_ref().take(&idx)))
                })
                .map(|opt_res| opt_res.transpose())
                .collect::<PolarsResult<ListChunked>>()?
                .with_name(ac.get_values().name().clone())
        };

        ac.with_agg_state(AggState::AggregatedList(taken.into_column()));
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

/// Also checks masked out values. Stops at the first block that has a negative value.
fn has_negative<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> bool {
    let zero = T::Native::zero();
    ca.downcast_iter().any(|arr| {
        arr.values()
            .chunks(1024)
            .any(|block| block.iter().fold(false, |acc, v| acc | (*v < zero)))
    })
}
