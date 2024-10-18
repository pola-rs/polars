use std::sync::Arc;
use arrow::legacy::error::PolarsResult;
use polars_core::datatypes::{DataType, Field, ListChunked};
use polars_core::error::PolarsError;
use polars_core::export::num::Float;
use polars_core::frame::DataFrame;
use polars_core::POOL;
use polars_core::prelude::{GroupsProxy, Schema, Series};
use polars_plan::dsl::Expr;
use crate::expressions::{AggregationContext, PhysicalExpr};
use crate::prelude::ExecutionState;

fn normalize_func<T: Float>(actual_float: T) -> T {
    if actual_float.is_nan() {
        T::nan()
    } else if actual_float == T::neg_zero() {
        T::zero()
    } else {
        actual_float
    }
}

fn normalize_series_with_dtype<const IS_AGG: bool>(input_series: &Series) -> PolarsResult<Series> {
    Ok(match input_series.dtype() {
        DataType::Float32 => input_series.f32()?.iter().map(|item: Option<f32> | {
                item.map(normalize_func)
            }).collect::<Series>(),
        DataType::Float64 => input_series.f64()?.iter().map(|item: Option<f64> | {
            item.map(normalize_func)
        }).collect::<Series>(),
        DataType::List(inner) if IS_AGG && inner.is_float() => {
            let normalized_list = input_series.list()?;
            Series::from(ListChunked::from_iter(normalized_list
                .into_iter()
                .map(|maybe_series| {
                    match maybe_series {
                        Some(inner_series) => {
                            normalize_series_with_dtype::<false>(&inner_series).map(Some)
                        },
                        None => Ok(None)
                    }
                }).collect::<PolarsResult<Vec<Option<_>>>>()?))
        }
        DataType::Null => input_series.clone(),
        _ => Err(PolarsError::ComputeError("NormalizeNanAndZero only supports floating point numbers".into()))?
    })
}

pub struct NormalizeNanAndZeroExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) expr: Expr,
}

impl NormalizeNanAndZeroExpr {
    pub fn new(input: Arc<dyn PhysicalExpr>, expr: Expr) -> Self {
        Self { input, expr }
    }
}

impl PhysicalExpr for NormalizeNanAndZeroExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let s_f = || self.input.evaluate(df, state);

        let series= POOL.install(s_f)?;

        normalize_series_with_dtype::<false>(&series)
    }

    fn evaluate_on_groups<'a>(&self, df: &DataFrame, groups: &'a GroupsProxy, state: &ExecutionState) -> PolarsResult<AggregationContext<'a>> {
        let ac_s_f = || self.input.evaluate_on_groups(df, groups, state);
        let mut ac_s: AggregationContext = POOL.install(ac_s_f)?;

        let is_aggregated = ac_s.is_aggregated();
        let new_series = if is_aggregated {
            // If the series' dtype is List, its ok, bc its an aggregate, we just need to operated on each subseries
            normalize_series_with_dtype::<true>(&ac_s.aggregated())?
        } else {
            // If the series' dtype is List, not ok, should be a float
            normalize_series_with_dtype::<false>(&ac_s.series())?
        };
        ac_s.with_series(new_series, is_aggregated, None)?;

        Ok(ac_s)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema)
    }

    fn is_scalar(&self) -> bool {
        true
    }
}