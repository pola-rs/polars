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

trait NanNormalizer<T: Float> {
    fn normalize_func(actual_float: T) -> T {
        if actual_float.is_nan() {
            Self::java_nan()
        } else if actual_float == T::neg_zero() {
            T::zero()
        } else {
            actual_float
        }
    }

    fn java_nan() -> T;
}

impl NanNormalizer<f32> for f32 {
    fn java_nan() -> f32 {
        f32::from_bits(0x7FC00000)
    }
}

impl NanNormalizer<f64> for f64 {
    fn java_nan() -> f64 {
        f64::from_bits(0x7FF8000000000000)
    }
}

fn normalize_series_with_dtype<const IS_AGG: bool>(input_series: &Series) -> PolarsResult<Series> {
    Ok(match input_series.dtype() {
        DataType::Float32 => input_series.f32()?.iter().map(|item: Option<f32> | {
                item.map(f32::normalize_func)
            }).collect::<Series>(),
        DataType::Float64 => input_series.f64()?.iter().map(|item: Option<f64> | {
            item.map(f64::normalize_func)
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
        _ => Err(PolarsError::ComputeError("FlarionNormalizeNanAndZero only supports floating point numbers".into()))?
    })
}

pub struct FlarionNormalizeNanAndZeroExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) expr: Expr,
}

impl FlarionNormalizeNanAndZeroExpr {
    pub fn new(input: Arc<dyn PhysicalExpr>, expr: Expr) -> Self {
        Self { input, expr }
    }
}

impl PhysicalExpr for FlarionNormalizeNanAndZeroExpr {
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
            normalize_series_with_dtype::<false>(ac_s.series())?
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

#[cfg(test)]
mod tests {
    use polars_core::prelude::AnyValue;
    use polars_core::utils::Container;
    use polars_utils::pl_str::PlSmallStr;
    use super::*;

    #[test]
    fn test_normalizer_f32() {
        let f32_nan1 = f32::from_bits(0x7FC00001);
        let f32_nan2 = f32::from_bits(0x7FC00002);
        let f32_nan3 = f32::from_bits(0x7FC00003);

        assert_eq!(f32::normalize_func(f32_nan1).to_le_bytes(), f32::from_bits(0x7FC00000).to_le_bytes());
        assert_eq!(f32::normalize_func(f32_nan2).to_le_bytes(), f32::from_bits(0x7FC00000).to_le_bytes());
        assert_eq!(f32::normalize_func(f32_nan3).to_le_bytes(), f32::from_bits(0x7FC00000).to_le_bytes());

        let f32_neg_zero1 = f32::neg_zero();
        let f32_neg_zero2 = -0.0;
        assert_eq!(f32::normalize_func(f32_neg_zero1).to_le_bytes(), 0.0f32.to_le_bytes());
        assert_eq!(f32::normalize_func(f32_neg_zero2).to_le_bytes(), 0.0f32.to_le_bytes());
    }

    #[test]
    fn test_normalizer_f64() {
        let f64_nan1 = f64::from_bits(0x7FF8000000000000);
        let f64_nan2 = f64::from_bits(0x7FF8000000000001);
        let f64_nan3 = f64::from_bits(0x7FF8000000000002);

        assert_eq!(f64::normalize_func(f64_nan1).to_le_bytes(), f64::from_bits(0x7FF8000000000000).to_le_bytes());
        assert_eq!(f64::normalize_func(f64_nan2).to_le_bytes(), f64::from_bits(0x7FF8000000000000).to_le_bytes());
        assert_eq!(f64::normalize_func(f64_nan3).to_le_bytes(), f64::from_bits(0x7FF8000000000000).to_le_bytes());

        let f64_neg_zero1 = f64::neg_zero();
        let f64_neg_zero2 = -0.0;
        assert_eq!(f64::normalize_func(f64_neg_zero1).to_le_bytes(), 0.0f64.to_le_bytes());
        assert_eq!(f64::normalize_func(f64_neg_zero2).to_le_bytes(), 0.0f64.to_le_bytes());
    }

    #[test]
    fn test_normalize_series() {
        let original_f32 = vec![
            Some(1.0),
            Some(f32::from_bits(0x7FC00001)),
            None,
            Some(f32::from_bits(0x7FC00002)),
            Some(f32::from_bits(0x7FC00003))
        ];
        let f32_series = Series::from_iter(original_f32.clone());

        let original_f64 = vec![
            Some(1.0),
            Some(f64::from_bits(0x7FF8000000000000)),
            None,
            Some(f64::from_bits(0x7FF8000000000001)),
            Some(f64::from_bits(0x7FF8000000000002))
        ];
        let f64_series = Series::from_iter(original_f64.clone());

        let null_series = Series::new_null(PlSmallStr::from_static("empty"), 5);

        let f32_normalized = normalize_series_with_dtype::<false>(&f32_series).unwrap();
        let f32_normalized_values = f32_normalized.iter().map(|a| {
            match a {
                AnyValue::Float32(f) => Some(f),
                AnyValue::Null => None,
                _ => unreachable!()
            }
        }).collect::<Vec<_>>();
        original_f32.iter().zip(f32_normalized_values.iter()).for_each(|(a, b)| {
            if a.is_none() {
                assert_eq!(b, &None);
            } else if !a.unwrap().is_nan() {
                assert_eq!(a.unwrap(), b.unwrap());
            } else  {
                assert_eq!(b.unwrap().to_le_bytes(), f32::from_bits(0x7FC00000).to_le_bytes());
            }
        });


        let f64_normalized = normalize_series_with_dtype::<false>(&f64_series).unwrap();
        let f64_normalized_values = f64_normalized.iter().map(|a| {
            match a {
                AnyValue::Float64(d) => Some(d),
                AnyValue::Null => None,
                _ => unreachable!()
            }
        }).collect::<Vec<_>>();
        original_f64.iter().zip(f64_normalized_values.iter()).for_each(|(a, b)| {
            if a.is_none() {
                assert_eq!(b, &None);
            } else if !a.unwrap().is_nan() {
                assert_eq!(a.unwrap(), b.unwrap());
            } else  {
                assert_eq!(b.unwrap().to_le_bytes(), f64::from_bits(0x7FF8000000000000).to_le_bytes());
            }
        });

        let null_normalized = normalize_series_with_dtype::<false>(&null_series).unwrap();
        assert_eq!(null_normalized.len(), null_series.len());
        let null_normalized_values = null_normalized.iter().map(|a| {
            assert_eq!(a, AnyValue::Null)
        }).collect::<Vec<_>>();
    }
}