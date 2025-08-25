use std::borrow::Cow;
use std::ops::Deref;

use arrow::temporal_conversions::NANOSECONDS_IN_DAY;
use polars_core::prelude::*;
use polars_core::utils::NoNull;
use polars_plan::constants::get_literal_name;

use super::*;
use crate::expressions::{AggregationContext, PartitionedAggregation, PhysicalExpr};

pub struct LiteralExpr(pub LiteralValue, Expr);

impl LiteralExpr {
    pub fn new(value: LiteralValue, expr: Expr) -> Self {
        Self(value, expr)
    }

    fn as_column(&self) -> PolarsResult<Column> {
        use LiteralValue as L;
        let column = match &self.0 {
            L::Scalar(sc) => {
                #[cfg(feature = "dtype-time")]
                if let AnyValue::Time(v) = sc.value() {
                    if !(0..NANOSECONDS_IN_DAY).contains(v) {
                        polars_bail!(
                            InvalidOperation: "value `{v}` is out-of-range for `time` which can be 0 - {}",
                            NANOSECONDS_IN_DAY - 1
                        );
                    }
                }

                sc.clone().into_column(get_literal_name().clone())
            },
            L::Series(s) => s.deref().clone().into_column(),
            lv @ L::Dyn(_) => polars_core::prelude::Series::from_any_values(
                get_literal_name().clone(),
                &[lv.to_any_value().unwrap()],
                false,
            )
            .unwrap()
            .into_column(),
            L::Range(RangeLiteralValue { low, high, dtype }) => {
                let low = *low;
                let high = *high;
                match dtype {
                    DataType::Int32 => {
                        polars_ensure!(
                            low >= i32::MIN as i128 && high <= i32::MAX as i128,
                            ComputeError: "range not within bounds of `Int32`: [{}, {}]", low, high
                        );
                        let low = low as i32;
                        let high = high as i32;
                        let ca: NoNull<Int32Chunked> = (low..high).collect();
                        ca.into_inner().into_column()
                    },
                    DataType::Int64 => {
                        polars_ensure!(
                            low >= i64::MIN as i128 && high <= i64::MAX as i128,
                            ComputeError: "range not within bounds of `Int32`: [{}, {}]", low, high
                        );
                        let low = low as i64;
                        let high = high as i64;
                        let ca: NoNull<Int64Chunked> = (low..high).collect();
                        ca.into_inner().into_column()
                    },
                    DataType::UInt32 => {
                        polars_ensure!(
                            low >= u32::MIN as i128 && high <= u32::MAX as i128,
                            ComputeError: "range not within bounds of `UInt32`: [{}, {}]", low, high
                        );
                        let low = low as u32;
                        let high = high as u32;
                        let ca: NoNull<UInt32Chunked> = (low..high).collect();
                        ca.into_inner().into_column()
                    },
                    dt => polars_bail!(
                        InvalidOperation: "datatype `{}` is not supported as range", dt
                    ),
                }
            },
        };
        Ok(column)
    }
}

impl PhysicalExpr for LiteralExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.1)
    }

    fn evaluate(&self, _df: &DataFrame, _state: &ExecutionState) -> PolarsResult<Column> {
        self.as_column()
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
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
            let groups = GroupsType::Slice {
                groups: (0..groups.len() as IdxSize)
                    .map(|i| [i * lit_length, lit_length])
                    .collect(),
                rolling: false,
            };
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

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        let dtype = self.0.get_datatype();
        Ok(Field::new(PlSmallStr::from_static("literal"), dtype))
    }
    fn is_literal(&self) -> bool {
        true
    }

    fn is_scalar(&self) -> bool {
        self.0.is_scalar()
    }
}

impl PartitionedAggregation for LiteralExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        _groups: &GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        self.evaluate(df, state)
    }

    fn finalize(
        &self,
        partitioned: Column,
        _groups: &GroupPositions,
        _state: &ExecutionState,
    ) -> PolarsResult<Column> {
        Ok(partitioned)
    }
}
