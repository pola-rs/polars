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
        use LiteralValue::*;
        let s = match &self.0 {
            #[cfg(feature = "dtype-i8")]
            Int8(v) => Int8Chunked::full(get_literal_name().clone(), *v, 1).into_column(),
            #[cfg(feature = "dtype-i16")]
            Int16(v) => Int16Chunked::full(get_literal_name().clone(), *v, 1).into_column(),
            Int32(v) => Int32Chunked::full(get_literal_name().clone(), *v, 1).into_column(),
            Int64(v) => Int64Chunked::full(get_literal_name().clone(), *v, 1).into_column(),
            #[cfg(feature = "dtype-i128")]
            Int128(v) => Int128Chunked::full(get_literal_name().clone(), *v, 1).into_column(),
            #[cfg(feature = "dtype-u8")]
            UInt8(v) => UInt8Chunked::full(get_literal_name().clone(), *v, 1).into_column(),
            #[cfg(feature = "dtype-u16")]
            UInt16(v) => UInt16Chunked::full(get_literal_name().clone(), *v, 1).into_column(),
            UInt32(v) => UInt32Chunked::full(get_literal_name().clone(), *v, 1).into_column(),
            UInt64(v) => UInt64Chunked::full(get_literal_name().clone(), *v, 1).into_column(),
            Float32(v) => Float32Chunked::full(get_literal_name().clone(), *v, 1).into_column(),
            Float64(v) => Float64Chunked::full(get_literal_name().clone(), *v, 1).into_column(),
            #[cfg(feature = "dtype-decimal")]
            Decimal(v, scale) => Int128Chunked::full(get_literal_name().clone(), *v, 1)
                .into_decimal_unchecked(None, *scale)
                .into_column(),
            Boolean(v) => BooleanChunked::full(get_literal_name().clone(), *v, 1).into_column(),
            Null => {
                polars_core::prelude::Series::new_null(get_literal_name().clone(), 1).into_column()
            },
            Range { low, high, dtype } => match dtype {
                DataType::Int32 => {
                    polars_ensure!(
                        *low >= i32::MIN as i64 && *high <= i32::MAX as i64,
                        ComputeError: "range not within bounds of `Int32`: [{}, {}]", *low, *high
                    );
                    let low = *low as i32;
                    let high = *high as i32;
                    let ca: NoNull<Int32Chunked> = (low..high).collect();
                    ca.into_inner().into_column()
                },
                DataType::Int64 => {
                    let low = *low;
                    let high = *high;
                    let ca: NoNull<Int64Chunked> = (low..high).collect();
                    ca.into_inner().into_column()
                },
                DataType::UInt32 => {
                    polars_ensure!(
                        *low >= 0 && *high <= u32::MAX as i64,
                        ComputeError: "range not within bounds of `UInt32`: [{}, {}]", *low, *high
                    );
                    let low = *low as u32;
                    let high = *high as u32;
                    let ca: NoNull<UInt32Chunked> = (low..high).collect();
                    ca.into_inner().into_column()
                },
                dt => polars_bail!(
                    InvalidOperation: "datatype `{}` is not supported as range", dt
                ),
            },
            String(v) => StringChunked::full(get_literal_name().clone(), v, 1).into_column(),
            Binary(v) => BinaryChunked::full(get_literal_name().clone(), v, 1).into_column(),
            #[cfg(feature = "dtype-datetime")]
            DateTime(timestamp, tu, tz) => {
                Int64Chunked::full(get_literal_name().clone(), *timestamp, 1)
                    .into_datetime(*tu, tz.clone())
                    .into_column()
            },
            #[cfg(feature = "dtype-duration")]
            Duration(v, tu) => Int64Chunked::full(get_literal_name().clone(), *v, 1)
                .into_duration(*tu)
                .into_column(),
            #[cfg(feature = "dtype-date")]
            Date(v) => Int32Chunked::full(get_literal_name().clone(), *v, 1)
                .into_date()
                .into_column(),
            #[cfg(feature = "dtype-time")]
            Time(v) => {
                if !(0..NANOSECONDS_IN_DAY).contains(v) {
                    polars_bail!(
                        InvalidOperation: "value `{v}` is out-of-range for `time` which can be 0 - {}",
                        NANOSECONDS_IN_DAY - 1
                    );
                }

                Int64Chunked::full(get_literal_name().clone(), *v, 1)
                    .into_time()
                    .into_column()
            },
            Series(series) => series.deref().clone().into_column(),
            OtherScalar(s) => s.clone().into_column(get_literal_name().clone()),
            lv @ (Int(_) | Float(_) | StrCat(_)) => polars_core::prelude::Series::from_any_values(
                get_literal_name().clone(),
                &[lv.to_any_value().unwrap()],
                false,
            )
            .unwrap()
            .into_column(),
        };
        Ok(s)
    }
}

impl PhysicalExpr for LiteralExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.1)
    }

    fn evaluate(&self, _df: &DataFrame, _state: &ExecutionState) -> PolarsResult<Column> {
        self.as_column()
    }

    fn evaluate_inline_impl(&self, _depth_limit: u8) -> Option<Column> {
        use LiteralValue::*;
        match &self.0 {
            Range { .. } => None,
            _ => self.as_column().ok(),
        }
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let s = self.evaluate(df, state)?;
        Ok(AggregationContext::from_literal(s, Cow::Borrowed(groups)))
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }

    fn collect_live_columns(&self, _lv: &mut PlIndexSet<PlSmallStr>) {}

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
