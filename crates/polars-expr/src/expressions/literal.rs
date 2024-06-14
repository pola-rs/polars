use std::borrow::Cow;
use std::ops::Deref;

use polars_core::prelude::*;
use polars_core::utils::NoNull;
use polars_plan::constants::LITERAL_NAME;

use super::*;
use crate::expressions::{AggregationContext, PartitionedAggregation, PhysicalExpr};

pub struct LiteralExpr(pub LiteralValue, Expr);

impl LiteralExpr {
    pub fn new(value: LiteralValue, expr: Expr) -> Self {
        Self(value, expr)
    }
}

impl PhysicalExpr for LiteralExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.1)
    }
    fn evaluate(&self, _df: &DataFrame, _state: &ExecutionState) -> PolarsResult<Series> {
        use LiteralValue::*;
        let s = match &self.0 {
            #[cfg(feature = "dtype-i8")]
            Int8(v) => Int8Chunked::full(LITERAL_NAME, *v, 1).into_series(),
            #[cfg(feature = "dtype-i16")]
            Int16(v) => Int16Chunked::full(LITERAL_NAME, *v, 1).into_series(),
            Int32(v) => Int32Chunked::full(LITERAL_NAME, *v, 1).into_series(),
            Int64(v) => Int64Chunked::full(LITERAL_NAME, *v, 1).into_series(),
            #[cfg(feature = "dtype-u8")]
            UInt8(v) => UInt8Chunked::full(LITERAL_NAME, *v, 1).into_series(),
            #[cfg(feature = "dtype-u16")]
            UInt16(v) => UInt16Chunked::full(LITERAL_NAME, *v, 1).into_series(),
            UInt32(v) => UInt32Chunked::full(LITERAL_NAME, *v, 1).into_series(),
            UInt64(v) => UInt64Chunked::full(LITERAL_NAME, *v, 1).into_series(),
            Float32(v) => Float32Chunked::full(LITERAL_NAME, *v, 1).into_series(),
            Float64(v) => Float64Chunked::full(LITERAL_NAME, *v, 1).into_series(),
            #[cfg(feature = "dtype-decimal")]
            Decimal(v, scale) => Int128Chunked::full(LITERAL_NAME, *v, 1)
                .into_decimal_unchecked(None, *scale)
                .into_series(),
            Boolean(v) => BooleanChunked::full(LITERAL_NAME, *v, 1).into_series(),
            Null => polars_core::prelude::Series::new_null(LITERAL_NAME, 1),
            Range {
                low,
                high,
                data_type,
            } => match data_type {
                DataType::Int32 => {
                    polars_ensure!(
                        *low >= i32::MIN as i64 && *high <= i32::MAX as i64,
                        ComputeError: "range not within bounds of `Int32`: [{}, {}]", *low, *high
                    );
                    let low = *low as i32;
                    let high = *high as i32;
                    let ca: NoNull<Int32Chunked> = (low..high).collect();
                    ca.into_inner().into_series()
                },
                DataType::Int64 => {
                    let low = *low;
                    let high = *high;
                    let ca: NoNull<Int64Chunked> = (low..high).collect();
                    ca.into_inner().into_series()
                },
                DataType::UInt32 => {
                    polars_ensure!(
                        *low >= 0 && *high <= u32::MAX as i64,
                        ComputeError: "range not within bounds of `UInt32`: [{}, {}]", *low, *high
                    );
                    let low = *low as u32;
                    let high = *high as u32;
                    let ca: NoNull<UInt32Chunked> = (low..high).collect();
                    ca.into_inner().into_series()
                },
                dt => polars_bail!(
                    InvalidOperation: "datatype `{}` is not supported as range", dt
                ),
            },
            String(v) => StringChunked::full(LITERAL_NAME, v, 1).into_series(),
            Binary(v) => BinaryChunked::full(LITERAL_NAME, v, 1).into_series(),
            #[cfg(feature = "dtype-datetime")]
            DateTime(timestamp, tu, tz) => Int64Chunked::full(LITERAL_NAME, *timestamp, 1)
                .into_datetime(*tu, tz.clone())
                .into_series(),
            #[cfg(feature = "dtype-duration")]
            Duration(v, tu) => Int64Chunked::full(LITERAL_NAME, *v, 1)
                .into_duration(*tu)
                .into_series(),
            #[cfg(feature = "dtype-date")]
            Date(v) => Int32Chunked::full(LITERAL_NAME, *v, 1)
                .into_date()
                .into_series(),
            #[cfg(feature = "dtype-time")]
            Time(v) => Int64Chunked::full(LITERAL_NAME, *v, 1)
                .into_time()
                .into_series(),
            Series(series) => series.deref().clone(),
            lv @ (Int(_) | Float(_) | StrCat(_)) => polars_core::prelude::Series::from_any_values(
                LITERAL_NAME,
                &[lv.to_any_value().unwrap()],
                false,
            )
            .unwrap(),
        };
        Ok(s)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let s = self.evaluate(df, state)?;
        Ok(AggregationContext::from_literal(s, Cow::Borrowed(groups)))
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        let dtype = self.0.get_datatype();
        Ok(Field::new("literal", dtype))
    }
    fn is_literal(&self) -> bool {
        true
    }
}

impl PartitionedAggregation for LiteralExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        _groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        self.evaluate(df, state)
    }

    fn finalize(
        &self,
        partitioned: Series,
        _groups: &GroupsProxy,
        _state: &ExecutionState,
    ) -> PolarsResult<Series> {
        Ok(partitioned)
    }
}
