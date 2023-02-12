use std::borrow::Cow;
use std::ops::Deref;

use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use polars_core::utils::NoNull;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

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
        const NAME: &str = "literal";
        use LiteralValue::*;
        let s = match &self.0 {
            #[cfg(feature = "dtype-i8")]
            Int8(v) => Int8Chunked::full(NAME, *v, 1).into_series(),
            #[cfg(feature = "dtype-i16")]
            Int16(v) => Int16Chunked::full(NAME, *v, 1).into_series(),
            Int32(v) => Int32Chunked::full(NAME, *v, 1).into_series(),
            Int64(v) => Int64Chunked::full(NAME, *v, 1).into_series(),
            #[cfg(feature = "dtype-u8")]
            UInt8(v) => UInt8Chunked::full(NAME, *v, 1).into_series(),
            #[cfg(feature = "dtype-u16")]
            UInt16(v) => UInt16Chunked::full(NAME, *v, 1).into_series(),
            UInt32(v) => UInt32Chunked::full(NAME, *v, 1).into_series(),
            UInt64(v) => UInt64Chunked::full(NAME, *v, 1).into_series(),
            Float32(v) => Float32Chunked::full(NAME, *v, 1).into_series(),
            Float64(v) => Float64Chunked::full(NAME, *v, 1).into_series(),
            Boolean(v) => BooleanChunked::full(NAME, *v, 1).into_series(),
            Null => polars_core::prelude::Series::new_null(NAME, 1),
            Range {
                low,
                high,
                data_type,
            } => match data_type {
                DataType::Int32 => {
                    let low = *low as i32;
                    let high = *high as i32;
                    let ca: NoNull<Int32Chunked> = (low..high).collect();
                    ca.into_inner().into_series()
                }
                DataType::Int64 => {
                    let low = *low;
                    let high = *high;
                    let ca: NoNull<Int64Chunked> = (low..high).collect();
                    ca.into_inner().into_series()
                }
                DataType::UInt32 => {
                    if *low >= 0 || *high <= u32::MAX as i64 {
                        return Err(PolarsError::ComputeError(
                            "range not within bounds of u32 type".into(),
                        ));
                    }
                    let low = *low as u32;
                    let high = *high as u32;
                    let ca: NoNull<UInt32Chunked> = (low..high).collect();
                    ca.into_inner().into_series()
                }
                dt => {
                    return Err(PolarsError::InvalidOperation(
                        format!("datatype {dt:?} not supported as range").into(),
                    ));
                }
            },
            Utf8(v) => Utf8Chunked::full(NAME, v, 1).into_series(),
            #[cfg(feature = "dtype-binary")]
            Binary(v) => BinaryChunked::full(NAME, v, 1).into_series(),
            #[cfg(feature = "temporal")]
            DateTime(ndt, tu) => {
                use polars_core::chunked_array::temporal::conversion::*;
                let timestamp = match tu {
                    TimeUnit::Nanoseconds => datetime_to_timestamp_ns(*ndt),
                    TimeUnit::Microseconds => datetime_to_timestamp_us(*ndt),
                    TimeUnit::Milliseconds => datetime_to_timestamp_ms(*ndt),
                };
                Int64Chunked::full(NAME, timestamp, 1)
                    .into_datetime(*tu, None)
                    .into_series()
            }
            #[cfg(feature = "temporal")]
            Duration(v, tu) => {
                let duration = match tu {
                    TimeUnit::Milliseconds => v.num_milliseconds(),
                    TimeUnit::Microseconds => match v.num_microseconds() {
                        Some(v) => v,
                        None => {
                            // Overflow
                            return Err(PolarsError::InvalidOperation(
                                format!("cannot represent {v:?} as {tu:?}").into(),
                            ));
                        }
                    },
                    TimeUnit::Nanoseconds => {
                        match v.num_nanoseconds() {
                            Some(v) => v,
                            None => {
                                // Overflow
                                return Err(PolarsError::InvalidOperation(
                                    format!("cannot represent {v:?} as {tu:?}").into(),
                                ));
                            }
                        }
                    }
                };
                Int64Chunked::full(NAME, duration, 1)
                    .into_duration(*tu)
                    .into_series()
            }
            Series(series) => series.deref().clone(),
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
    fn is_valid_aggregation(&self) -> bool {
        // literals can be both
        true
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
