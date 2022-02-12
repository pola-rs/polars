use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PhysicalAggregation;
use crate::prelude::*;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use polars_core::utils::NoNull;
use std::borrow::Cow;
use std::ops::Deref;

pub struct LiteralExpr(pub LiteralValue, Expr);

impl LiteralExpr {
    pub fn new(value: LiteralValue, expr: Expr) -> Self {
        Self(value, expr)
    }
}

impl PhysicalExpr for LiteralExpr {
    fn as_expression(&self) -> &Expr {
        &self.1
    }
    fn evaluate(&self, _df: &DataFrame, _state: &ExecutionState) -> Result<Series> {
        use LiteralValue::*;
        let s = match &self.0 {
            #[cfg(feature = "dtype-i8")]
            Int8(v) => Int8Chunked::full("literal", *v, 1).into_series(),
            #[cfg(feature = "dtype-i16")]
            Int16(v) => Int16Chunked::full("literal", *v, 1).into_series(),
            Int32(v) => Int32Chunked::full("literal", *v, 1).into_series(),
            Int64(v) => Int64Chunked::full("literal", *v, 1).into_series(),
            #[cfg(feature = "dtype-u8")]
            UInt8(v) => UInt8Chunked::full("literal", *v, 1).into_series(),
            #[cfg(feature = "dtype-u16")]
            UInt16(v) => UInt16Chunked::full("literal", *v, 1).into_series(),
            UInt32(v) => UInt32Chunked::full("literal", *v, 1).into_series(),
            UInt64(v) => UInt64Chunked::full("literal", *v, 1).into_series(),
            Float32(v) => Float32Chunked::full("literal", *v, 1).into_series(),
            Float64(v) => Float64Chunked::full("literal", *v, 1).into_series(),
            Boolean(v) => BooleanChunked::full("literal", *v, 1).into_series(),
            Null => BooleanChunked::new("literal", &[None]).into_series(),
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
                    let low = *low as i64;
                    let high = *high as i64;
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
                        format!("datatype {:?} not supported as range", dt).into(),
                    ));
                }
            },
            Utf8(v) => Utf8Chunked::full("literal", v, 1).into_series(),
            #[cfg(all(feature = "temporal", feature = "dtype-datetime"))]
            DateTime(ndt, tu) => {
                use polars_core::chunked_array::temporal::conversion::*;
                let timestamp = match tu {
                    TimeUnit::Nanoseconds => datetime_to_timestamp_ns(*ndt),
                    TimeUnit::Microseconds => datetime_to_timestamp_us(*ndt),
                    TimeUnit::Milliseconds => datetime_to_timestamp_ms(*ndt),
                };
                Int64Chunked::full("literal", timestamp, 1)
                    .into_datetime(*tu, None)
                    .into_series()
            }
            #[cfg(all(feature = "temporal", feature = "dtype-duration"))]
            Duration(v, tu) => {
                let duration = match tu {
                    TimeUnit::Milliseconds => v.num_milliseconds(),
                    TimeUnit::Microseconds => match v.num_microseconds() {
                        Some(v) => v,
                        None => {
                            // Overflow
                            return Err(PolarsError::InvalidOperation(
                                format!("cannot represent {:?} as {:?}", v, tu).into(),
                            ));
                        }
                    },
                    TimeUnit::Nanoseconds => {
                        match v.num_nanoseconds() {
                            Some(v) => v,
                            None => {
                                // Overflow
                                return Err(PolarsError::InvalidOperation(
                                    format!("cannot represent {:?} as {:?}", v, tu).into(),
                                ));
                            }
                        }
                    }
                };
                Int64Chunked::full("literal", duration, 1)
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
    ) -> Result<AggregationContext<'a>> {
        let s = self.evaluate(df, state)?;
        Ok(AggregationContext::from_literal(s, Cow::Borrowed(groups)))
    }

    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        use LiteralValue::*;
        let name = "literal";
        let field = match &self.0 {
            #[cfg(feature = "dtype-i8")]
            Int8(_) => Field::new(name, DataType::Int8),
            #[cfg(feature = "dtype-i16")]
            Int16(_) => Field::new(name, DataType::Int16),
            Int32(_) => Field::new(name, DataType::Int32),
            Int64(_) => Field::new(name, DataType::Int64),
            #[cfg(feature = "dtype-u8")]
            UInt8(_) => Field::new(name, DataType::UInt8),
            #[cfg(feature = "dtype-u16")]
            UInt16(_) => Field::new(name, DataType::UInt16),
            UInt32(_) => Field::new(name, DataType::UInt32),
            UInt64(_) => Field::new(name, DataType::UInt64),
            Float32(_) => Field::new(name, DataType::Float32),
            Float64(_) => Field::new(name, DataType::Float64),
            Boolean(_) => Field::new(name, DataType::Boolean),
            Utf8(_) => Field::new(name, DataType::Utf8),
            Null => Field::new(name, DataType::Null),
            Range { data_type, .. } => Field::new(name, data_type.clone()),
            #[cfg(all(feature = "temporal", feature = "dtype-datetime"))]
            DateTime(_, tu) => Field::new(name, DataType::Datetime(*tu, None)),
            #[cfg(all(feature = "temporal", feature = "dtype-duration"))]
            Duration(_, tu) => Field::new(name, DataType::Duration(*tu)),
            Series(s) => s.field().into_owned(),
        };
        Ok(field)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

impl PhysicalAggregation for LiteralExpr {
    fn aggregate(
        &self,
        df: &DataFrame,
        _groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        PhysicalExpr::evaluate(self, df, state).map(Some)
    }
}
