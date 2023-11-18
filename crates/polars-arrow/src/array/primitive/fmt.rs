#![allow(clippy::redundant_closure_call)]
use std::fmt::{Debug, Formatter, Result, Write};

use super::PrimitiveArray;
use crate::array::fmt::write_vec;
use crate::array::Array;
use crate::datatypes::{IntervalUnit, TimeUnit};
use crate::temporal_conversions;
use crate::types::{days_ms, i256, months_days_ns, NativeType};

macro_rules! dyn_primitive {
    ($array:expr, $ty:ty, $expr:expr) => {{
        let array = ($array as &dyn Array)
            .as_any()
            .downcast_ref::<PrimitiveArray<$ty>>()
            .unwrap();
        Box::new(move |f, index| write!(f, "{}", $expr(array.value(index))))
    }};
}

pub fn get_write_value<'a, T: NativeType, F: Write>(
    array: &'a PrimitiveArray<T>,
) -> Box<dyn Fn(&mut F, usize) -> Result + 'a> {
    use crate::datatypes::ArrowDataType::*;
    match array.data_type().to_logical_type() {
        Int8 => Box::new(|f, index| write!(f, "{}", array.value(index))),
        Int16 => Box::new(|f, index| write!(f, "{}", array.value(index))),
        Int32 => Box::new(|f, index| write!(f, "{}", array.value(index))),
        Int64 => Box::new(|f, index| write!(f, "{}", array.value(index))),
        UInt8 => Box::new(|f, index| write!(f, "{}", array.value(index))),
        UInt16 => Box::new(|f, index| write!(f, "{}", array.value(index))),
        UInt32 => Box::new(|f, index| write!(f, "{}", array.value(index))),
        UInt64 => Box::new(|f, index| write!(f, "{}", array.value(index))),
        Float16 => unreachable!(),
        Float32 => Box::new(|f, index| write!(f, "{}", array.value(index))),
        Float64 => Box::new(|f, index| write!(f, "{}", array.value(index))),
        Date32 => {
            dyn_primitive!(array, i32, temporal_conversions::date32_to_date)
        },
        Date64 => {
            dyn_primitive!(array, i64, temporal_conversions::date64_to_date)
        },
        Time32(TimeUnit::Second) => {
            dyn_primitive!(array, i32, temporal_conversions::time32s_to_time)
        },
        Time32(TimeUnit::Millisecond) => {
            dyn_primitive!(array, i32, temporal_conversions::time32ms_to_time)
        },
        Time32(_) => unreachable!(), // remaining are not valid
        Time64(TimeUnit::Microsecond) => {
            dyn_primitive!(array, i64, temporal_conversions::time64us_to_time)
        },
        Time64(TimeUnit::Nanosecond) => {
            dyn_primitive!(array, i64, temporal_conversions::time64ns_to_time)
        },
        Time64(_) => unreachable!(), // remaining are not valid
        Timestamp(time_unit, tz) => {
            if let Some(tz) = tz {
                let timezone = temporal_conversions::parse_offset(tz);
                match timezone {
                    Ok(timezone) => {
                        dyn_primitive!(array, i64, |time| {
                            temporal_conversions::timestamp_to_datetime(time, *time_unit, &timezone)
                        })
                    },
                    #[cfg(feature = "chrono-tz")]
                    Err(_) => {
                        let timezone = temporal_conversions::parse_offset_tz(tz);
                        match timezone {
                            Ok(timezone) => dyn_primitive!(array, i64, |time| {
                                temporal_conversions::timestamp_to_datetime(
                                    time, *time_unit, &timezone,
                                )
                            }),
                            Err(_) => {
                                let tz = tz.clone();
                                Box::new(move |f, index| {
                                    write!(f, "{} ({})", array.value(index), tz)
                                })
                            },
                        }
                    },
                    #[cfg(not(feature = "chrono-tz"))]
                    _ => {
                        let tz = tz.clone();
                        Box::new(move |f, index| write!(f, "{} ({})", array.value(index), tz))
                    },
                }
            } else {
                dyn_primitive!(array, i64, |time| {
                    temporal_conversions::timestamp_to_naive_datetime(time, *time_unit)
                })
            }
        },
        Interval(IntervalUnit::YearMonth) => {
            dyn_primitive!(array, i32, |x| format!("{x}m"))
        },
        Interval(IntervalUnit::DayTime) => {
            dyn_primitive!(array, days_ms, |x: days_ms| format!(
                "{}d{}ms",
                x.days(),
                x.milliseconds()
            ))
        },
        Interval(IntervalUnit::MonthDayNano) => {
            dyn_primitive!(array, months_days_ns, |x: months_days_ns| format!(
                "{}m{}d{}ns",
                x.months(),
                x.days(),
                x.ns()
            ))
        },
        Duration(TimeUnit::Second) => dyn_primitive!(array, i64, |x| format!("{x}s")),
        Duration(TimeUnit::Millisecond) => dyn_primitive!(array, i64, |x| format!("{x}ms")),
        Duration(TimeUnit::Microsecond) => dyn_primitive!(array, i64, |x| format!("{x}us")),
        Duration(TimeUnit::Nanosecond) => dyn_primitive!(array, i64, |x| format!("{x}ns")),
        Decimal(_, scale) => {
            // The number 999.99 has a precision of 5 and scale of 2
            let scale = *scale as u32;
            let factor = 10i128.pow(scale);
            let display = move |x: i128| {
                let base = x / factor;
                let decimals = (x - base * factor).abs();
                format!("{base}.{decimals}")
            };
            dyn_primitive!(array, i128, display)
        },
        Decimal256(_, scale) => {
            let scale = *scale as u32;
            let factor = (ethnum::I256::ONE * 10).pow(scale);
            let display = move |x: i256| {
                let base = x.0 / factor;
                let decimals = (x.0 - base * factor).abs();
                format!("{base}.{decimals}")
            };
            dyn_primitive!(array, i256, display)
        },
        _ => unreachable!(),
    }
}

impl<T: NativeType> Debug for PrimitiveArray<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let writer = get_write_value(self);

        write!(f, "{:?}", self.data_type())?;
        write_vec(f, &*writer, self.validity(), self.len(), "None", false)
    }
}
