// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Defines temporal kernels for time and date related functions.

use jiff::civil::{DateTime, Time};
use jiff::tz::TimeZone;
use polars_error::PolarsResult;

use super::arity::unary;
use crate::array::*;
use crate::datatypes::*;
use crate::temporal_conversions::*;
use crate::types::NativeType;

// Macro to avoid repetition in functions, that apply
// civil-datetime field extraction on Arrays
macro_rules! date_like {
    ($extract:ident, $array:ident, $dtype:path) => {
        match $array.dtype().to_storage() {
            ArrowDataType::Date32 | ArrowDataType::Date64 | ArrowDataType::Timestamp(_, None) => {
                date_variants($array, $dtype, |x| x.$extract().try_into().unwrap())
            },
            ArrowDataType::Timestamp(time_unit, Some(timezone_str)) => {
                let array = $array.as_any().downcast_ref().unwrap();
                let timezone = parse_timezone(timezone_str.as_str())?;
                Ok(extract_impl(array, *time_unit, &timezone, |x| {
                    x.$extract().try_into().unwrap()
                }))
            },
            _ => unimplemented!(),
        }
    };
}

/// Extracts the years of a temporal array as [`PrimitiveArray<i32>`].
pub fn year(array: &dyn Array) -> PolarsResult<PrimitiveArray<i32>> {
    date_like!(year, array, ArrowDataType::Int32)
}

/// Extracts the months of a temporal array as [`PrimitiveArray<i8>`].
///
/// Value ranges from 1 to 12.
pub fn month(array: &dyn Array) -> PolarsResult<PrimitiveArray<i8>> {
    date_like!(month, array, ArrowDataType::Int8)
}

/// Extracts the days of a temporal array as [`PrimitiveArray<i8>`].
///
/// Value ranges from 1 to 32 (Last day depends on month).
pub fn day(array: &dyn Array) -> PolarsResult<PrimitiveArray<i8>> {
    date_like!(day, array, ArrowDataType::Int8)
}

// Extension traits so the `date_like!` macro can call `.i8_weekday()` /
// `.i8_iso_week()` on a civil `DateTime` the same way it calls `.year()` etc.
trait Int8Weekday {
    fn i8_weekday(&self) -> i8;
}
impl Int8Weekday for DateTime {
    fn i8_weekday(&self) -> i8 {
        self.weekday().to_monday_one_offset()
    }
}

trait Int8IsoWeek {
    fn i8_iso_week(&self) -> i8;
}
impl Int8IsoWeek for DateTime {
    fn i8_iso_week(&self) -> i8 {
        self.iso_week_date().week()
    }
}

/// Extracts weekday of a temporal array as [`PrimitiveArray<i8>`].
///
/// Monday is 1, Tuesday is 2, ..., Sunday is 7.
pub fn weekday(array: &dyn Array) -> PolarsResult<PrimitiveArray<i8>> {
    date_like!(i8_weekday, array, ArrowDataType::Int8)
}

/// Extracts ISO week of a temporal array as [`PrimitiveArray<i8>`].
///
/// Value ranges from 1 to 53 (Last week depends on the year).
pub fn iso_week(array: &dyn Array) -> PolarsResult<PrimitiveArray<i8>> {
    date_like!(i8_iso_week, array, ArrowDataType::Int8)
}

// Macro to avoid repetition in functions, that apply
// civil-time field extraction on Arrays
macro_rules! time_like {
    ($extract:ident, $array:ident, $dtype:path) => {
        match $array.dtype().to_storage() {
            ArrowDataType::Date32 | ArrowDataType::Date64 | ArrowDataType::Timestamp(_, None) => {
                date_variants($array, $dtype, |x| x.$extract().try_into().unwrap())
            },
            ArrowDataType::Time32(_) | ArrowDataType::Time64(_) => {
                time_variants($array, ArrowDataType::UInt32, |x| {
                    x.$extract().try_into().unwrap()
                })
            },
            ArrowDataType::Timestamp(time_unit, Some(timezone_str)) => {
                let array = $array.as_any().downcast_ref().unwrap();
                let timezone = parse_timezone(timezone_str.as_str())?;
                Ok(extract_impl(array, *time_unit, &timezone, |x| {
                    x.$extract().try_into().unwrap()
                }))
            },
            _ => unimplemented!(),
        }
    };
}

/// Extracts the hours of a temporal array as [`PrimitiveArray<i8>`].
/// Value ranges from 0 to 23.
/// Use [`can_hour`] to check if this operation is supported for the target [`ArrowDataType`].
pub fn hour(array: &dyn Array) -> PolarsResult<PrimitiveArray<i8>> {
    time_like!(hour, array, ArrowDataType::Int8)
}

/// Extracts the minutes of a temporal array as [`PrimitiveArray<i8>`].
/// Value ranges from 0 to 59.
/// Use [`can_minute`] to check if this operation is supported for the target [`ArrowDataType`].
pub fn minute(array: &dyn Array) -> PolarsResult<PrimitiveArray<i8>> {
    time_like!(minute, array, ArrowDataType::Int8)
}

/// Extracts the seconds of a temporal array as [`PrimitiveArray<i8>`].
/// Value ranges from 0 to 59.
/// Use [`can_second`] to check if this operation is supported for the target [`ArrowDataType`].
pub fn second(array: &dyn Array) -> PolarsResult<PrimitiveArray<i8>> {
    time_like!(second, array, ArrowDataType::Int8)
}

/// Extracts the nanoseconds of a temporal array as [`PrimitiveArray<i32>`].
///
/// Value ranges from 0 to 1_999_999_999.
/// The range from 1_000_000_000 to 1_999_999_999 represents the leap second.
/// Use [`can_nanosecond`] to check if this operation is supported for the target [`ArrowDataType`].
pub fn nanosecond(array: &dyn Array) -> PolarsResult<PrimitiveArray<i32>> {
    time_like!(subsec_nanosecond, array, ArrowDataType::Int32)
}

fn date_variants<F, O>(
    array: &dyn Array,
    dtype: ArrowDataType,
    op: F,
) -> PolarsResult<PrimitiveArray<O>>
where
    O: NativeType,
    F: Fn(DateTime) -> O,
{
    match array.dtype().to_storage() {
        ArrowDataType::Date32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .unwrap();
            Ok(unary(array, |x| op(date32_to_datetime(x)), dtype))
        },
        ArrowDataType::Date64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .unwrap();
            Ok(unary(array, |x| op(date64_to_datetime(x)), dtype))
        },
        ArrowDataType::Timestamp(time_unit, None) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .unwrap();
            let func: fn(i64) -> Option<DateTime> = match time_unit {
                TimeUnit::Second => timestamp_s_to_datetime_opt,
                TimeUnit::Millisecond => timestamp_ms_to_datetime_opt,
                TimeUnit::Microsecond => timestamp_us_to_datetime_opt,
                TimeUnit::Nanosecond => timestamp_ns_to_datetime_opt,
            };
            Ok(PrimitiveArray::<O>::from_trusted_len_iter(
                array.iter().map(|v| v.and_then(|x| func(*x).map(&op))),
            ))
        },
        _ => unreachable!(),
    }
}

fn time_variants<F, O>(
    array: &dyn Array,
    dtype: ArrowDataType,
    op: F,
) -> PolarsResult<PrimitiveArray<O>>
where
    O: NativeType,
    F: Fn(Time) -> O,
{
    match array.dtype().to_storage() {
        ArrowDataType::Time32(TimeUnit::Second) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .unwrap();
            Ok(unary(array, |x| op(time32s_to_time(x)), dtype))
        },
        ArrowDataType::Time32(TimeUnit::Millisecond) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .unwrap();
            Ok(unary(array, |x| op(time32ms_to_time(x)), dtype))
        },
        ArrowDataType::Time64(TimeUnit::Microsecond) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .unwrap();
            Ok(unary(array, |x| op(time64us_to_time(x)), dtype))
        },
        ArrowDataType::Time64(TimeUnit::Nanosecond) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .unwrap();
            Ok(unary(array, |x| op(time64ns_to_time(x)), dtype))
        },
        _ => unreachable!(),
    }
}

/// Parses a `timezone` string into a [`TimeZone`], accepting either a fixed
/// offset (e.g. `"+01:00"`, `"UTC"`) or, when the `chrono-tz` feature is
/// active, an IANA timezone name (e.g. `"Europe/Amsterdam"`).
fn parse_timezone(timezone_str: &str) -> PolarsResult<TimeZone> {
    match parse_offset(timezone_str) {
        Ok(tz) => Ok(tz),
        Err(_) => parse_offset_tz_checked(timezone_str),
    }
}

#[cfg(feature = "chrono-tz")]
fn parse_offset_tz_checked(timezone_str: &str) -> PolarsResult<TimeZone> {
    parse_offset_tz(timezone_str)
}

#[cfg(not(feature = "chrono-tz"))]
fn parse_offset_tz_checked(timezone_str: &str) -> PolarsResult<TimeZone> {
    panic!(
        "timezone \"{}\" cannot be parsed (feature chrono-tz is not active)",
        timezone_str
    )
}

fn extract_impl<A, F>(
    array: &PrimitiveArray<i64>,
    time_unit: TimeUnit,
    timezone: &TimeZone,
    extract: F,
) -> PrimitiveArray<A>
where
    A: NativeType,
    F: Fn(DateTime) -> A,
{
    let iter = array.iter().map(|opt| {
        opt.and_then(|&x| {
            timestamp_to_timestamp_opt(x, time_unit).map(|ts| extract(timezone.to_datetime(ts)))
        })
    });
    PrimitiveArray::from_trusted_len_iter(iter)
}
