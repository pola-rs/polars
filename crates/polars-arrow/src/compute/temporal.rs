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

use chrono::{Datelike, Timelike};
use polars_error::PolarsResult;

use super::arity::unary;
use crate::array::*;
use crate::datatypes::*;
use crate::temporal_conversions::*;
use crate::types::NativeType;

// Create and implement a trait that converts chrono's `Weekday`
// type into `i8`
trait Int8Weekday: Datelike {
    fn i8_weekday(&self) -> i8 {
        self.weekday().number_from_monday().try_into().unwrap()
    }
}

impl Int8Weekday for chrono::NaiveDateTime {}
impl<T: chrono::TimeZone> Int8Weekday for chrono::DateTime<T> {}

// Create and implement a trait that converts chrono's `IsoWeek`
// type into `i8`
trait Int8IsoWeek: Datelike {
    fn i8_iso_week(&self) -> i8 {
        self.iso_week().week().try_into().unwrap()
    }
}

impl Int8IsoWeek for chrono::NaiveDateTime {}
impl<T: chrono::TimeZone> Int8IsoWeek for chrono::DateTime<T> {}

// Macro to avoid repetition in functions, that apply
// `chrono::Datelike` methods on Arrays
macro_rules! date_like {
    ($extract:ident, $array:ident, $data_type:path) => {
        match $array.data_type().to_logical_type() {
            ArrowDataType::Date32 | ArrowDataType::Date64 | ArrowDataType::Timestamp(_, None) => {
                date_variants($array, $data_type, |x| x.$extract().try_into().unwrap())
            },
            ArrowDataType::Timestamp(time_unit, Some(timezone_str)) => {
                let array = $array.as_any().downcast_ref().unwrap();

                if let Ok(timezone) = parse_offset(timezone_str) {
                    Ok(extract_impl(array, *time_unit, timezone, |x| {
                        x.$extract().try_into().unwrap()
                    }))
                } else {
                    chrono_tz(array, *time_unit, timezone_str, |x| {
                        x.$extract().try_into().unwrap()
                    })
                }
            },
            _ => unimplemented!(),
        }
    };
}

/// Extracts the years of a temporal array as [`PrimitiveArray<i32>`].
///
/// Use [`can_year`] to check if this operation is supported for the target [`ArrowDataType`].
pub fn year(array: &dyn Array) -> PolarsResult<PrimitiveArray<i32>> {
    date_like!(year, array, ArrowDataType::Int32)
}

/// Extracts the months of a temporal array as [`PrimitiveArray<i8>`].
///
/// Value ranges from 1 to 12.
/// Use [`can_month`] to check if this operation is supported for the target [`ArrowDataType`].
pub fn month(array: &dyn Array) -> PolarsResult<PrimitiveArray<i8>> {
    date_like!(month, array, ArrowDataType::Int8)
}

/// Extracts the days of a temporal array as [`PrimitiveArray<i8>`].
///
/// Value ranges from 1 to 32 (Last day depends on month).
/// Use [`can_day`] to check if this operation is supported for the target [`ArrowDataType`].
pub fn day(array: &dyn Array) -> PolarsResult<PrimitiveArray<i8>> {
    date_like!(day, array, ArrowDataType::Int8)
}

/// Extracts weekday of a temporal array as [`PrimitiveArray<i8>`].
///
/// Monday is 1, Tuesday is 2, ..., Sunday is 7.
/// Use [`can_weekday`] to check if this operation is supported for the target [`ArrowDataType`]
pub fn weekday(array: &dyn Array) -> PolarsResult<PrimitiveArray<i8>> {
    date_like!(i8_weekday, array, ArrowDataType::Int8)
}

/// Extracts ISO week of a temporal array as [`PrimitiveArray<i8>`].
///
/// Value ranges from 1 to 53 (Last week depends on the year).
/// Use [`can_iso_week`] to check if this operation is supported for the target [`ArrowDataType`]
pub fn iso_week(array: &dyn Array) -> PolarsResult<PrimitiveArray<i8>> {
    date_like!(i8_iso_week, array, ArrowDataType::Int8)
}

// Macro to avoid repetition in functions, that apply
// `chrono::Timelike` methods on Arrays
macro_rules! time_like {
    ($extract:ident, $array:ident, $data_type:path) => {
        match $array.data_type().to_logical_type() {
            ArrowDataType::Date32 | ArrowDataType::Date64 | ArrowDataType::Timestamp(_, None) => {
                date_variants($array, $data_type, |x| x.$extract().try_into().unwrap())
            },
            ArrowDataType::Time32(_) | ArrowDataType::Time64(_) => {
                time_variants($array, ArrowDataType::UInt32, |x| {
                    x.$extract().try_into().unwrap()
                })
            },
            ArrowDataType::Timestamp(time_unit, Some(timezone_str)) => {
                let array = $array.as_any().downcast_ref().unwrap();

                if let Ok(timezone) = parse_offset(timezone_str) {
                    Ok(extract_impl(array, *time_unit, timezone, |x| {
                        x.$extract().try_into().unwrap()
                    }))
                } else {
                    chrono_tz(array, *time_unit, timezone_str, |x| {
                        x.$extract().try_into().unwrap()
                    })
                }
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
    time_like!(nanosecond, array, ArrowDataType::Int32)
}

fn date_variants<F, O>(
    array: &dyn Array,
    data_type: ArrowDataType,
    op: F,
) -> PolarsResult<PrimitiveArray<O>>
where
    O: NativeType,
    F: Fn(chrono::NaiveDateTime) -> O,
{
    match array.data_type().to_logical_type() {
        ArrowDataType::Date32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .unwrap();
            Ok(unary(array, |x| op(date32_to_datetime(x)), data_type))
        },
        ArrowDataType::Date64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .unwrap();
            Ok(unary(array, |x| op(date64_to_datetime(x)), data_type))
        },
        ArrowDataType::Timestamp(time_unit, None) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .unwrap();
            let func = match time_unit {
                TimeUnit::Second => timestamp_s_to_datetime,
                TimeUnit::Millisecond => timestamp_ms_to_datetime,
                TimeUnit::Microsecond => timestamp_us_to_datetime,
                TimeUnit::Nanosecond => timestamp_ns_to_datetime,
            };
            Ok(PrimitiveArray::<O>::from_trusted_len_iter(
                array.iter().map(|v| v.map(|x| op(func(*x)))),
            ))
        },
        _ => unreachable!(),
    }
}

fn time_variants<F, O>(
    array: &dyn Array,
    data_type: ArrowDataType,
    op: F,
) -> PolarsResult<PrimitiveArray<O>>
where
    O: NativeType,
    F: Fn(chrono::NaiveTime) -> O,
{
    match array.data_type().to_logical_type() {
        ArrowDataType::Time32(TimeUnit::Second) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .unwrap();
            Ok(unary(array, |x| op(time32s_to_time(x)), data_type))
        },
        ArrowDataType::Time32(TimeUnit::Millisecond) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .unwrap();
            Ok(unary(array, |x| op(time32ms_to_time(x)), data_type))
        },
        ArrowDataType::Time64(TimeUnit::Microsecond) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .unwrap();
            Ok(unary(array, |x| op(time64us_to_time(x)), data_type))
        },
        ArrowDataType::Time64(TimeUnit::Nanosecond) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .unwrap();
            Ok(unary(array, |x| op(time64ns_to_time(x)), data_type))
        },
        _ => unreachable!(),
    }
}

#[cfg(feature = "chrono-tz")]
fn chrono_tz<F, O>(
    array: &PrimitiveArray<i64>,
    time_unit: TimeUnit,
    timezone_str: &str,
    op: F,
) -> PolarsResult<PrimitiveArray<O>>
where
    O: NativeType,
    F: Fn(chrono::DateTime<chrono_tz::Tz>) -> O,
{
    let timezone = parse_offset_tz(timezone_str)?;
    Ok(extract_impl(array, time_unit, timezone, op))
}

#[cfg(not(feature = "chrono-tz"))]
fn chrono_tz<F, O>(
    _: &PrimitiveArray<i64>,
    _: TimeUnit,
    timezone_str: &str,
    _: F,
) -> PolarsResult<PrimitiveArray<O>>
where
    O: NativeType,
    F: Fn(chrono::DateTime<chrono::FixedOffset>) -> O,
{
    panic!(
        "timezone \"{}\" cannot be parsed (feature chrono-tz is not active)",
        timezone_str
    )
}

fn extract_impl<T, A, F>(
    array: &PrimitiveArray<i64>,
    time_unit: TimeUnit,
    timezone: T,
    extract: F,
) -> PrimitiveArray<A>
where
    T: chrono::TimeZone,
    A: NativeType,
    F: Fn(chrono::DateTime<T>) -> A,
{
    match time_unit {
        TimeUnit::Second => {
            let op = |x| {
                let datetime = timestamp_s_to_datetime(x);
                let offset = timezone.offset_from_utc_datetime(&datetime);
                extract(chrono::DateTime::<T>::from_naive_utc_and_offset(
                    datetime, offset,
                ))
            };
            unary(array, op, A::PRIMITIVE.into())
        },
        TimeUnit::Millisecond => {
            let op = |x| {
                let datetime = timestamp_ms_to_datetime(x);
                let offset = timezone.offset_from_utc_datetime(&datetime);
                extract(chrono::DateTime::<T>::from_naive_utc_and_offset(
                    datetime, offset,
                ))
            };
            unary(array, op, A::PRIMITIVE.into())
        },
        TimeUnit::Microsecond => {
            let op = |x| {
                let datetime = timestamp_us_to_datetime(x);
                let offset = timezone.offset_from_utc_datetime(&datetime);
                extract(chrono::DateTime::<T>::from_naive_utc_and_offset(
                    datetime, offset,
                ))
            };
            unary(array, op, A::PRIMITIVE.into())
        },
        TimeUnit::Nanosecond => {
            let op = |x| {
                let datetime = timestamp_ns_to_datetime(x);
                let offset = timezone.offset_from_utc_datetime(&datetime);
                extract(chrono::DateTime::<T>::from_naive_utc_and_offset(
                    datetime, offset,
                ))
            };
            unary(array, op, A::PRIMITIVE.into())
        },
    }
}

/// Checks if an array of type `datatype` can perform year operation
///
/// # Examples
/// ```
/// use polars_arrow::compute::temporal::can_year;
/// use polars_arrow::datatypes::{ArrowDataType};
///
/// assert_eq!(can_year(&ArrowDataType::Date32), true);
/// assert_eq!(can_year(&ArrowDataType::Int8), false);
/// ```
pub fn can_year(data_type: &ArrowDataType) -> bool {
    can_date(data_type)
}

/// Checks if an array of type `datatype` can perform month operation
pub fn can_month(data_type: &ArrowDataType) -> bool {
    can_date(data_type)
}

/// Checks if an array of type `datatype` can perform day operation
pub fn can_day(data_type: &ArrowDataType) -> bool {
    can_date(data_type)
}

/// Checks if an array of type `data_type` can perform weekday operation
pub fn can_weekday(data_type: &ArrowDataType) -> bool {
    can_date(data_type)
}

/// Checks if an array of type `data_type` can perform ISO week operation
pub fn can_iso_week(data_type: &ArrowDataType) -> bool {
    can_date(data_type)
}

fn can_date(data_type: &ArrowDataType) -> bool {
    matches!(
        data_type,
        ArrowDataType::Date32 | ArrowDataType::Date64 | ArrowDataType::Timestamp(_, _)
    )
}

/// Checks if an array of type `datatype` can perform hour operation
///
/// # Examples
/// ```
/// use polars_arrow::compute::temporal::can_hour;
/// use polars_arrow::datatypes::{ArrowDataType, TimeUnit};
///
/// assert_eq!(can_hour(&ArrowDataType::Time32(TimeUnit::Second)), true);
/// assert_eq!(can_hour(&ArrowDataType::Int8), false);
/// ```
pub fn can_hour(data_type: &ArrowDataType) -> bool {
    can_time(data_type)
}

/// Checks if an array of type `datatype` can perform minute operation
pub fn can_minute(data_type: &ArrowDataType) -> bool {
    can_time(data_type)
}

/// Checks if an array of type `datatype` can perform second operation
pub fn can_second(data_type: &ArrowDataType) -> bool {
    can_time(data_type)
}

/// Checks if an array of type `datatype` can perform nanosecond operation
pub fn can_nanosecond(data_type: &ArrowDataType) -> bool {
    can_time(data_type)
}

fn can_time(data_type: &ArrowDataType) -> bool {
    matches!(
        data_type,
        ArrowDataType::Time32(TimeUnit::Second)
            | ArrowDataType::Time32(TimeUnit::Millisecond)
            | ArrowDataType::Time64(TimeUnit::Microsecond)
            | ArrowDataType::Time64(TimeUnit::Nanosecond)
            | ArrowDataType::Date32
            | ArrowDataType::Date64
            | ArrowDataType::Timestamp(_, _)
    )
}
