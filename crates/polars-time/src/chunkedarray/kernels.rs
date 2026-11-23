//! macros that define kernels for extracting
//! `week`, `weekday`, `year`, `hour` etc. from primitive arrays.
use chrono::{Datelike, Timelike};
use polars_arrow::array::{BooleanArray, PrimitiveArray};
use polars_arrow::bitmap::BitmapBuilder;
use polars_arrow::compute::arity::unary;
#[cfg(feature = "dtype-time")]
use polars_arrow::temporal_conversions::time64ns_to_time_opt;
use polars_arrow::temporal_conversions::{
    EPOCH_DAYS_FROM_CE, date32_to_datetime_opt, timestamp_ms_to_datetime_opt,
    timestamp_ns_to_datetime_opt, timestamp_us_to_datetime_opt,
};
use polars_arrow::types::NativeType;
use polars_utils::time::{NS_DAY, is_leap_year};

use super::*;

trait PolarsIso {
    fn week(&self) -> i8;
    fn iso_year(&self) -> i32;
}

impl PolarsIso for NaiveDateTime {
    fn week(&self) -> i8 {
        self.iso_week().week().try_into().unwrap()
    }
    fn iso_year(&self) -> i32 {
        self.iso_week().year()
    }
}

impl PolarsIso for NaiveDate {
    fn week(&self) -> i8 {
        self.iso_week().week().try_into().unwrap()
    }
    fn iso_year(&self) -> i32 {
        self.iso_week().year()
    }
}

macro_rules! to_temporal_unit {
    ($name: ident, $chrono_method: ident, $to_datetime_fn: expr,
    $primitive_in: ty,
    $primitive_out: ty,
    $dtype_out:expr) => {
        pub(crate) fn $name(arr: &PrimitiveArray<$primitive_in>) -> ArrayRef {
            Box::new(PrimitiveArray::<$primitive_out>::from_trusted_len_iter(
                arr.iter().map(|opt_value| {
                    opt_value.and_then(|&value| {
                        $to_datetime_fn(value).map(|dt| dt.$chrono_method() as $primitive_out)
                    })
                }),
            )) as ArrayRef
        }
    };
}

// Dates
#[cfg(feature = "dtype-date")]
to_temporal_unit!(
    date_to_iso_week,
    week,
    date32_to_datetime_opt,
    i32,
    i8,
    ArrowDataType::Int8
);
#[cfg(feature = "dtype-date")]
to_temporal_unit!(
    date_to_iso_year,
    iso_year,
    date32_to_datetime_opt,
    i32,
    i32,
    ArrowDataType::Int32
);

// Times
#[cfg(feature = "dtype-time")]
to_temporal_unit!(
    time_to_hour,
    hour,
    time64ns_to_time_opt,
    i64,
    i8,
    ArrowDataType::Int8
);
#[cfg(feature = "dtype-time")]
to_temporal_unit!(
    time_to_minute,
    minute,
    time64ns_to_time_opt,
    i64,
    i8,
    ArrowDataType::Int8
);
#[cfg(feature = "dtype-time")]
to_temporal_unit!(
    time_to_second,
    second,
    time64ns_to_time_opt,
    i64,
    i8,
    ArrowDataType::Int8
);
#[cfg(feature = "dtype-time")]
to_temporal_unit!(
    time_to_nanosecond,
    nanosecond,
    time64ns_to_time_opt,
    i64,
    i32,
    ArrowDataType::Int32
);

#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_iso_year_ns,
    iso_year,
    timestamp_ns_to_datetime_opt,
    i64,
    i32,
    ArrowDataType::Int32
);

#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_iso_year_us,
    iso_year,
    timestamp_us_to_datetime_opt,
    i64,
    i32,
    ArrowDataType::Int32
);

#[cfg(feature = "dtype-datetime")]
to_temporal_unit!(
    datetime_to_iso_year_ms,
    iso_year,
    timestamp_ms_to_datetime_opt,
    i64,
    i32,
    ArrowDataType::Int32
);

#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
/// The days since the Unix epoch of the first and the last date that chrono supports. The date
/// parts of days outside of this range are null.
fn supported_days() -> (i64, i64) {
    let first = NaiveDate::MIN.num_days_from_ce() - EPOCH_DAYS_FROM_CE;
    let last = NaiveDate::MAX.num_days_from_ce() - EPOCH_DAYS_FROM_CE;
    (first as i64, last as i64)
}

#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
/// The year, the month (1-12) and the day of the month (1-31) in the proleptic Gregorian calendar
/// of `days` since the Unix epoch, for days in [`supported_days`].
#[inline(always)]
fn civil_from_days(days: i32) -> (i32, u32, u32) {
    // Count from 0000-03-01, so that the leap day is the last day of a year.
    let days = days + 719_468;
    let era = days.div_euclid(146_097);
    let day_of_era = (days - era * 146_097) as u32;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_from_march = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_from_march + 2) / 5 + 1;
    let month = if month_from_march < 10 {
        month_from_march + 3
    } else {
        month_from_march - 9
    };
    let year = year_of_era as i32 + era * 400 + (month <= 2) as i32;
    (year, month, day)
}

#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
#[inline(always)]
fn ordinal_from_days(days: i32) -> i16 {
    let (year, month, day) = civil_from_days(days);
    let february_short = 2 - is_leap_year(year) as u32;
    (275 * month / 9 - february_short * ((month + 9) / 12) + day - 30) as i16
}

#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
#[inline(always)]
fn days_in_month_from_days(days: i32) -> i8 {
    let (year, month, _) = civil_from_days(days);
    if month == 2 {
        28 + is_leap_year(year) as i8
    } else {
        30 + ((month ^ (month >> 3)) & 1) as i8
    }
}

#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
/// A temporal array as days since the Unix epoch: `to_days` maps its values to days and `bounds`
/// are the values whose days are in [`supported_days`].
struct DaysArray<'a, T: NativeType, F: Fn(T) -> i32> {
    arr: &'a PrimitiveArray<T>,
    to_days: F,
    bounds: (T, T),
}

#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
impl<T: NativeType + PartialOrd, F: Fn(T) -> i32> DaysArray<'_, T, F> {
    fn all_supported(&self) -> bool {
        let (first, last) = self.bounds;
        self.arr
            .values()
            .iter()
            .fold(true, |acc, &v| acc & (first <= v) & (v <= last))
    }

    fn map<O: NativeType>(&self, f: impl Fn(i32) -> O) -> ArrayRef {
        let out = if self.all_supported() {
            unary(self.arr, |v| f((self.to_days)(v)), O::PRIMITIVE.into())
        } else {
            let (first, last) = self.bounds;
            PrimitiveArray::from_trusted_len_iter(self.arr.iter().map(|v| {
                let v = *v?;
                (first <= v && v <= last).then(|| f((self.to_days)(v)))
            }))
        };
        Box::new(out)
    }

    fn map_bool(&self, f: impl Fn(i32) -> bool) -> ArrayRef {
        let out = if self.all_supported() {
            let values = self.arr.values();
            let mut bits = BitmapBuilder::with_capacity(values.len());
            // Fill a buffer first, so that `f` is vectorized.
            let mut buf = [false; 64];
            for chunk in values.chunks(64) {
                for (b, &v) in buf.iter_mut().zip(chunk) {
                    *b = f((self.to_days)(v));
                }
                let word = buf[..chunk.len()]
                    .iter()
                    .enumerate()
                    .fold(0u64, |word, (i, &b)| word | ((b as u64) << i));
                // SAFETY: the capacity is the number of values and a chunk has at most 64.
                unsafe { bits.push_word_with_len_unchecked(word, chunk.len()) };
            }
            BooleanArray::new(
                ArrowDataType::Boolean,
                bits.freeze(),
                self.arr.validity().cloned(),
            )
        } else {
            let (first, last) = self.bounds;
            BooleanArray::from_trusted_len_iter(self.arr.iter().map(|v| {
                let v = *v?;
                (first <= v && v <= last).then(|| f((self.to_days)(v)))
            }))
        };
        Box::new(out)
    }
}

#[cfg(feature = "dtype-date")]
fn date_days(arr: &PrimitiveArray<i32>) -> DaysArray<'_, i32, impl Fn(i32) -> i32> {
    let (first, last) = supported_days();
    DaysArray {
        arr,
        to_days: |v| v,
        bounds: (first as i32, last as i32),
    }
}

#[cfg(feature = "dtype-datetime")]
fn datetime_days(
    arr: &PrimitiveArray<i64>,
    time_unit: TimeUnit,
) -> DaysArray<'_, i64, impl Fn(i64) -> i32> {
    let per_day = match time_unit {
        TimeUnit::Nanoseconds => NS_DAY,
        TimeUnit::Microseconds => NS_DAY / 1_000,
        TimeUnit::Milliseconds => NS_DAY / 1_000_000,
    };
    let (first, last) = supported_days();
    let bounds = (
        first.checked_mul(per_day).unwrap_or(i64::MIN),
        (last + 1).checked_mul(per_day).map_or(i64::MAX, |v| v - 1),
    );
    DaysArray {
        arr,
        to_days: move |v: i64| v.div_euclid(per_day) as i32,
        bounds,
    }
}

macro_rules! date_parts {
    ($date_fn:ident, $datetime_fn:ident, $method:ident, $f:expr) => {
        #[cfg(feature = "dtype-date")]
        pub(crate) fn $date_fn(arr: &PrimitiveArray<i32>) -> ArrayRef {
            date_days(arr).$method($f)
        }

        #[cfg(feature = "dtype-datetime")]
        pub(crate) fn $datetime_fn(arr: &PrimitiveArray<i64>, time_unit: TimeUnit) -> ArrayRef {
            datetime_days(arr, time_unit).$method($f)
        }
    };
}

date_parts!(date_to_year, datetime_to_year, map, |d| civil_from_days(d)
    .0);
date_parts!(
    date_to_month,
    datetime_to_month,
    map,
    |d| civil_from_days(d).1 as i8
);
date_parts!(date_to_day, datetime_to_day, map, |d| civil_from_days(d).2
    as i8);
date_parts!(date_to_ordinal, datetime_to_ordinal, map, ordinal_from_days);
date_parts!(
    date_to_is_leap_year,
    datetime_to_is_leap_year,
    map_bool,
    |d| { is_leap_year(civil_from_days(d).0) }
);
date_parts!(
    date_to_days_in_month,
    datetime_to_days_in_month,
    map,
    days_in_month_from_days
);

#[cfg(test)]
#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
mod test {
    use polars_arrow::temporal_conversions::date32_to_date_opt;

    use super::*;

    #[test]
    fn test_civil_from_days_matches_chrono() {
        let (first, last) = supported_days();
        let near_epoch = -800_000..=800_000;
        let whole_range = (first..=last).step_by(997).chain([first, last]);
        for days in near_epoch.chain(whole_range) {
            let days = days as i32;
            let date = date32_to_date_opt(days).unwrap();
            assert_eq!(
                civil_from_days(days),
                (date.year(), date.month(), date.day()),
                "{days}"
            );
            assert_eq!(ordinal_from_days(days), date.ordinal() as i16, "{days}");
            assert_eq!(
                days_in_month_from_days(days),
                polars_utils::time::days_in_month(date.year(), date.month() as u8) as i8,
                "{days}"
            );
            assert_eq!(is_leap_year(date.year()), date.leap_year(), "{days}");
        }
        assert!(date32_to_date_opt(first as i32 - 1).is_none());
        assert!(date32_to_date_opt(last as i32 + 1).is_none());
    }
}
