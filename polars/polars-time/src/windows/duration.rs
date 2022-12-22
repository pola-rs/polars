use std::cmp::Ordering;
use std::ops::Mul;

use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use polars_arrow::export::arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime, MILLISECONDS,
};
use polars_core::export::arrow::temporal_conversions::MICROSECONDS;
use polars_core::prelude::{
    datetime_to_timestamp_ms, datetime_to_timestamp_ns, datetime_to_timestamp_us,
};
use polars_core::utils::arrow::temporal_conversions::NANOSECONDS;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::calendar::{
    is_leap_year, last_day_of_month, NS_DAY, NS_HOUR, NS_MICROSECOND, NS_MILLISECOND, NS_MINUTE,
    NS_SECOND, NS_WEEK,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Duration {
    // the number of months for the duration
    months: i64,
    // the number of nanoseconds for the duration
    nsecs: i64,
    // indicates if the duration is negative
    pub(crate) negative: bool,
    // indicates if an integer string was passed. e.g. "2i"
    pub parsed_int: bool,
}

impl PartialOrd<Self> for Duration {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.duration_ns().partial_cmp(&other.duration_ns())
    }
}

impl Ord for Duration {
    fn cmp(&self, other: &Self) -> Ordering {
        self.duration_ns().cmp(&other.duration_ns())
    }
}

impl Duration {
    /// Create a new integer size `Duration`
    pub fn new(fixed_slots: i64) -> Self {
        Duration {
            months: 0,
            nsecs: fixed_slots.abs(),
            negative: fixed_slots < 0,
            parsed_int: true,
        }
    }

    /// 1ns // 1 nanosecond
    /// 1us // 1 microsecond
    /// 1ms // 1 millisecond
    /// 1s  // 1 second
    /// 1m  // 1 minute
    /// 1h  // 1 hour
    /// 1d  // 1 day
    /// 1w  // 1 week
    /// 1mo // 1 calendar month
    /// 1y  // 1 calendar year
    /// 1i  // 1 index value (only for {Int32, Int64} dtypes
    ///
    /// 3d12h4m25s // 3 days, 12 hours, 4 minutes, and 25 seconds
    ///
    /// # Panics if given str is incorrect
    pub fn parse(duration: &str) -> Self {
        let num_minus_signs = duration.matches('-').count();
        if num_minus_signs > 1 {
            panic!("a Duration string can only have a single minus sign")
        }
        if (num_minus_signs > 0) & !duration.starts_with('-') {
            panic!("only a single minus sign is allowed, at the front of the string")
        }

        let mut nsecs = 0;
        let mut months = 0;
        let mut iter = duration.char_indices();
        let negative = duration.starts_with('-');
        let mut start = 0;

        // skip the '-' char
        if negative {
            start += 1;
            iter.next().unwrap();
        }

        let mut parsed_int = false;

        let mut unit = String::with_capacity(2);
        while let Some((i, mut ch)) = iter.next() {
            if !ch.is_ascii_digit() {
                let n = duration[start..i].parse::<i64>().unwrap();

                loop {
                    if ch.is_ascii_alphabetic() {
                        unit.push(ch)
                    } else {
                        break;
                    }
                    match iter.next() {
                        Some((i, ch_)) => {
                            ch = ch_;
                            start = i
                        }
                        None => {
                            break;
                        }
                    }
                }
                if unit.is_empty() {
                    panic!("expected a unit in the duration string")
                }

                match &*unit {
                    "ns" => nsecs += n,
                    "us" => nsecs += n * NS_MICROSECOND,
                    "ms" => nsecs += n * NS_MILLISECOND,
                    "s" => nsecs += n * NS_SECOND,
                    "m" => nsecs += n * NS_MINUTE,
                    "h" => nsecs += n * NS_HOUR,
                    "d" => nsecs += n * NS_DAY,
                    "w" => nsecs += n * NS_WEEK,
                    "mo" => months += n,
                    "y" => months += n * 12,
                    // we will read indexes as nanoseconds
                    "i" => {
                        nsecs += n;
                        parsed_int = true;
                    }
                    unit => panic!("unit: '{unit}' not supported"),
                }
                unit.clear();
            }
        }
        Duration {
            nsecs: nsecs.abs(),
            months: months.abs(),
            negative,
            parsed_int,
        }
    }

    fn to_positive(v: i64) -> (bool, i64) {
        if v < 0 {
            (true, -v)
        } else {
            (false, v)
        }
    }

    /// Normalize the duration within the interval.
    /// It will ensure that the output duration is the smallest positive
    /// duration that is the equivalent of the current duration.
    #[allow(dead_code)]
    pub(crate) fn normalize(&self, interval: &Duration) -> Self {
        if self.months_only() && interval.months_only() {
            let mut months = self.months() % interval.months();

            match (self.negative, interval.negative) {
                (true, true) | (true, false) => months = -months + interval.months(),
                _ => {}
            }
            Duration::from_months(months)
        } else {
            let mut offset = self.duration_ns();
            if offset == 0 {
                return *self;
            }
            let every = interval.duration_ns();

            if offset < 0 {
                offset += every * ((offset / -every) + 1)
            } else {
                offset -= every * (offset / every)
            }
            Duration::from_nsecs(offset)
        }
    }

    /// Creates a [`Duration`] that represents a fixed number of nanoseconds.
    pub(crate) fn from_nsecs(v: i64) -> Self {
        let (negative, nsecs) = Self::to_positive(v);
        Self {
            months: 0,
            nsecs,
            negative,
            parsed_int: false,
        }
    }

    /// Creates a [`Duration`] that represents a fixed number of months.
    pub(crate) fn from_months(v: i64) -> Self {
        let (negative, months) = Self::to_positive(v);
        Self {
            months,
            nsecs: 0,
            negative,
            parsed_int: false,
        }
    }

    /// `true` if zero duration.
    pub fn is_zero(&self) -> bool {
        self.months == 0 && self.nsecs == 0
    }

    pub fn months_only(&self) -> bool {
        self.months != 0 && self.nsecs == 0
    }

    pub fn months(&self) -> i64 {
        self.months
    }

    pub fn nanoseconds(&self) -> i64 {
        self.nsecs
    }

    /// Estimated duration of the window duration. Not a very good one if months != 0.
    #[cfg(feature = "private")]
    #[doc(hidden)]
    pub const fn duration_ns(&self) -> i64 {
        self.months * 28 * 24 * 3600 * NANOSECONDS + self.nsecs
    }

    #[cfg(feature = "private")]
    #[doc(hidden)]
    pub const fn duration_us(&self) -> i64 {
        self.months * 28 * 24 * 3600 * MICROSECONDS + self.nsecs / 1000
    }

    #[cfg(feature = "private")]
    #[doc(hidden)]
    pub const fn duration_ms(&self) -> i64 {
        self.months * 28 * 24 * 3600 * MILLISECONDS + self.nsecs / 1_000_000
    }

    #[inline]
    pub fn truncate_impl<F, G, J>(
        &self,
        t: i64,
        nsecs_to_unit: F,
        timestamp_to_datetime: G,
        datetime_to_timestamp: J,
    ) -> i64
    where
        F: Fn(i64) -> i64,
        G: Fn(i64) -> NaiveDateTime,
        J: Fn(NaiveDateTime) -> i64,
    {
        match (self.months, self.nsecs) {
            (0, 0) => panic!("duration may not be zero"),
            // truncate by ns/us/ms
            (0, _) => {
                let duration = nsecs_to_unit(self.nsecs);
                let mut remainder = t % duration;
                if remainder < 0 {
                    remainder += duration
                }
                t - remainder
            }
            // truncate by months
            (_, 0) => {
                let ts = timestamp_to_datetime(t);
                let (year, month) = (ts.year(), ts.month());

                // determine the total number of months and truncate
                // the number of months by the duration amount
                let mut total = (year * 12) + (month as i32 - 1);
                let remainder = total % self.months as i32;
                total -= remainder;

                // recreate a new time from the year and month combination
                let (year, month) = ((total / 12), ((total % 12) + 1) as u32);
                let dt = new_datetime(year, month, 1, 0, 0, 0, 0);
                datetime_to_timestamp(dt)
            }
            _ => panic!("duration may not mix month and nanosecond units"),
        }
    }

    // Truncate the given ns timestamp by the window boundary.
    #[inline]
    pub fn truncate_ns(&self, t: i64) -> i64 {
        self.truncate_impl(
            t,
            |nsecs| nsecs,
            timestamp_ns_to_datetime,
            datetime_to_timestamp_ns,
        )
    }

    // Truncate the given ns timestamp by the window boundary.
    #[inline]
    pub fn truncate_us(&self, t: i64) -> i64 {
        self.truncate_impl(
            t,
            |nsecs| nsecs / 1000,
            timestamp_us_to_datetime,
            datetime_to_timestamp_us,
        )
    }

    // Truncate the given ms timestamp by the window boundary.
    #[inline]
    pub fn truncate_ms(&self, t: i64) -> i64 {
        self.truncate_impl(
            t,
            |nsecs| nsecs / 1_000_000,
            timestamp_ms_to_datetime,
            datetime_to_timestamp_ms,
        )
    }

    fn add_impl_month<F, G>(
        &self,
        t: i64,
        timestamp_to_datetime: F,
        datetime_to_timestamp: G,
    ) -> i64
    where
        F: Fn(i64) -> NaiveDateTime,
        G: Fn(NaiveDateTime) -> i64,
    {
        let d = self;
        let mut new_t = t;

        if d.months > 0 {
            let mut months = d.months;
            if d.negative {
                months = -months;
            }

            // Retrieve the current date and increment the values
            // based on the number of months
            let ts = timestamp_to_datetime(t);
            let mut year = ts.year();
            let mut month = ts.month() as i32;
            let mut day = ts.day();
            year += (months / 12) as i32;
            month += (months % 12) as i32;

            // if the month overflowed or underflowed, adjust the year
            // accordingly. Because we add the modulo for the months
            // the year will only adjust by one
            if month > 12 {
                year += 1;
                month -= 12;
            } else if month <= 0 {
                year -= 1;
                month += 12;
            }

            // Normalize the day if we are past the end of the month.
            let mut last_day_of_month = last_day_of_month(month);
            if month == (chrono::Month::February.number_from_month() as i32) && is_leap_year(year) {
                last_day_of_month += 1;
            }

            if day > last_day_of_month {
                day = last_day_of_month
            }

            // Retrieve the original time and construct a data
            // with the new year, month and day
            let hour = ts.hour();
            let minute = ts.minute();
            let sec = ts.second();
            let nsec = ts.nanosecond();
            let dt = new_datetime(year, month as u32, day, hour, minute, sec, nsec);
            new_t = datetime_to_timestamp(dt);
        }
        new_t
    }

    pub fn add_ns(&self, t: i64) -> i64 {
        let d = self;
        let new_t = self.add_impl_month(t, timestamp_ns_to_datetime, datetime_to_timestamp_ns);
        let nsecs = if d.negative { -d.nsecs } else { d.nsecs };
        new_t + nsecs
    }

    pub fn add_us(&self, t: i64) -> i64 {
        let d = self;
        let new_t = self.add_impl_month(t, timestamp_us_to_datetime, datetime_to_timestamp_us);
        let nsecs = if d.negative { -d.nsecs } else { d.nsecs };
        new_t + nsecs / 1_000
    }

    pub fn add_ms(&self, t: i64) -> i64 {
        let d = self;
        let new_t = self.add_impl_month(t, timestamp_ms_to_datetime, datetime_to_timestamp_ms);
        let nsecs = if d.negative { -d.nsecs } else { d.nsecs };
        new_t + nsecs / 1_000_000
    }
}

impl Mul<i64> for Duration {
    type Output = Self;

    fn mul(mut self, mut rhs: i64) -> Self {
        if rhs < 0 {
            rhs = -rhs;
            self.negative = !self.negative
        }
        self.months *= rhs;
        self.nsecs *= rhs;
        self
    }
}

fn new_datetime(
    year: i32,
    month: u32,
    days: u32,
    hour: u32,
    min: u32,
    sec: u32,
    nano: u32,
) -> NaiveDateTime {
    let date = NaiveDate::from_ymd_opt(year, month, days).unwrap();
    let time = NaiveTime::from_hms_nano_opt(hour, min, sec, nano).unwrap();

    NaiveDateTime::new(date, time)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parse() {
        let out = Duration::parse("1ns");
        assert_eq!(out.nsecs, 1);
        let out = Duration::parse("1ns1ms");
        assert_eq!(out.nsecs, NS_MILLISECOND + 1);
        let out = Duration::parse("123ns40ms");
        assert_eq!(out.nsecs, 40 * NS_MILLISECOND + 123);
        let out = Duration::parse("123ns40ms1w");
        assert_eq!(out.nsecs, 40 * NS_MILLISECOND + 123 + NS_WEEK);
        let out = Duration::parse("-123ns40ms1w");
        assert!(out.negative);
    }
}
