use std::cmp::Ordering;
use std::ops::Mul;

use arrow::legacy::kernels::Ambiguous;
use arrow::legacy::time_zone::Tz;
use arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime, MILLISECONDS,
};
use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike, Weekday};
use polars_core::export::arrow::temporal_conversions::MICROSECONDS;
use polars_core::prelude::{
    datetime_to_timestamp_ms, datetime_to_timestamp_ns, datetime_to_timestamp_us, polars_bail,
    PolarsResult,
};
use polars_core::utils::arrow::temporal_conversions::NANOSECONDS;
use polars_error::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::calendar::{
    NS_DAY, NS_HOUR, NS_MICROSECOND, NS_MILLISECOND, NS_MINUTE, NS_SECOND, NS_WEEK,
};
#[cfg(feature = "timezones")]
use crate::utils::{localize_datetime, unlocalize_datetime};
use crate::windows::calendar::{is_leap_year, last_day_of_month};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Duration {
    // the number of months for the duration
    months: i64,
    // the number of weeks for the duration
    weeks: i64,
    // the number of nanoseconds for the duration
    days: i64,
    // the number of nanoseconds for the duration
    nsecs: i64,
    // indicates if the duration is negative
    pub(crate) negative: bool,
    // indicates if an integer string was passed. e.g. "2i"
    pub parsed_int: bool,
    // indicates if an offset to a non-existent date (e.g. 2022-02-29)
    // should saturate (to 2022-02-28) as opposed to erroring
    pub(crate) saturating: bool,
}

impl PartialOrd<Self> for Duration {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
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
            weeks: 0,
            days: 0,
            nsecs: fixed_slots.abs(),
            negative: fixed_slots < 0,
            parsed_int: true,
            saturating: false,
        }
    }

    /// Parse a string into a `Duration`
    ///
    /// Strings are composed of a sequence of number-unit pairs, such as `5d` (5 days). A string may begin with a minus
    /// sign, in which case it is interpreted as a negative duration. Some examples:
    ///
    /// * `"1y"`: 1 year
    /// * `"-1w2d"`: negative 1 week, 2 days (i.e. -9 days)
    /// * `"3d12h4m25s"`: 3 days, 12 hours, 4 minutes, and 25 seconds
    ///
    /// Aside from a leading minus sign, strings may not contain any characters other than numbers and letters
    /// (including whitespace).
    ///
    /// The available units, in ascending order of magnitude, are as follows:
    ///
    /// * `ns`: nanosecond
    /// * `us`: microsecond
    /// * `ms`: millisecond
    /// * `s`:  second
    /// * `m`:  minute
    /// * `h`:  hour
    /// * `d`:  day
    /// * `w`:  week
    /// * `mo`: calendar month
    /// * `q`: calendar quarter
    /// * `y`:  calendar year
    /// * `i`:  index value (only for {Int32, Int64} dtypes)
    ///
    /// Suffix with `"_saturating"` to indicate that dates too large for
    /// their month should saturate at the largest date (e.g. 2022-02-29 -> 2022-02-28)
    /// instead of erroring.
    ///
    /// By "calendar day", we mean the corresponding time on the next
    /// day (which may not be 24 hours, depending on daylight savings).
    /// Similarly for "calendar week", "calendar month", "calendar quarter",
    /// and "calendar year".
    ///
    /// # Panics
    /// If the given str is invalid for any reason.
    pub fn parse(duration: &str) -> Self {
        let num_minus_signs = duration.matches('-').count();
        if num_minus_signs > 1 {
            panic!("a Duration string can only have a single minus sign")
        }
        if (num_minus_signs > 0) & !duration.starts_with('-') {
            panic!("only a single minus sign is allowed, at the front of the string")
        }

        let mut nsecs = 0;
        let mut weeks = 0;
        let mut days = 0;
        let mut months = 0;
        let negative = duration.starts_with('-');
        let (saturating, mut iter) = match duration.ends_with("_saturating") {
            true => (
                true,
                duration[..duration.len() - "_saturating".len()].char_indices(),
            ),
            false => (false, duration.char_indices()),
        };
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
                let n = duration[start..i]
                    .parse::<i64>()
                    .expect("expected an integer in the duration string");

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
                        },
                        None => {
                            break;
                        },
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
                    "d" => days += n,
                    "w" => weeks += n,
                    "mo" => {
                        months += n
                    }
                    "q" => months += n * 3,
                    "y" => months += n * 12,
                    // we will read indexes as nanoseconds
                    "i" => {
                        nsecs += n;
                        parsed_int = true;
                    }
                    unit => panic!("unit: '{unit}' not supported. Available units are: 'ns', 'us', 'ms', 's', 'm', 'h', 'd', 'w', 'q', 'mo', 'y', 'i'"),
                }
                unit.clear();
            }
        }
        Duration {
            nsecs: nsecs.abs(),
            days: days.abs(),
            weeks: weeks.abs(),
            months: months.abs(),
            negative,
            parsed_int,
            saturating,
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
                _ => {},
            }
            Duration::from_months(months)
        } else if self.weeks_only() && interval.weeks_only() {
            let mut weeks = self.weeks() % interval.weeks();

            match (self.negative, interval.negative) {
                (true, true) | (true, false) => weeks = -weeks + interval.weeks(),
                _ => {},
            }
            Duration::from_weeks(weeks)
        } else if self.days_only() && interval.days_only() {
            let mut days = self.days() % interval.days();

            match (self.negative, interval.negative) {
                (true, true) | (true, false) => days = -days + interval.days(),
                _ => {},
            }
            Duration::from_days(days)
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
            weeks: 0,
            days: 0,
            nsecs,
            negative,
            parsed_int: false,
            saturating: false,
        }
    }

    /// Creates a [`Duration`] that represents a fixed number of months.
    pub(crate) fn from_months(v: i64) -> Self {
        let (negative, months) = Self::to_positive(v);
        Self {
            months,
            weeks: 0,
            days: 0,
            nsecs: 0,
            negative,
            parsed_int: false,
            saturating: false,
        }
    }

    /// Creates a [`Duration`] that represents a fixed number of weeks.
    pub(crate) fn from_weeks(v: i64) -> Self {
        let (negative, weeks) = Self::to_positive(v);
        Self {
            months: 0,
            weeks,
            days: 0,
            nsecs: 0,
            negative,
            parsed_int: false,
            saturating: false,
        }
    }

    /// Creates a [`Duration`] that represents a fixed number of days.
    pub(crate) fn from_days(v: i64) -> Self {
        let (negative, days) = Self::to_positive(v);
        Self {
            months: 0,
            weeks: 0,
            days,
            nsecs: 0,
            negative,
            parsed_int: false,
            saturating: false,
        }
    }

    /// `true` if zero duration.
    pub fn is_zero(&self) -> bool {
        self.months == 0 && self.weeks == 0 && self.days == 0 && self.nsecs == 0
    }

    pub fn months_only(&self) -> bool {
        self.months != 0 && self.weeks == 0 && self.days == 0 && self.nsecs == 0
    }

    pub fn months(&self) -> i64 {
        self.months
    }

    pub fn weeks_only(&self) -> bool {
        self.months == 0 && self.weeks != 0 && self.days == 0 && self.nsecs == 0
    }

    pub fn weeks(&self) -> i64 {
        self.weeks
    }

    pub fn days_only(&self) -> bool {
        self.months == 0 && self.weeks == 0 && self.days != 0 && self.nsecs == 0
    }

    pub fn days(&self) -> i64 {
        self.days
    }

    /// Returns whether the duration consists of full days.
    ///
    /// Note that 24 hours is not considered a full day due to possible
    /// daylight savings time transitions.
    pub fn is_full_days(&self) -> bool {
        self.nsecs == 0
    }

    pub fn is_constant_duration(&self) -> bool {
        self.months == 0 && self.weeks == 0 && self.days == 0
    }

    /// Returns the nanoseconds from the `Duration` without the weeks or months part.
    pub fn nanoseconds(&self) -> i64 {
        self.nsecs
    }

    /// Estimated duration of the window duration. Not a very good one if months != 0.
    #[doc(hidden)]
    pub const fn duration_ns(&self) -> i64 {
        self.months * 28 * 24 * 3600 * NANOSECONDS
            + self.weeks * NS_WEEK
            + self.days * NS_DAY
            + self.nsecs
    }

    #[doc(hidden)]
    pub const fn duration_us(&self) -> i64 {
        self.months * 28 * 24 * 3600 * MICROSECONDS
            + (self.weeks * NS_WEEK + self.nsecs + self.days * NS_DAY) / 1000
    }

    #[doc(hidden)]
    pub const fn duration_ms(&self) -> i64 {
        self.months * 28 * 24 * 3600 * MILLISECONDS
            + (self.weeks * NS_WEEK + self.nsecs + self.days * NS_DAY) / 1_000_000
    }

    #[doc(hidden)]
    fn add_month(
        ts: NaiveDateTime,
        n_months: i64,
        negative: bool,
        saturating: bool,
    ) -> PolarsResult<NaiveDateTime> {
        let mut months = n_months;
        if negative {
            months = -months;
        }

        // Retrieve the current date and increment the values
        // based on the number of months
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

        if saturating {
            // Normalize the day if we are past the end of the month.
            let mut last_day_of_month = last_day_of_month(month);
            if month == (chrono::Month::February.number_from_month() as i32) && is_leap_year(year) {
                last_day_of_month += 1;
            }

            if day > last_day_of_month {
                day = last_day_of_month
            }
        }

        // Retrieve the original time and construct a data
        // with the new year, month and day
        let hour = ts.hour();
        let minute = ts.minute();
        let sec = ts.second();
        let nsec = ts.nanosecond();
        new_datetime(year, month as u32, day, hour, minute, sec, nsec).ok_or(
            polars_err!(
                ComputeError: format!(
                    "cannot advance '{}' by {} month(s). \
                        If you were trying to get the last day of each month, you may want to try `.dt.month_end` \
                        or append \"_saturating\" to your duration string.",
                        ts,
                        if negative {-n_months} else {n_months}
                )
            ),
        )
    }

    fn truncate_subweekly<G, J>(
        &self,
        t: i64,
        tz: Option<&Tz>,
        duration: i64,
        _timestamp_to_datetime: G,
        _datetime_to_timestamp: J,
        _ambiguous: Ambiguous,
    ) -> PolarsResult<i64>
    where
        G: Fn(i64) -> NaiveDateTime,
        J: Fn(NaiveDateTime) -> i64,
    {
        let t = match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => _datetime_to_timestamp(unlocalize_datetime(_timestamp_to_datetime(t), tz)),
            _ => t,
        };
        let mut remainder = t % duration;
        if remainder < 0 {
            remainder += duration
        }
        match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => Ok(_datetime_to_timestamp(localize_datetime(
                _timestamp_to_datetime(t - remainder),
                tz,
                _ambiguous,
            )?)),
            _ => Ok(t - remainder),
        }
    }

    fn truncate_weekly<G, J>(
        &self,
        t: i64,
        tz: Option<&Tz>,
        timestamp_to_datetime: G,
        datetime_to_timestamp: J,
        _ambiguous: Ambiguous,
    ) -> PolarsResult<i64>
    where
        G: Fn(i64) -> NaiveDateTime,
        J: Fn(NaiveDateTime) -> i64,
    {
        let dt = match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => unlocalize_datetime(timestamp_to_datetime(t), tz).date(),
            _ => timestamp_to_datetime(t).date(),
        };
        let week_timestamp = dt.week(Weekday::Mon);
        let first_day_of_week =
            week_timestamp.first_day() - chrono::Duration::weeks(self.weeks - 1);
        match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => Ok(datetime_to_timestamp(localize_datetime(
                first_day_of_week.and_time(NaiveTime::default()),
                tz,
                _ambiguous,
            )?)),
            _ => Ok(datetime_to_timestamp(
                first_day_of_week.and_time(NaiveTime::default()),
            )),
        }
    }
    fn truncate_monthly<G, J>(
        &self,
        t: i64,
        tz: Option<&Tz>,
        timestamp_to_datetime: G,
        datetime_to_timestamp: J,
        _ambiguous: Ambiguous,
    ) -> PolarsResult<i64>
    where
        G: Fn(i64) -> NaiveDateTime,
        J: Fn(NaiveDateTime) -> i64,
    {
        let ts = match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => unlocalize_datetime(timestamp_to_datetime(t), tz),
            _ => timestamp_to_datetime(t),
        };
        let (year, month) = (ts.year(), ts.month());

        // determine the total number of months and truncate
        // the number of months by the duration amount
        let mut total = (year * 12) + (month as i32 - 1);
        let remainder = total % self.months as i32;
        total -= remainder;

        // recreate a new time from the year and month combination
        let (year, month) = ((total / 12), ((total % 12) + 1) as u32);

        let dt = new_datetime(year, month, 1, 0, 0, 0, 0).ok_or(polars_err!(
            ComputeError: format!("date '{}-{}-1' does not exist", year, month)
        ))?;
        match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => Ok(datetime_to_timestamp(localize_datetime(
                dt, tz, _ambiguous,
            )?)),
            _ => Ok(datetime_to_timestamp(dt)),
        }
    }

    #[inline]
    pub fn truncate_impl<F, G, J>(
        &self,
        t: i64,
        tz: Option<&Tz>,
        nsecs_to_unit: F,
        timestamp_to_datetime: G,
        datetime_to_timestamp: J,
        _ambiguous: Ambiguous,
    ) -> PolarsResult<i64>
    where
        F: Fn(i64) -> i64,
        G: Fn(i64) -> NaiveDateTime,
        J: Fn(NaiveDateTime) -> i64,
    {
        match (self.months, self.weeks, self.days, self.nsecs) {
            (0, 0, 0, 0) => polars_bail!(ComputeError: "duration cannot be zero"),
            // truncate by ns/us/ms
            (0, 0, 0, _) => {
                let duration = nsecs_to_unit(self.nsecs);
                self.truncate_subweekly(
                    t,
                    tz,
                    duration,
                    timestamp_to_datetime,
                    datetime_to_timestamp,
                    _ambiguous,
                )
            },
            // truncate by days
            (0, 0, _, 0) => {
                let duration = self.days * nsecs_to_unit(NS_DAY);
                self.truncate_subweekly(
                    t,
                    tz,
                    duration,
                    timestamp_to_datetime,
                    datetime_to_timestamp,
                    _ambiguous,
                )
            },
            // truncate by weeks
            (0, _, 0, 0) => self.truncate_weekly(
                t,
                tz,
                timestamp_to_datetime,
                datetime_to_timestamp,
                _ambiguous,
            ),
            // truncate by months
            (_, 0, 0, 0) => self.truncate_monthly(
                t,
                tz,
                timestamp_to_datetime,
                datetime_to_timestamp,
                _ambiguous,
            ),
            _ => {
                polars_bail!(ComputeError: "duration may not mix month, weeks and nanosecond units")
            },
        }
    }

    // Truncate the given ns timestamp by the window boundary.
    #[inline]
    pub fn truncate_ns(&self, t: i64, tz: Option<&Tz>, ambiguous: Ambiguous) -> PolarsResult<i64> {
        self.truncate_impl(
            t,
            tz,
            |nsecs| nsecs,
            timestamp_ns_to_datetime,
            datetime_to_timestamp_ns,
            ambiguous,
        )
    }

    // Truncate the given ns timestamp by the window boundary.
    #[inline]
    pub fn truncate_us(&self, t: i64, tz: Option<&Tz>, ambiguous: Ambiguous) -> PolarsResult<i64> {
        self.truncate_impl(
            t,
            tz,
            |nsecs| nsecs / 1000,
            timestamp_us_to_datetime,
            datetime_to_timestamp_us,
            ambiguous,
        )
    }

    // Truncate the given ms timestamp by the window boundary.
    #[inline]
    pub fn truncate_ms(&self, t: i64, tz: Option<&Tz>, ambiguous: Ambiguous) -> PolarsResult<i64> {
        self.truncate_impl(
            t,
            tz,
            |nsecs| nsecs / 1_000_000,
            timestamp_ms_to_datetime,
            datetime_to_timestamp_ms,
            ambiguous,
        )
    }

    fn add_impl_month_week_or_day<F, G, J>(
        &self,
        t: i64,
        tz: Option<&Tz>,
        nsecs_to_unit: F,
        timestamp_to_datetime: G,
        datetime_to_timestamp: J,
    ) -> PolarsResult<i64>
    where
        F: Fn(i64) -> i64,
        G: Fn(i64) -> NaiveDateTime,
        J: Fn(NaiveDateTime) -> i64,
    {
        let d = self;
        let mut new_t = t;

        if d.months > 0 {
            let ts = match tz {
                #[cfg(feature = "timezones")]
                Some(tz) => unlocalize_datetime(timestamp_to_datetime(t), tz),
                _ => timestamp_to_datetime(t),
            };
            let dt = Self::add_month(ts, d.months, d.negative, d.saturating)?;
            new_t = match tz {
                #[cfg(feature = "timezones")]
                Some(tz) => datetime_to_timestamp(localize_datetime(dt, tz, Ambiguous::Raise)?),
                _ => datetime_to_timestamp(dt),
            };
        }

        if d.weeks > 0 {
            let t_weeks = nsecs_to_unit(self.weeks * NS_WEEK);
            match tz {
                #[cfg(feature = "timezones")]
                Some(tz) => {
                    new_t =
                        datetime_to_timestamp(unlocalize_datetime(timestamp_to_datetime(t), tz));
                    new_t += if d.negative { -t_weeks } else { t_weeks };
                    new_t = datetime_to_timestamp(localize_datetime(
                        timestamp_to_datetime(new_t),
                        tz,
                        Ambiguous::Raise,
                    )?);
                },
                _ => new_t += if d.negative { -t_weeks } else { t_weeks },
            };
        }

        if d.days > 0 {
            let t_days = nsecs_to_unit(self.days * NS_DAY);
            match tz {
                #[cfg(feature = "timezones")]
                Some(tz) => {
                    new_t =
                        datetime_to_timestamp(unlocalize_datetime(timestamp_to_datetime(t), tz));
                    new_t += if d.negative { -t_days } else { t_days };
                    new_t = datetime_to_timestamp(localize_datetime(
                        timestamp_to_datetime(new_t),
                        tz,
                        Ambiguous::Raise,
                    )?);
                },
                _ => new_t += if d.negative { -t_days } else { t_days },
            };
        }

        Ok(new_t)
    }

    pub fn add_ns(&self, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
        let d = self;
        let new_t = self.add_impl_month_week_or_day(
            t,
            tz,
            |nsecs| nsecs,
            timestamp_ns_to_datetime,
            datetime_to_timestamp_ns,
        );
        let nsecs = if d.negative { -d.nsecs } else { d.nsecs };
        Ok(new_t? + nsecs)
    }

    pub fn add_us(&self, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
        let d = self;
        let new_t = self.add_impl_month_week_or_day(
            t,
            tz,
            |nsecs| nsecs / 1000,
            timestamp_us_to_datetime,
            datetime_to_timestamp_us,
        );
        let nsecs = if d.negative { -d.nsecs } else { d.nsecs };
        Ok(new_t? + nsecs / 1_000)
    }

    pub fn add_ms(&self, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
        let d = self;
        let new_t = self.add_impl_month_week_or_day(
            t,
            tz,
            |nsecs| nsecs / 1_000_000,
            timestamp_ms_to_datetime,
            datetime_to_timestamp_ms,
        );
        let nsecs = if d.negative { -d.nsecs } else { d.nsecs };
        Ok(new_t? + nsecs / 1_000_000)
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
        self.weeks *= rhs;
        self.days *= rhs;
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
) -> Option<NaiveDateTime> {
    let date = NaiveDate::from_ymd_opt(year, month, days)?;
    let time = NaiveTime::from_hms_nano_opt(hour, min, sec, nano)?;
    Some(NaiveDateTime::new(date, time))
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
        assert_eq!(out.nsecs, 40 * NS_MILLISECOND + 123);
        assert_eq!(out.duration_ns(), 40 * NS_MILLISECOND + 123 + NS_WEEK);
        let out = Duration::parse("-123ns40ms1w");
        assert!(out.negative);
        let out = Duration::parse("5w");
        assert_eq!(out.weeks(), 5);
    }

    #[test]
    fn test_add_ns() {
        let t = 1;
        let seven_days = Duration::parse("7d");
        let one_week = Duration::parse("1w");

        // add_ns can only error if a time zone is passed, so it's
        // safe to unwrap here
        assert_eq!(
            seven_days.add_ns(t, None).unwrap(),
            one_week.add_ns(t, None).unwrap()
        );

        let seven_days_negative = Duration::parse("-7d");
        let one_week_negative = Duration::parse("-1w");

        // add_ns can only error if a time zone is passed, so it's
        // safe to unwrap here
        assert_eq!(
            seven_days_negative.add_ns(t, None).unwrap(),
            one_week_negative.add_ns(t, None).unwrap()
        );
    }
}
