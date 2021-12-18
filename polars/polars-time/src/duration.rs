use crate::calendar::{
    is_leap_year, last_day_of_month, timestamp_ns_to_datetime, NS_DAY, NS_HOUR, NS_MICROSECOND,
    NS_MILLISECOND, NS_MINUTE, NS_SECOND, NS_WEEK,
};
use crate::unit::TimeNanoseconds;
use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use std::ops::{Add, Mul, Sub};

#[derive(Copy, Clone, Debug)]
pub struct Duration {
    // the number of months for the duration
    months: i64,
    // the number of nanoseconds for the duration
    nsecs: i64,
    // indicates if the duration is negative
    negative: bool,
}

impl Duration {
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
    ///
    /// 3d12h4m25s // 3 days, 12 hours, 4 minutes, and 25 seconds
    ///
    /// # Panics if given str is incorrect
    pub fn parse(duration: &str) -> Self {
        let mut nsecs = 0;
        let mut months = 0;
        let mut iter = duration.char_indices();
        let negative = duration.starts_with('-');
        // skip the '-' char
        if negative {
            iter.next().unwrap();
        }

        let mut start = 0;

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
                    unit => panic!("unit: '{}' not supported", unit),
                }
                unit.clear();
            }
        }
        Duration {
            nsecs: nsecs.abs(),
            months: months.abs(),
            negative,
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
            let mut offset = self.duration();
            if offset == 0 {
                return *self;
            }
            let every = interval.duration();

            if offset < 0 {
                offset += every * ((offset / -every) + 1)
            } else {
                offset -= every * (offset / every)
            }
            Duration::from_nsecs(offset)
        }
    }

    /// Creates a [`Duration`] that represents a fixed number of nanoseconds.
    pub fn from_nsecs(v: i64) -> Self {
        let (negative, nsecs) = Self::to_positive(v);
        Self {
            months: 0,
            nsecs,
            negative,
        }
    }

    pub fn from_seconds(v: i64) -> Self {
        Self::from_nsecs(v * NS_SECOND)
    }

    pub fn from_minutes(v: i64) -> Self {
        Self::from_nsecs(v * NS_MINUTE)
    }

    /// Creates a [`Duration`] that represents a fixed number of months.
    pub fn from_months(v: i64) -> Self {
        let (negative, months) = Self::to_positive(v);
        Self {
            months,
            nsecs: 0,
            negative,
        }
    }

    /// `true` if zero duration.
    pub fn is_zero(&self) -> bool {
        self.months == 0 && self.nsecs == 0
    }

    fn months_only(&self) -> bool {
        self.months != 0 && self.nsecs == 0
    }

    pub fn months(&self) -> i64 {
        self.months
    }

    pub fn nanoseconds(&self) -> i64 {
        self.nsecs
    }

    /// Estimated duration of the window duration. Not a very good one if months != 0.
    #[inline]
    pub const fn duration(&self) -> TimeNanoseconds {
        self.months * 30 * 24 * 3600 * NS_SECOND + self.nsecs
    }

    // Truncate the given nanoseconds timestamp by the window boundary.
    #[inline]
    pub fn truncate_nanoseconds(&self, t: i64) -> i64 {
        match (self.months, self.nsecs) {
            (0, 0) => panic!("duration may not be zero"),
            // truncate by nanoseconds
            (0, _) => {
                let duration = self.nsecs as i64;
                let mut remainder = t % duration;
                if remainder < 0 {
                    remainder += duration
                }
                t - remainder
            }
            // truncate by months
            (_, 0) => {
                let ts = nsecs_timestamp_to_datetime(t);
                let (year, month) = (ts.year(), ts.month());

                // determine the total number of months and truncate
                // the number of months by the duration amount
                let mut total = (year * 12) as i32 + (month - 1) as i32;
                let remainder = total % self.months as i32;
                total -= remainder;

                // recreate a new time from the year and month combination
                let (year, month) = ((total / 12), ((total % 12) + 1) as u32);
                new_datetime(year, month, 1, 0, 0, 0, 0).timestamp_nanos()
            }
            _ => panic!("duration may not mix month and nanosecond units"),
        }
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

impl Add<Duration> for TimeNanoseconds {
    type Output = Self;

    /// Adds a duration to a nanosecond timestamp

    fn add(self, rhs: Duration) -> Self {
        let t = self;
        let d = rhs;
        let mut new_t = t;

        if d.months > 0 {
            let mut months = d.months;
            if d.negative {
                months = -months;
            }

            // Retrieve the current date and increment the values
            // based on the number of months
            let ts = timestamp_ns_to_datetime(t);
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
            let ts =
                new_datetime(year, month as u32, day, hour, minute, sec, nsec).timestamp_nanos();
            new_t = ts;
        }
        let nsecs = if d.negative { -d.nsecs } else { d.nsecs };

        // original silently overflows:
        // see https://github.com/influxdata/influxdb_iox/issues/2890

        // We keep the panic for now until we better understand the issue
        new_t + nsecs
    }
}

impl Sub<Duration> for TimeNanoseconds {
    type Output = Self;

    fn sub(self, rhs: Duration) -> Self::Output {
        self + rhs * -1
    }
}

fn nsecs_timestamp_to_datetime(ts: i64) -> NaiveDateTime {
    let secs = ts / 1_000_000_000;
    let nsec = ts % 1_000_000_000;
    // Note that nsec as u32 is safe here because modulo on a negative ts value
    //  still produces a positive remainder.
    NaiveDateTime::from_timestamp(secs, nsec as u32)
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
    let date = NaiveDate::from_ymd(year, month, days);
    let time = NaiveTime::from_hms_nano(hour, min, sec, nano);

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
