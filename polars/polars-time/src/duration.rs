use crate::calendar::{
    is_leap_year, last_day_of_month, timestamp_ns_to_datetime, NS_MINUTE, NS_SECONDS,
};
use crate::unit::{TimeMilliseconds, TimeNanoseconds};
use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use std::ops::{Add, Mul};

#[derive(Copy, Clone)]
pub struct Duration {
    // the number of months for the duration
    months: i64,
    // the number of nanoseconds for the duration
    nsecs: i64,
    // indicates if the duration is negative
    negative: bool,
}

impl Duration {
    fn to_positive(v: i64) -> (bool, i64) {
        if v < 0 {
            (true, -v)
        } else {
            (false, v)
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
        Self::from_nsecs(v * NS_SECONDS)
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
    fn is_zero(&self) -> bool {
        self.months == 0 && self.nsecs == 0
    }

    pub fn months(&self) -> i64 {
        self.months
    }

    pub fn nanoseconds(&self) -> i64 {
        self.nsecs
    }

    /// Estimated duration of the window duration. Not a very good one if months != 0.
    pub fn duration(&self) -> TimeNanoseconds {
        (self.months * 30 * 24 * 3600 * NS_SECONDS + self.nsecs).into()
    }

    // Truncate the given nanoseconds timestamp by the window boundary.
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
        let t = *self;
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
        (new_t + nsecs).into()
    }
}

impl Add<Duration> for TimeMilliseconds {
    type Output = Self;

    /// Adds a duration to a nanosecond timestamp

    fn add(self, rhs: Duration) -> Self {
        self.to_nsecs().add(rhs).to_millisecs()
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
