use crate::prelude::*;
use chrono::{NaiveDateTime, NaiveTime, Timelike};

// Conversion extracted from:
// https://docs.rs/arrow/1.0.0/src/arrow/array/array.rs.html#589

/// Number of seconds in a day
const SECONDS_IN_DAY: i64 = 86_400;
/// Number of milliseconds in a second
const MILLISECONDS_IN_SECOND: i64 = 1_000;
/// Number of microseconds in a second
const MICROSECONDS_IN_SECOND: i64 = 1_000_000;
/// Number of nanoseconds in a second
const NANOSECONDS_IN_SECOND: i64 = 1_000_000_000;

pub fn date32_as_datetime(v: i32) -> NaiveDateTime {
    NaiveDateTime::from_timestamp(v as i64 * SECONDS_IN_DAY, 0)
}

pub fn date64_as_datetime(v: i64) -> NaiveDateTime {
    NaiveDateTime::from_timestamp(
        // extract seconds from milliseconds
        v / MILLISECONDS_IN_SECOND,
        // discard extracted seconds and convert milliseconds to nanoseconds
        (v % MILLISECONDS_IN_SECOND * MICROSECONDS_IN_SECOND) as u32,
    )
}

pub fn time64_nanosecond_as_time(v: i64) -> NaiveTime {
    NaiveTime::from_num_seconds_from_midnight(
        // extract seconds from nanoseconds
        (v / NANOSECONDS_IN_SECOND) as u32,
        // discard extracted seconds
        (v % NANOSECONDS_IN_SECOND) as u32,
    )
}

pub fn naive_time_to_time64_nanoseconds(v: &NaiveTime) -> i64 {
    // 3600 seconds in an hour
    v.hour() as i64 * 3600 * NANOSECONDS_IN_SECOND
        // 60 seconds in a minute
        + v.minute() as i64 * 60 * NANOSECONDS_IN_SECOND
        + v.second() as i64 * NANOSECONDS_IN_SECOND
        + v.nanosecond() as i64
}

pub fn naive_time_to_time64_microseconds(v: &NaiveTime) -> i64 {
    v.hour() as i64 * 3600 * MICROSECONDS_IN_SECOND
        + v.minute() as i64 * 60 * MICROSECONDS_IN_SECOND
        + v.second() as i64 * MICROSECONDS_IN_SECOND
        + v.nanosecond() as i64 / 1000
}

pub fn naive_time_to_time32_milliseconds(v: &NaiveTime) -> i32 {
    v.hour() as i32 * 3600 * MILLISECONDS_IN_SECOND as i32
        + v.minute() as i32 * 60 * MILLISECONDS_IN_SECOND as i32
        + v.second() as i32 * MILLISECONDS_IN_SECOND as i32
        + v.nanosecond() as i32 / 1000_000
}

pub fn naive_time_to_time32_seconds(v: &NaiveTime) -> i32 {
    v.hour() as i32 * 3600 + v.minute() as i32 * 60 + v.second() as i32 + v.nanosecond() as i32
}

pub fn time64_microsecond_as_time(v: i64) -> NaiveTime {
    NaiveTime::from_num_seconds_from_midnight(
        // extract seconds from microseconds
        (v / MICROSECONDS_IN_SECOND) as u32,
        // discard extracted seconds and convert microseconds to
        // nanoseconds
        (v % MICROSECONDS_IN_SECOND * MILLISECONDS_IN_SECOND) as u32,
    )
}

pub fn time32_second_as_time(v: i32) -> NaiveTime {
    NaiveTime::from_num_seconds_from_midnight(v as u32, 0)
}

pub fn time32_millisecond_as_time(v: i32) -> NaiveTime {
    let v = v as u32;
    NaiveTime::from_num_seconds_from_midnight(
        // extract seconds from milliseconds
        v / MILLISECONDS_IN_SECOND as u32,
        // discard extracted seconds and convert milliseconds to
        // nanoseconds
        v % MILLISECONDS_IN_SECOND as u32 * MICROSECONDS_IN_SECOND as u32,
    )
}

pub fn unix_time() -> NaiveDateTime {
    NaiveDateTime::from_timestamp(0, 0)
}

pub trait FromNaiveTime<T, N> {
    fn new_from_naive_time(name: &str, v: &[N]) -> Self;

    fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> Self;
}

fn parse_from_str(s: &str, fmt: &str) -> Option<NaiveTime> {
    NaiveTime::parse_from_str(s, fmt).ok()
}

macro_rules! impl_from_naive_time {
    ($arrowtype:ident, $chunkedtype:ident, $func:ident) => {
        impl FromNaiveTime<$arrowtype, NaiveTime> for $chunkedtype {
            fn new_from_naive_time(name: &str, v: &[NaiveTime]) -> Self {
                let unit = v.iter().map($func).collect::<AlignedVec<_>>();
                ChunkedArray::new_from_aligned_vec(name, unit)
            }

            fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> Self {
                ChunkedArray::new_from_opt_iter(
                    name,
                    v.iter().map(|s| parse_from_str(s, fmt).as_ref().map($func)),
                )
            }
        }
    };
}

impl_from_naive_time!(
    Time64NanosecondType,
    Time64NanosecondChunked,
    naive_time_to_time64_nanoseconds
);
impl_from_naive_time!(
    Time64MicrosecondType,
    Time64MicrosecondChunked,
    naive_time_to_time64_microseconds
);
impl_from_naive_time!(
    Time32MillisecondType,
    Time32MillisecondChunked,
    naive_time_to_time32_milliseconds
);
impl_from_naive_time!(
    Time32SecondType,
    Time32SecondChunked,
    naive_time_to_time32_seconds
);

pub trait AsNaiveTime {
    fn as_naive_time(&self) -> Vec<Option<NaiveTime>>;
}

macro_rules! impl_as_naivetime {
    ($ca:ty, $fun:ident) => {
        impl AsNaiveTime for $ca {
            fn as_naive_time(&self) -> Vec<Option<NaiveTime>> {
                self.into_iter().map(|opt_t| opt_t.map($fun)).collect()
            }
        }
    };
}

impl_as_naivetime!(Time32SecondChunked, time32_second_as_time);
impl_as_naivetime!(&Time32SecondChunked, time32_second_as_time);
impl_as_naivetime!(Time32MillisecondChunked, time32_millisecond_as_time);
impl_as_naivetime!(&Time32MillisecondChunked, time32_millisecond_as_time);
impl_as_naivetime!(Time64NanosecondChunked, time64_nanosecond_as_time);
impl_as_naivetime!(&Time64NanosecondChunked, time64_nanosecond_as_time);
impl_as_naivetime!(Time64MicrosecondChunked, time64_microsecond_as_time);
impl_as_naivetime!(&Time64MicrosecondChunked, time64_microsecond_as_time);

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use chrono::NaiveTime;

    #[test]
    fn from_time() {
        let times: Vec<_> = ["23:56:04", "00:00:00"]
            .iter()
            .map(|s| NaiveTime::parse_from_str(s, "%H:%M:%S").unwrap())
            .collect();
        let t = Time64NanosecondChunked::new_from_naive_time("times", &times);
        // NOTE: the values are checked and correct.
        assert_eq!([86164000000000, 0], t.cont_slice().unwrap());
        let t = Time64MicrosecondChunked::new_from_naive_time("times", &times);
        assert_eq!([86164000000, 0], t.cont_slice().unwrap());
        let t = Time32MillisecondChunked::new_from_naive_time("times", &times);
        assert_eq!([86164000, 0], t.cont_slice().unwrap());
        let t = Time32SecondChunked::new_from_naive_time("times", &times);
        assert_eq!([86164, 0], t.cont_slice().unwrap());
    }
}
