#[cfg(feature = "temporal")]
use chrono::NaiveDateTime;
#[cfg(feature = "temporal")]
use polars_arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};

#[cfg(feature = "temporal")]
use crate::chunked_array::temporal::{
    datetime_to_timestamp_ms, datetime_to_timestamp_ns, datetime_to_timestamp_us,
};
use crate::prelude::ArrowTimeUnit;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Eq, Hash)]
#[cfg_attr(
    any(feature = "serde-lazy", feature = "serde"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum TimeUnit {
    Nanoseconds,
    Microseconds,
    Milliseconds,
}

impl From<&ArrowTimeUnit> for TimeUnit {
    fn from(tu: &ArrowTimeUnit) -> Self {
        match tu {
            ArrowTimeUnit::Nanosecond => TimeUnit::Nanoseconds,
            ArrowTimeUnit::Microsecond => TimeUnit::Microseconds,
            ArrowTimeUnit::Millisecond => TimeUnit::Milliseconds,
            // will be cast
            ArrowTimeUnit::Second => TimeUnit::Milliseconds,
        }
    }
}

impl std::fmt::Display for TimeUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeUnit::Nanoseconds => {
                write!(f, "ns")
            },
            TimeUnit::Microseconds => {
                write!(f, "μs")
            },
            TimeUnit::Milliseconds => {
                write!(f, "ms")
            },
        }
    }
}

impl TimeUnit {
    pub fn to_ascii(self) -> &'static str {
        use TimeUnit::*;
        match self {
            Nanoseconds => "ns",
            Microseconds => "us",
            Milliseconds => "ms",
        }
    }

    pub fn to_arrow(self) -> ArrowTimeUnit {
        match self {
            TimeUnit::Nanoseconds => ArrowTimeUnit::Nanosecond,
            TimeUnit::Microseconds => ArrowTimeUnit::Microsecond,
            TimeUnit::Milliseconds => ArrowTimeUnit::Millisecond,
        }
    }

    /// `ns` nanoseconds expressed in this unit, truncated towards zero.
    #[inline(always)]
    pub const fn from_nsecs(self, ns: i64) -> i64 {
        match self {
            TimeUnit::Nanoseconds => ns,
            TimeUnit::Microseconds => ns / 1_000,
            TimeUnit::Milliseconds => ns / 1_000_000,
        }
    }
}

#[cfg(feature = "temporal")]
impl TimeUnit {
    /// The naive UTC datetime of the timestamp `t` in this unit.
    #[inline]
    pub fn timestamp_to_datetime(self, t: i64) -> NaiveDateTime {
        match self {
            TimeUnit::Nanoseconds => timestamp_ns_to_datetime(t),
            TimeUnit::Microseconds => timestamp_us_to_datetime(t),
            TimeUnit::Milliseconds => timestamp_ms_to_datetime(t),
        }
    }

    /// The naive UTC datetime `dt` as a timestamp in this unit.
    #[inline]
    pub fn datetime_to_timestamp(self, dt: NaiveDateTime) -> i64 {
        match self {
            TimeUnit::Nanoseconds => datetime_to_timestamp_ns(dt),
            TimeUnit::Microseconds => datetime_to_timestamp_us(dt),
            TimeUnit::Milliseconds => datetime_to_timestamp_ms(dt),
        }
    }
}

#[cfg(any(feature = "rows", feature = "object"))]
#[cfg(any(feature = "dtype-datetime", feature = "dtype-duration"))]
#[inline]
pub fn convert_time_units<T>(v: T, tu_l: TimeUnit, tu_r: TimeUnit) -> T
where
    T: num_traits::Num + num_traits::NumCast,
{
    use TimeUnit::*;
    match (tu_l, tu_r) {
        (Nanoseconds, Microseconds) => v / T::from(1_000).unwrap(),
        (Nanoseconds, Milliseconds) => v / T::from(1_000_000).unwrap(),
        (Microseconds, Nanoseconds) => v * T::from(1_000).unwrap(),
        (Microseconds, Milliseconds) => v / T::from(1_000).unwrap(),
        (Milliseconds, Microseconds) => v * T::from(1_000).unwrap(),
        (Milliseconds, Nanoseconds) => v * T::from(1_000_000).unwrap(),
        _ => v,
    }
}
