use std::cmp::Ordering;

#[cfg(feature = "temporal")]
use chrono::NaiveDateTime;
#[cfg(feature = "temporal")]
use polars_arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};
use polars_error::{PolarsError, polars_err};

#[cfg(feature = "temporal")]
use crate::chunked_array::temporal::{
    datetime_to_timestamp_ms, datetime_to_timestamp_ns, datetime_to_timestamp_us,
};
use crate::prelude::{ArrowTimeUnit, DataType};

/// Note: variants are ordered from finest to coarsest (`Nanoseconds < Milliseconds`).
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
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

    /// How integer values in this unit are rescaled to `to`.
    pub const fn rescale_to(self, to: TimeUnit) -> TimeUnitRescale {
        use TimeUnit::*;
        match (self, to) {
            (Nanoseconds, Microseconds) | (Microseconds, Milliseconds) => {
                TimeUnitRescale::Divide(1_000)
            },
            (Nanoseconds, Milliseconds) => TimeUnitRescale::Divide(1_000_000),
            (Microseconds, Nanoseconds) | (Milliseconds, Microseconds) => {
                TimeUnitRescale::Multiply(1_000)
            },
            (Milliseconds, Nanoseconds) => TimeUnitRescale::Multiply(1_000_000),
            (Nanoseconds, Nanoseconds)
            | (Microseconds, Microseconds)
            | (Milliseconds, Milliseconds) => TimeUnitRescale::Unchanged,
        }
    }
}

/// How integer values in one [`TimeUnit`] are rescaled to another.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TimeUnitRescale {
    Unchanged,
    Multiply(i64),
    Divide(i64),
}

impl TimeUnitRescale {
    /// How each (weeks, days, hours, minutes, seconds, milliseconds, microseconds,
    /// nanoseconds) duration component is rescaled to `time_unit`: multiplied if coarser,
    /// or truncated if finer (each component separately).
    pub fn duration_components(time_unit: TimeUnit) -> [Self; 8] {
        const NS: i64 = 1_000_000_000;
        const SCALE_NS: [i64; 8] = [
            NS * 86_400 * 7, // weeks
            NS * 86_400,     // days
            NS * 3_600,      // hours
            NS * 60,         // minutes
            NS,              // seconds
            1_000_000,       // milliseconds
            1_000,           // microseconds
            1,               // nanoseconds
        ];
        let unit_ns = match time_unit {
            TimeUnit::Nanoseconds => 1,
            TimeUnit::Microseconds => 1_000,
            TimeUnit::Milliseconds => 1_000_000,
        };
        SCALE_NS.map(|ns| match ns.cmp(&unit_ns) {
            Ordering::Greater => TimeUnitRescale::Multiply(ns / unit_ns),
            Ordering::Less => TimeUnitRescale::Divide(unit_ns / ns),
            Ordering::Equal => TimeUnitRescale::Unchanged,
        })
    }

    /// Rescales a datetime timestamp, rounding down; `None` on overflow.
    #[inline]
    pub fn datetime_value(self, v: i64) -> Option<i64> {
        use polars_utils::floor_divmod::FloorDivMod;
        match self {
            TimeUnitRescale::Unchanged => Some(v),
            TimeUnitRescale::Multiply(k) => v.checked_mul(k),
            TimeUnitRescale::Divide(k) => Some(v.wrapping_floor_div_mod(k).0),
        }
    }

    /// Rescales a duration, rounding towards zero; `None` on overflow.
    #[inline]
    pub fn duration_value(self, v: i64) -> Option<i64> {
        match self {
            TimeUnitRescale::Unchanged => Some(v),
            TimeUnitRescale::Multiply(k) => v.checked_mul(k),
            TimeUnitRescale::Divide(k) => Some(v.wrapping_div(k)),
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

/// Error for a temporal value that can't be represented in another time unit.
pub fn time_unit_conversion_error(
    from: impl std::fmt::Display,
    to: &DataType,
    value: impl std::fmt::Display,
) -> PolarsError {
    polars_err!(InvalidOperation: "conversion from `{from}` to `{to}` failed for value: {value}")
}
