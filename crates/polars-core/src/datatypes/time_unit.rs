use super::*;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Eq, Hash)]
#[cfg_attr(
    any(feature = "serde-lazy", feature = "serde"),
    derive(Serialize, Deserialize)
)]
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

impl Display for TimeUnit {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
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
}

#[inline]
pub(crate) fn conversion_factor_time_units(tu_l: &TimeUnit, tu_r: &TimeUnit) -> f64 {
    timeunit_scale(tu_l.to_arrow(), tu_r.to_arrow())
}

#[inline]
pub(crate) fn convert_time_units(v: i64, tu_l: TimeUnit, tu_r: TimeUnit) -> i64 {
    let factor = conversion_factor_time_units(&tu_l, &tu_r);

    (v as f64 * factor) as i64
}

#[inline]
#[cfg(feature = "temporal")]
pub(crate) fn get_seconds_in_day(tu: &TimeUnit) -> i64 {
    match tu {
        TimeUnit::Milliseconds => MS_IN_DAY,
        TimeUnit::Microseconds => US_IN_DAY,
        TimeUnit::Nanoseconds => NS_IN_DAY,
    }
}

/// Largely based on nano-arrow::timestamp_to_naive_datetime, but just returns the
/// method instead of doing the calculations.
#[inline]
pub fn timestamp_to_naive_datetime_method(
    time_unit: &TimeUnit,
) -> fn(i64) -> chrono::NaiveDateTime {
    match time_unit {
        TimeUnit::Milliseconds => timestamp_ms_to_datetime,
        TimeUnit::Microseconds => timestamp_us_to_datetime,
        TimeUnit::Nanoseconds => timestamp_ns_to_datetime,
    }
}

#[inline]
pub fn datetime_to_timestamp_method(time_unit: &TimeUnit) -> fn(chrono::NaiveDateTime) -> i64 {
    match time_unit {
        TimeUnit::Milliseconds => datetime_to_timestamp_ms,
        TimeUnit::Microseconds => datetime_to_timestamp_us,
        TimeUnit::Nanoseconds => datetime_to_timestamp_ns,
    }
}
