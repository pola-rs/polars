use super::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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
            }
            TimeUnit::Microseconds => {
                write!(f, "μs")
            }
            TimeUnit::Milliseconds => {
                write!(f, "ms")
            }
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
pub(crate) fn convert_time_units(v: i64, tu_l: TimeUnit, tu_r: TimeUnit) -> i64 {
    use TimeUnit::*;
    match (tu_l, tu_r) {
        (Nanoseconds, Microseconds) => v / 1_000,
        (Nanoseconds, Milliseconds) => v / 1_000_000,
        (Microseconds, Nanoseconds) => v * 1_000,
        (Microseconds, Milliseconds) => v / 1_000,
        (Milliseconds, Microseconds) => v * 1_000,
        (Milliseconds, Nanoseconds) => v * 1_000_000,
        _ => v,
    }
}
