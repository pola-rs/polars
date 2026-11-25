#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

/// Unit in which `date_diff` counts the boundaries crossed between two temporal values.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[strum(serialize_all = "snake_case")]
pub enum DateDiffUnit {
    Year,
    Quarter,
    Month,
    Week,
    Day,
    Hour,
    Minute,
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
}

impl DateDiffUnit {
    /// Length of the unit in nanoseconds, or `None` for units of a day or longer.
    pub fn nanoseconds(self) -> Option<i64> {
        Some(match self {
            Self::Hour => 3_600_000_000_000,
            Self::Minute => 60_000_000_000,
            Self::Second => 1_000_000_000,
            Self::Millisecond => 1_000_000,
            Self::Microsecond => 1_000,
            Self::Nanosecond => 1,
            Self::Year | Self::Quarter | Self::Month | Self::Week | Self::Day => return None,
        })
    }
}
