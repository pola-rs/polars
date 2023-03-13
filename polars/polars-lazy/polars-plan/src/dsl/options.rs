use std::borrow::Cow;

use polars_core::datatypes::DataType;
use polars_core::prelude::{JoinType, TimeUnit};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StrpTimeOptions {
    /// DataType to parse in. One of {Date, Datetime}
    pub date_dtype: DataType,
    /// Formatting string
    pub fmt: Option<String>,
    /// If set then polars will return an error if any date parsing fails
    pub strict: bool,
    /// If polars may parse matches that not contain the whole string
    /// e.g. "foo-2021-01-01-bar" could match "2021-01-01"
    pub exact: bool,
    /// use a cache of unique, converted dates to apply the datetime conversion.
    pub cache: bool,
    /// Parse a timezone aware timestamp
    pub tz_aware: bool,
    /// Convert timezone aware to UTC
    pub utc: bool,
}

impl Default for StrpTimeOptions {
    fn default() -> Self {
        StrpTimeOptions {
            date_dtype: DataType::Datetime(TimeUnit::Microseconds, None),
            fmt: None,
            strict: false,
            exact: false,
            cache: true,
            tz_aware: false,
            utc: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct JoinOptions {
    pub allow_parallel: bool,
    pub force_parallel: bool,
    pub how: JoinType,
    pub suffix: Cow<'static, str>,
    pub slice: Option<(i64, usize)>,
    /// Proxy of the number of rows in both sides of the joins
    /// Holds `(Option<known_size>, estimated_size)`
    pub rows_left: (Option<usize>, usize),
    pub rows_right: (Option<usize>, usize),
}

impl Default for JoinOptions {
    fn default() -> Self {
        JoinOptions {
            allow_parallel: true,
            force_parallel: false,
            how: JoinType::Left,
            suffix: "_right".into(),
            slice: None,
            rows_left: (None, usize::MAX),
            rows_right: (None, usize::MAX),
        }
    }
}
