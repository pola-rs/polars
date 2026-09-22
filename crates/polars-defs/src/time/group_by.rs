use polars_core::datatypes::{DataType, TimeUnit};
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

use crate::time::duration::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[strum(serialize_all = "snake_case")]
pub enum ClosedWindow {
    Left,
    Right,
    Both,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[strum(serialize_all = "snake_case")]
pub enum Label {
    Left,
    Right,
    DataPoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[strum(serialize_all = "snake_case")]
#[derive(Default)]
pub enum StartBy {
    #[default]
    WindowBound,
    DataPoint,
    /// only useful if periods are weekly
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

impl StartBy {
    pub fn weekday(&self) -> Option<u32> {
        match self {
            StartBy::Monday => Some(0),
            StartBy::Tuesday => Some(1),
            StartBy::Wednesday => Some(2),
            StartBy::Thursday => Some(3),
            StartBy::Friday => Some(4),
            StartBy::Saturday => Some(5),
            StartBy::Sunday => Some(6),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct DynamicGroupOptions {
    /// Time or index column.
    pub index_column: PlSmallStr,
    /// Start a window at this interval.
    pub every: Duration,
    /// Window duration.
    pub period: Duration,
    /// Offset window boundaries.
    pub offset: Duration,
    /// Truncate the time column values to the window.
    pub label: Label,
    /// Add the boundaries to the DataFrame.
    pub include_boundaries: bool,
    pub closed_window: ClosedWindow,
    pub start_by: StartBy,
}

impl Default for DynamicGroupOptions {
    fn default() -> Self {
        Self {
            index_column: "".into(),
            every: Duration::new(1),
            period: Duration::new(1),
            offset: Duration::new(1),
            label: Label::Left,
            include_boundaries: false,
            closed_window: ClosedWindow::Left,
            start_by: Default::default(),
        }
    }
}

/// The dtype of the `_lower_boundary` and `_upper_boundary` columns of a dynamic group-by on
/// an index of `index_dtype`.
///
/// A `Date` index gets `Datetime` boundaries, because `every`, `period` and `offset` may be
/// sub-day and a `Date` cannot hold the resulting window bounds.
pub fn dynamic_boundary_dtype(index_dtype: &DataType) -> DataType {
    if index_dtype.is_date() {
        DataType::Datetime(TimeUnit::Microseconds, None)
    } else {
        index_dtype.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct RollingGroupOptions {
    /// Time or index column.
    pub index_column: PlSmallStr,
    /// Window duration.
    pub period: Duration,
    pub offset: Duration,
    pub closed_window: ClosedWindow,
}

impl Default for RollingGroupOptions {
    fn default() -> Self {
        Self {
            index_column: "".into(),
            period: Duration::new(1),
            offset: Duration::new(1),
            closed_window: ClosedWindow::Left,
        }
    }
}
