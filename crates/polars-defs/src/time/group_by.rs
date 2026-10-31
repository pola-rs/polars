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

/// A half-open range `start..end` of index values in the physical `i64` space of the index
/// column: `Datetime` in its own time unit, `Date` as microseconds, integers as themselves.
///
/// `(i64::MIN, None)` stands for an unbounded range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IndexRange {
    start: i64,
    end: Option<i64>,
}

impl IndexRange {
    pub const ALL: Self = Self {
        start: i64::MIN,
        end: None,
    };

    /// The range `start..end`.
    pub const fn new(start: i64, end: Option<i64>) -> Self {
        Self { start, end }
    }

    pub fn start(&self) -> i64 {
        self.start
    }

    pub fn end(&self) -> Option<i64> {
        self.end
    }

    pub fn contains(&self, t: i64) -> bool {
        !self.is_before(t) && !self.is_past(t)
    }

    /// Whether `t`, and so every earlier value of an ascending index, lies before the range.
    pub fn is_before(&self, t: i64) -> bool {
        t < self.start
    }

    /// Whether `t`, and so every later value of an ascending index, lies past the range.
    pub fn is_past(&self, t: i64) -> bool {
        self.end.is_some_and(|end| t >= end)
    }

    /// The rows of the ascending `values` that lie in the range.
    pub fn row_range(&self, values: &[i64]) -> std::ops::Range<usize> {
        let start = values.partition_point(|&v| self.is_before(v));
        let end = values.partition_point(|&v| !self.is_past(v));
        start..end.max(start)
    }
}

/// Where the window grid of a dynamic group-by is placed.
///
/// Set by a planner on the IR, never by the DSL.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DynamicWindowPlacement {
    /// Start of the first window. `start_by` and `offset` are not used.
    pub origin: i64,
    /// Only windows whose start lies in this range are emitted.
    pub start_range: IndexRange,
}

/// Which rows of a rolling group-by produce an output row.
///
/// Set by a planner on the IR, never by the DSL. Rows outside the range still take part in
/// the windows of the rows inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollingWindowPlacement {
    /// Only rows whose index value lies in this range get a window.
    pub owned_range: IndexRange,
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

/// [`DynamicGroupOptions`] as the IR carries them: the same fields plus the window placement
/// a planner may set. The DSL never sets `placement`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DynamicGroupOptionsIR {
    pub index_column: PlSmallStr,
    pub every: Duration,
    pub period: Duration,
    pub offset: Duration,
    pub label: Label,
    pub include_boundaries: bool,
    pub closed_window: ClosedWindow,
    pub start_by: StartBy,
    /// Where the window grid is placed. `None` anchors on the first row.
    ///
    /// Only supported without group_by keys: with keys, every group anchors its own grid on its
    /// own first row, which a single origin cannot reproduce.
    pub placement: Option<DynamicWindowPlacement>,
}

impl From<DynamicGroupOptions> for DynamicGroupOptionsIR {
    fn from(options: DynamicGroupOptions) -> Self {
        let DynamicGroupOptions {
            index_column,
            every,
            period,
            offset,
            label,
            include_boundaries,
            closed_window,
            start_by,
        } = options;
        Self {
            index_column,
            every,
            period,
            offset,
            label,
            include_boundaries,
            closed_window,
            start_by,
            placement: None,
        }
    }
}

impl Default for DynamicGroupOptionsIR {
    fn default() -> Self {
        DynamicGroupOptions::default().into()
    }
}

/// [`RollingGroupOptions`] as the IR carries them: the same fields plus the window placement
/// a planner may set. The DSL never sets `placement`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollingGroupOptionsIR {
    pub index_column: PlSmallStr,
    pub period: Duration,
    pub offset: Duration,
    pub closed_window: ClosedWindow,
    /// Which rows get a window. `None` gives every row one.
    ///
    /// Only supported without group_by keys.
    pub placement: Option<RollingWindowPlacement>,
}

impl From<RollingGroupOptions> for RollingGroupOptionsIR {
    fn from(options: RollingGroupOptions) -> Self {
        let RollingGroupOptions {
            index_column,
            period,
            offset,
            closed_window,
        } = options;
        Self {
            index_column,
            period,
            offset,
            closed_window,
            placement: None,
        }
    }
}

impl Default for RollingGroupOptionsIR {
    fn default() -> Self {
        RollingGroupOptions::default().into()
    }
}
