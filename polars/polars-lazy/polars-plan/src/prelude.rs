#[cfg(feature = "rolling_window")]
pub(crate) use polars_time::chunkedarray::{RollingOptions, RollingOptionsImpl};
#[cfg(feature = "temporal")]
pub(crate) use polars_time::in_nanoseconds_window;
#[cfg(feature = "dynamic_groupby")]
pub(crate) use polars_time::{DynamicGroupOptions, PolarsTemporalGroupby, RollingGroupOptions};
#[cfg(not(feature = "dynamic_groupby"))]
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "dynamic_groupby"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DynamicGroupOptions {
    pub index_column: String,
}

#[cfg(any(
    feature = "temporal",
    feature = "dtype-duration",
    feature = "dtype-date",
    feature = "dtype-date",
    feature = "dtype-time"
))]
pub(crate) use polars_time::prelude::*;

#[cfg(not(feature = "dynamic_groupby"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RollingGroupOptions {
    pub index_column: String,
}

pub(crate) use polars_ops::prelude::*;
pub use polars_utils::arena::{Arena, Node};

pub use crate::dsl::*;
pub use crate::frame::*;
pub(crate) use crate::logical_plan::aexpr::*;
pub(crate) use crate::logical_plan::alp::*;
pub(crate) use crate::logical_plan::conversion::*;
pub(crate) use crate::logical_plan::iterator::*;
pub use crate::logical_plan::options::*;
pub use crate::logical_plan::*;
pub use crate::utils::*;
