pub(crate) use polars_utils::arena::{Arena, Node};

#[cfg(feature = "temporal")]
pub(crate) use polars_time::in_nanoseconds_window;
#[cfg(feature = "dynamic_groupby")]
pub(crate) use polars_time::{DynamicGroupOptions, PolarsTemporalGroupby, RollingGroupOptions};

#[cfg(not(feature = "dynamic_groupby"))]
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "dynamic_groupby"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct DynamicGroupOptions {
    pub index_column: String,
}

pub(crate) use polars_time::prelude::*;

#[cfg(not(feature = "dynamic_groupby"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct RollingGroupOptions {
    pub index_column: String,
}

pub(crate) use polars_ops::prelude::*;

pub use crate::{
    dsl::*,
    frame::*,
    logical_plan::{
        optimizer::{type_coercion::TypeCoercionRule, Optimize, *},
        options::*,
        *,
    },
    physical_plan::{expressions::*, planner::DefaultPlanner, Executor, PhysicalPlanner},
};

pub(crate) use crate::{
    logical_plan::{aexpr::*, alp::*, conversion::*, iterator::*},
    utils::*,
};
