use std::hash::Hash;
use std::sync::Arc;

use polars_core::error::PolarsResult;
#[cfg(feature = "iejoin")]
use polars_ops::frame::IEJoinOptions;
use polars_ops::frame::{CrossJoinFilter, CrossJoinOptions, JoinTypeOptions};
use polars_ops::prelude::{JoinArgs, JoinType};
#[cfg(feature = "dynamic_group_by")]
use polars_time::RollingGroupOptions;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::IdxSize;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

use super::ExprIR;
use crate::dsl::Selector;

#[derive(Copy, Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollingCovOptions {
    pub window_size: IdxSize,
    pub min_periods: IdxSize,
    pub ddof: u8,
}

#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StrptimeOptions {
    /// Formatting string
    pub format: Option<PlSmallStr>,
    /// If set then polars will return an error if any date parsing fails
    pub strict: bool,
    /// If polars may parse matches that not contain the whole string
    /// e.g. "foo-2021-01-01-bar" could match "2021-01-01"
    pub exact: bool,
    /// use a cache of unique, converted dates to apply the datetime conversion.
    pub cache: bool,
}

impl Default for StrptimeOptions {
    fn default() -> Self {
        StrptimeOptions {
            format: None,
            strict: true,
            exact: true,
            cache: true,
        }
    }
}

#[derive(Clone, PartialEq, Eq, IntoStaticStr, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[strum(serialize_all = "snake_case")]
pub enum JoinTypeOptionsIR {
    #[cfg(feature = "iejoin")]
    IEJoin(IEJoinOptions),
    #[cfg_attr(feature = "serde", serde(skip))]
    // Fused cross join and filter (only in in-memory engine)
    Cross { predicate: ExprIR },
}

impl Hash for JoinTypeOptionsIR {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use JoinTypeOptionsIR::*;
        match self {
            #[cfg(feature = "iejoin")]
            IEJoin(opt) => opt.hash(state),
            Cross { predicate } => predicate.node().hash(state),
        }
    }
}

impl JoinTypeOptionsIR {
    pub fn compile<C: FnOnce(&ExprIR) -> PolarsResult<Arc<dyn CrossJoinFilter>>>(
        self,
        plan: C,
    ) -> PolarsResult<JoinTypeOptions> {
        use JoinTypeOptionsIR::*;
        match self {
            Cross { predicate } => {
                let predicate = plan(&predicate)?;

                Ok(JoinTypeOptions::Cross(CrossJoinOptions { predicate }))
            },
            #[cfg(feature = "iejoin")]
            IEJoin(opt) => Ok(JoinTypeOptions::IEJoin(opt)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct JoinOptions {
    pub allow_parallel: bool,
    pub force_parallel: bool,
    pub args: JoinArgs,
    pub options: Option<JoinTypeOptionsIR>,
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
            // Todo!: make default
            args: JoinArgs::new(JoinType::Left),
            options: Default::default(),
            rows_left: (None, usize::MAX),
            rows_right: (None, usize::MAX),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum WindowType {
    /// Explode the aggregated list and just do a hstack instead of a join
    /// this requires the groups to be sorted to make any sense
    Over(WindowMapping),
    #[cfg(feature = "dynamic_group_by")]
    Rolling(RollingGroupOptions),
}

impl From<WindowMapping> for WindowType {
    fn from(value: WindowMapping) -> Self {
        Self::Over(value)
    }
}

impl Default for WindowType {
    fn default() -> Self {
        Self::Over(WindowMapping::default())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[strum(serialize_all = "snake_case")]
pub enum WindowMapping {
    /// Map the group values to the position
    #[default]
    GroupsToRows,
    /// Explode the aggregated list and just do a hstack instead of a join
    /// this requires the groups to be sorted to make any sense
    Explode,
    /// Join the groups as 'List<group_dtype>' to the row positions.
    /// warning: this can be memory intensive
    Join,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NestedType {
    #[cfg(feature = "dtype-array")]
    Array,
    // List,
}

#[derive(Clone, Default, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UnpivotArgsDSL {
    pub on: Vec<Selector>,
    pub index: Vec<Selector>,
    pub variable_name: Option<PlSmallStr>,
    pub value_name: Option<PlSmallStr>,
}
