use bitflags::bitflags;
use polars_core::prelude::*;
use polars_core::utils::SuperTypeOptions;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::plans::PlSmallStr;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct DistinctOptionsIR {
    /// Subset of columns that will be taken into account.
    pub subset: Option<Arc<[PlSmallStr]>>,
    /// This will maintain the order of the input.
    /// Note that this is more expensive.
    /// `maintain_order` is not supported in the streaming
    /// engine.
    pub maintain_order: bool,
    /// Which rows to keep.
    pub keep_strategy: UniqueKeepStrategy,
    /// Take only a slice of the result
    pub slice: Option<(i64, usize)>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ApplyOptions {
    /// Collect groups to a list and apply the function over the groups.
    /// This can be important in aggregation context.
    /// e.g. [g1, g1, g2] -> [[g1, g1], g2]
    GroupWise,
    /// collect groups to a list and then apply
    /// e.g. [g1, g1, g2] -> list([g1, g1, g2])
    ApplyList,
    /// do not collect before apply
    /// e.g. [g1, g1, g2] -> [g1, g1, g2]
    ElementWise,
}

// a boolean that can only be set to `false` safely
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UnsafeBool(bool);
impl Default for UnsafeBool {
    fn default() -> Self {
        UnsafeBool(true)
    }
}

bitflags!(
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
        pub struct FunctionFlags: u8 {
            // Raise if use in group by
            const ALLOW_GROUP_AWARE = 1 << 0;
            // For example a `unique` or a `slice`
            const CHANGES_LENGTH = 1 << 1;
            // The physical expression may rename the output of this function.
            // If set to `false` the physical engine will ensure the left input
            // expression is the output name.
            const ALLOW_RENAME = 1 << 2;
            // if set, then the `Series` passed to the function in the group_by operation
            // will ensure the name is set. This is an extra heap allocation per group.
            const PASS_NAME_TO_APPLY = 1 << 3;
            /// There can be two ways of expanding wildcards:
            ///
            /// Say the schema is 'a', 'b' and there is a function `f`. In this case, `f('*')` can expand
            /// to:
            /// 1. `f('a', 'b')`
            /// 2. `f('a'), f('b')`
            ///
            /// Setting this to true, will lead to behavior 1.
            ///
            /// This also accounts for regex expansion.
            const INPUT_WILDCARD_EXPANSION = 1 << 4;
            /// Automatically explode on unit length if it ran as final aggregation.
            ///
            /// this is the case for aggregations like sum, min, covariance etc.
            /// We need to know this because we cannot see the difference between
            /// the following functions based on the output type and number of elements:
            ///
            /// x: {1, 2, 3}
            ///
            /// head_1(x) -> {1}
            /// sum(x) -> {4}
            const RETURNS_SCALAR = 1 << 5;
            /// This can happen with UDF's that use Polars within the UDF.
            /// This can lead to recursively entering the engine and sometimes deadlocks.
            /// This flag must be set to handle that.
            const OPTIONAL_RE_ENTRANT = 1 << 6;
            /// Whether this function allows no inputs.
            const ALLOW_EMPTY_INPUTS = 1 << 7;
        }
);

impl Default for FunctionFlags {
    fn default() -> Self {
        Self::from_bits_truncate(0) | Self::ALLOW_GROUP_AWARE
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum CastingRules {
    /// Whether information may be lost during cast. E.g. a float to int is considered lossy,
    /// whereas int to int is considered lossless.
    /// Overflowing is not considered in this flag, that's handled in `strict` casting
    FirstArgLossless,
    Supertype(SuperTypeOptions),
}

impl CastingRules {
    pub fn cast_to_supertypes() -> CastingRules {
        Self::Supertype(Default::default())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(any(feature = "serde"), derive(Serialize, Deserialize))]
pub struct FunctionOptions {
    /// Collect groups to a list and apply the function over the groups.
    /// This can be important in aggregation context.
    pub collect_groups: ApplyOptions,

    // Validate the output of a `map`.
    // this should always be true or we could OOB
    pub check_lengths: UnsafeBool,
    pub flags: FunctionFlags,

    // used for formatting, (only for anonymous functions)
    #[cfg_attr(feature = "serde", serde(skip))]
    pub fmt_str: &'static str,
    /// Options used when deciding how to cast the arguments of the function.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub cast_options: Option<CastingRules>,
}

impl FunctionOptions {
    #[cfg(feature = "fused")]
    pub(crate) unsafe fn no_check_lengths(&mut self) {
        self.check_lengths = UnsafeBool(false);
    }
    pub fn check_lengths(&self) -> bool {
        self.check_lengths.0
    }

    pub fn set_elementwise(&mut self) {
        self.collect_groups = ApplyOptions::ElementWise
    }

    pub fn is_elementwise(&self) -> bool {
        matches!(
            self.collect_groups,
            ApplyOptions::ElementWise | ApplyOptions::ApplyList
        ) && !self.flags.contains(FunctionFlags::CHANGES_LENGTH)
            && !self.flags.contains(FunctionFlags::RETURNS_SCALAR)
    }

    pub fn is_length_preserving(&self) -> bool {
        !self.flags.contains(FunctionFlags::CHANGES_LENGTH)
    }

    pub fn returns_scalar(&self) -> bool {
        self.flags.contains(FunctionFlags::RETURNS_SCALAR)
    }
}

impl Default for FunctionOptions {
    fn default() -> Self {
        FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            check_lengths: UnsafeBool(true),
            fmt_str: Default::default(),
            cast_options: Default::default(),
            flags: Default::default(),
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProjectionOptions {
    pub run_parallel: bool,
    pub duplicate_check: bool,
    // Should length-1 Series be broadcast to the length of the dataframe.
    // Only used by CSE optimizer
    pub should_broadcast: bool,
}

impl Default for ProjectionOptions {
    fn default() -> Self {
        Self {
            run_parallel: true,
            duplicate_check: true,
            should_broadcast: true,
        }
    }
}

impl ProjectionOptions {
    /// Conservatively merge the options of two [`ProjectionOptions`]
    pub fn merge_options(&self, other: &Self) -> Self {
        Self {
            run_parallel: self.run_parallel & other.run_parallel,
            duplicate_check: self.duplicate_check & other.duplicate_check,
            should_broadcast: self.should_broadcast | other.should_broadcast,
        }
    }
}
