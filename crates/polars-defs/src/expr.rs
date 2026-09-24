//! Option and enum definitions for Series-level operations.

use std::fmt::{Display, Formatter};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[strum(serialize_all = "snake_case")]
pub enum SetOperation {
    Intersection,
    Union,
    Difference,
    SymmetricDifference,
}

impl Display for SetOperation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            SetOperation::Intersection => "intersection",
            SetOperation::Union => "union",
            SetOperation::Difference => "difference",
            SetOperation::SymmetricDifference => "symmetric_difference",
        };
        write!(f, "{s}")
    }
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum UnicodeForm {
    NFC,
    NFKC,
    NFD,
    NFKD,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, strum_macros::IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum InterpolationMethod {
    Linear,
    Nearest,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum Roll {
    Forward,
    Backward,
    Raise,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[strum(serialize_all = "snake_case")]
#[derive(Default)]
pub enum RoundMode {
    #[default]
    HalfToEven,
    HalfAwayFromZero,
    ToZero,
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Default, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[strum(serialize_all = "snake_case")]
pub enum ClosedInterval {
    #[default]
    Both,
    Left,
    Right,
    None,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum RankMethod {
    Average,
    Min,
    Max,
    Dense,
    Ordinal,
    #[cfg(feature = "random")]
    Random,
}

// We might want to add a `nulls_last` or `null_behavior` field.
#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct RankOptions {
    pub method: RankMethod,
    pub descending: bool,
}

impl Default for RankOptions {
    fn default() -> Self {
        Self {
            method: RankMethod::Dense,
            descending: false,
        }
    }
}
