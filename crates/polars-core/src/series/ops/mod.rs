mod canonicalize_maps;
mod downcast;
mod extend;
mod from_physical;
pub mod int_range;
mod json_map;
mod null;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
mod reshape;

#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum NullBehavior {
    /// drop nulls
    Drop,
    /// ignore nulls
    #[default]
    Ignore,
}
