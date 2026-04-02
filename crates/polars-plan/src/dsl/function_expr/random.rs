#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Copy, Clone, PartialEq, Debug, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum RandomMethod {
    Shuffle,
    Sample {
        is_fraction: bool,
        with_replacement: bool,
        shuffle: bool,
    },
}

impl Hash for RandomMethod {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state)
    }
}
