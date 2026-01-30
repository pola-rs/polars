use serde::{Deserialize, Serialize};

// a boolean that can only be set to `true` safely
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct UnsafeBool(pub bool);
impl Default for UnsafeBool {
    fn default() -> Self {
        UnsafeBool(true)
    }
}
