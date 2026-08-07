use std::ops::Deref;

// a boolean flag that can only be set to `true` safely
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct UnsafeBool(bool);
impl Default for UnsafeBool {
    fn default() -> Self {
        UnsafeBool(true)
    }
}

impl UnsafeBool {
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn new_false() -> Self {
        UnsafeBool(false)
    }
}

impl Deref for UnsafeBool {
    type Target = bool;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<bool> for UnsafeBool {
    fn as_ref(&self) -> &bool {
        &self.0
    }
}
