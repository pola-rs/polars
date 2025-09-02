use uuid::Uuid;

/// Unique identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct UniqueId(Uuid);

impl UniqueId {
    #[expect(clippy::new_without_default)]
    #[inline]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn as_u128(&self) -> u128 {
        self.0.as_u128()
    }
}

impl std::fmt::Display for UniqueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.as_hyphenated())
    }
}
