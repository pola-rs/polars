#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum UuidFunction {
    GenerateV4,
    GenerateV7,
    Version,
    #[cfg(feature = "dtype-datetime")]
    Timestamp {
        strict: bool,
    },
}

impl Display for UuidFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::GenerateV4 => "generate_v4",
            Self::GenerateV7 => "generate_v7",
            Self::Version => "version",
            #[cfg(feature = "dtype-datetime")]
            Self::Timestamp { .. } => "timestamp",
        };
        write!(f, "uuid.{name}")
    }
}
