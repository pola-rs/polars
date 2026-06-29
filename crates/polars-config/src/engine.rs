use std::fmt;
use std::str::FromStr;

#[repr(u8)]
#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Engine {
    Auto = 0,
    Streaming = 1,
    InMemory = 2,
    Gpu = 3,
}

impl fmt::Display for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_static_str())
    }
}

impl FromStr for Engine {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Engine::Auto),
            // We keep "cpu" for backwards compatibility.
            "cpu" | "in-memory" => Ok(Engine::InMemory),
            "streaming" => Ok(Engine::Streaming),
            "gpu" => Ok(Engine::Gpu),
            "old-streaming" => Err("the 'old-streaming' engine has been removed".to_owned()),
            v => Err(format!(
                "`engine` must be one of {{'auto', 'in-memory', 'streaming', 'gpu'}}, got {v}",
            )),
        }
    }
}

impl Engine {
    pub(crate) fn from_discriminant(d: u8) -> Self {
        match d {
            0 => Self::Auto,
            1 => Self::Streaming,
            2 => Self::InMemory,
            3 => Self::Gpu,
            _ => unreachable!(),
        }
    }

    pub fn as_static_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Streaming => "streaming",
            Self::InMemory => "in-memory",
            Self::Gpu => "gpu",
        }
    }
}
