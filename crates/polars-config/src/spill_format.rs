use std::fmt;
use std::str::FromStr;

#[repr(u8)]
#[derive(Clone, Debug, Copy, Default, Eq, PartialEq, Hash)]
pub enum SpillFormat {
    #[default]
    Ipc = 0,
}

impl fmt::Display for SpillFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_static_str())
    }
}

impl FromStr for SpillFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ipc" => Ok(Self::Ipc),
            v => Err(format!("`spill_format` must be one of {{'ipc'}}, got {v}",)),
        }
    }
}

impl SpillFormat {
    pub(crate) fn from_discriminant(d: u8) -> Self {
        match d {
            0 => Self::Ipc,
            _ => unreachable!(),
        }
    }

    pub fn as_static_str(&self) -> &'static str {
        match self {
            Self::Ipc => "ipc",
        }
    }
}
