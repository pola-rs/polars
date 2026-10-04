use std::fmt;
use std::str::FromStr;

/// Access pattern hint applied to a file when it is opened for reading.
#[repr(u8)]
#[derive(Clone, Debug, Copy, Default, Eq, PartialEq, Hash)]
pub enum FileAdvice {
    /// Leave the OS default in place.
    Normal = 0,
    /// Double read-ahead.
    Sequential = 1,
    /// Suppresses read-ahead.
    #[default]
    Random = 2,
    /// For prefetch only.
    WillNeed = 3,
}

impl fmt::Display for FileAdvice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_static_str())
    }
}

impl FromStr for FileAdvice {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "normal" => Ok(Self::Normal),
            "sequential" => Ok(Self::Sequential),
            "random" => Ok(Self::Random),
            "willneed" => Ok(Self::WillNeed),
            v => Err(format!(
                "`file_advice` must be one of \
                {{'normal', 'sequential', 'random', 'willneed'}}, got {v}",
            )),
        }
    }
}

impl FileAdvice {
    pub(crate) fn from_discriminant(d: u8) -> Self {
        match d {
            0 => Self::Normal,
            1 => Self::Sequential,
            2 => Self::Random,
            3 => Self::WillNeed,
            _ => unreachable!(),
        }
    }

    pub fn as_static_str(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Sequential => "sequential",
            Self::Random => "random",
            Self::WillNeed => "willneed",
        }
    }
}
