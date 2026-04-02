use std::fmt;
use std::str::FromStr;

#[repr(u8)]
#[derive(Clone, Debug, Copy, Default, Eq, PartialEq, Hash)]
pub enum SpillPolicy {
    #[default]
    NoSpill = 0,
    Spill = 1,
}

impl fmt::Display for SpillPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_static_str())
    }
}

impl FromStr for SpillPolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "no_spill" => Ok(Self::NoSpill),
            "spill" => Ok(Self::Spill),
            v => Err(format!(
                "`spill_policy` must be one of {{'no_spill', 'spill'}}, got {v}",
            )),
        }
    }
}

impl SpillPolicy {
    pub(crate) fn from_discriminant(d: u8) -> Self {
        match d {
            0 => Self::NoSpill,
            1 => Self::Spill,
            _ => unreachable!(),
        }
    }

    pub fn as_static_str(&self) -> &'static str {
        match self {
            Self::NoSpill => "no_spill",
            Self::Spill => "spill",
        }
    }
}
