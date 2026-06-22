use std::fmt;
use std::str::FromStr;

#[repr(u8)]
#[derive(Clone, Debug, Copy, Default, Eq, PartialEq, Hash)]
pub enum ResolveMode {
    #[default]
    None = 0,
    RowCounts = 1,
    Full = 2,
}

impl fmt::Display for ResolveMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_static_str())
    }
}

impl FromStr for ResolveMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "row_counts" => Ok(Self::RowCounts),
            "full" => Ok(Self::Full),
            v => Err(format!(
                "`resolve_metadata_level` must be one of {{'none', 'row_counts', 'full'}}, got {v}",
            )),
        }
    }
}

impl ResolveMode {
    pub(crate) fn from_discriminant(d: u8) -> Self {
        match d {
            0 => Self::None,
            1 => Self::RowCounts,
            2 => Self::Full,
            _ => unreachable!(),
        }
    }

    pub fn as_static_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::RowCounts => "row_counts",
            Self::Full => "full",
        }
    }
}
