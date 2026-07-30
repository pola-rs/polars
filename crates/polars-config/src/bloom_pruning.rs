use std::fmt;
use std::str::FromStr;

/// Bloom-filter row-group pruning mode: disabled, or enabled with a filter-read strategy.
#[repr(u8)]
#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
pub enum BloomPruning {
    /// Pruning disabled.
    Off = 0,
    /// Pruning where the engine expects it to pay off.
    Auto = 1,
    /// Pruning enabled; always fetch the whole filter in one request.
    Whole = 2,
    /// Pruning enabled; always fetch the header, then only the block(s) the hashes map to.
    Blocks = 3,
}

impl fmt::Display for BloomPruning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_static_str())
    }
}

impl FromStr for BloomPruning {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" | "false" | "off" => Ok(Self::Off),
            "1" | "true" | "auto" => Ok(Self::Auto),
            "whole" => Ok(Self::Whole),
            "blocks" => Ok(Self::Blocks),
            v => Err(format!(
                "`bloom_filter_prune` must be one of {{'0', 'false', 'off', '1', 'true', \
                 'auto', 'whole', 'blocks'}}, got {v}",
            )),
        }
    }
}

impl BloomPruning {
    pub(crate) fn from_discriminant(d: u8) -> Self {
        match d {
            0 => Self::Off,
            1 => Self::Auto,
            2 => Self::Whole,
            3 => Self::Blocks,
            _ => unreachable!(),
        }
    }

    pub fn as_static_str(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::Whole => "whole",
            Self::Blocks => "blocks",
        }
    }
}
