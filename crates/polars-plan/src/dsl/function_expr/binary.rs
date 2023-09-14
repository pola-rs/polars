#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum BinaryFunction {
    Contains { pat: Vec<u8>, literal: bool },
    StartsWith,
    EndsWith,
}

impl Display for BinaryFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use BinaryFunction::*;
        let s = match self {
            Contains { .. } => "contains",
            StartsWith => "starts_with",
            EndsWith => "ends_with",
        };
        write!(f, "bin.{s}")
    }
}

pub(super) fn contains(s: &Series, pat: &[u8], literal: bool) -> PolarsResult<Series> {
    let ca = s.binary()?;
    if literal {
        ca.contains_literal(pat).map(|ca| ca.into_series())
    } else {
        ca.contains(pat).map(|ca| ca.into_series())
    }
}

pub(super) fn ends_with(s: &[Series]) -> PolarsResult<Series> {
    let ca = s[0].binary()?;
    let suffix = s[1].binary()?;

    Ok(ca
        .ends_with_chunked(suffix)
        .with_name(ca.name())
        .into_series())
}
pub(super) fn starts_with(s: &[Series]) -> PolarsResult<Series> {
    let ca = s[0].binary()?;
    let prefix = s[1].binary()?;

    Ok(ca
        .starts_with_chunked(prefix)
        .with_name(ca.name())
        .into_series())
}

impl From<BinaryFunction> for FunctionExpr {
    fn from(b: BinaryFunction) -> Self {
        FunctionExpr::BinaryExpr(b)
    }
}
