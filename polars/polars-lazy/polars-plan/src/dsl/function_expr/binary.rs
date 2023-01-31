#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum BinaryFunction {
    Contains { pat: Vec<u8>, literal: bool },
    StartsWith(Vec<u8>),
    EndsWith(Vec<u8>),
}

impl Display for BinaryFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use self::*;
        let s = match self {
            BinaryFunction::Contains { .. } => "contains",
            BinaryFunction::StartsWith(_) => "starts_with",
            BinaryFunction::EndsWith(_) => "ends_with",
        };

        write!(f, "str.{s}")
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

pub(super) fn ends_with(s: &Series, sub: &[u8]) -> PolarsResult<Series> {
    let ca = s.binary()?;
    Ok(ca.ends_with(sub).into_series())
}
pub(super) fn starts_with(s: &Series, sub: &[u8]) -> PolarsResult<Series> {
    let ca = s.binary()?;
    Ok(ca.starts_with(sub).into_series())
}

impl From<BinaryFunction> for FunctionExpr {
    fn from(b: BinaryFunction) -> Self {
        FunctionExpr::BinaryExpr(b)
    }
}
