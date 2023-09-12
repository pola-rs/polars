use polars_core::prelude::arity::binary_elementwise;
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
    let sub = s[1].binary()?;

    Ok(match sub.len() {
        1 => match sub.get(0) {
            Some(s) => ca.ends_with(s),
            None => BooleanChunked::full(ca.name(), false, ca.len()),
        },
        _ => binary_elementwise(ca, sub, |opt_s, opt_sub| match (opt_s, opt_sub) {
            (Some(s), Some(sub)) => Some(s.ends_with(sub)),
            _ => Some(false),
        }),
    }
    .with_name(ca.name())
    .into_series())
}
pub(super) fn starts_with(s: &[Series]) -> PolarsResult<Series> {
    let ca = s[0].binary()?;
    let sub = s[1].binary()?;

    Ok(match sub.len() {
        1 => match sub.get(0) {
            Some(s) => ca.starts_with(s),
            None => BooleanChunked::full(ca.name(), false, ca.len()),
        },
        _ => binary_elementwise(ca, sub, |opt_s, opt_sub| match (opt_s, opt_sub) {
            (Some(s), Some(sub)) => Some(s.starts_with(sub)),
            _ => Some(false),
        }),
    }
    .with_name(ca.name())
    .into_series())
}

impl From<BinaryFunction> for FunctionExpr {
    fn from(b: BinaryFunction) -> Self {
        FunctionExpr::BinaryExpr(b)
    }
}
