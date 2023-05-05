use std::fmt::{Display, Formatter};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum FusedOperator {
    MultiplyAdd,
}

impl Display for FusedOperator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            FusedOperator::MultiplyAdd => "fma",
        };
        write!(f, "{s}")
    }
}

pub(super) fn fused(input: &[Series], op: FusedOperator) -> PolarsResult<Series> {
    match op {
        FusedOperator::MultiplyAdd => Ok(fma_series(&input[0], &input[1], &input[2])),
    }
}
