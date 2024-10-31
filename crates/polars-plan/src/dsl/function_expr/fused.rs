#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, PartialEq, Debug, Hash)]
pub enum FusedOperator {
    MultiplyAdd,
    SubMultiply,
    MultiplySub,
}

impl Display for FusedOperator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            FusedOperator::MultiplyAdd => "fma",
            FusedOperator::SubMultiply => "fsm",
            FusedOperator::MultiplySub => "fms",
        };
        write!(f, "{s}")
    }
}

pub(super) fn fused(input: &[Column], op: FusedOperator) -> PolarsResult<Column> {
    let s0 = &input[0];
    let s1 = &input[1];
    let s2 = &input[2];
    match op {
        FusedOperator::MultiplyAdd => Ok(fma_columns(s0, s1, s2)),
        FusedOperator::SubMultiply => Ok(fsm_columns(s0, s1, s2)),
        FusedOperator::MultiplySub => Ok(fms_columns(s0, s1, s2)),
    }
}
