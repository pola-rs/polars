use std::ops::BitAnd;

use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Default, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[strum(serialize_all = "snake_case")]
pub enum ClosedInterval {
    #[default]
    Both,
    Left,
    Right,
    None,
}

pub fn is_between(
    s: &Series,
    lower: &Series,
    upper: &Series,
    closed: ClosedInterval,
) -> PolarsResult<BooleanChunked> {
    let left_cmp_op = match closed {
        ClosedInterval::None | ClosedInterval::Right => Series::gt,
        ClosedInterval::Both | ClosedInterval::Left => Series::gt_eq,
    };
    let right_cmp_op = match closed {
        ClosedInterval::None | ClosedInterval::Left => Series::lt,
        ClosedInterval::Both | ClosedInterval::Right => Series::lt_eq,
    };
    let left = left_cmp_op(s, lower)?;
    let right = right_cmp_op(s, upper)?;
    Ok(left.bitand(right))
}
