use polars_ops::chunked_array::array::*;

use super::*;

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ArrayFunction {
    Min,
    Max,
    Sum,
}

impl Display for ArrayFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use ArrayFunction::*;
        let name = match self {
            Min => "min",
            Max => "max",
            Sum => "sum",
        };

        write!(f, "arr.{name}")
    }
}

pub(super) fn max(s: &Series) -> PolarsResult<Series> {
    Ok(s.array()?.array_max())
}

pub(super) fn min(s: &Series) -> PolarsResult<Series> {
    Ok(s.array()?.array_min())
}

pub(super) fn sum(s: &Series) -> PolarsResult<Series> {
    s.array()?.array_sum()
}
