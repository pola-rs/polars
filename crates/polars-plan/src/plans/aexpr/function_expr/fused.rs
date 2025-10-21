use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
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
