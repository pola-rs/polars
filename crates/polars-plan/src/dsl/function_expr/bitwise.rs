use std::fmt;

use strum_macros::IntoStaticStr;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum BitwiseFunction {
    CountOnes,
    CountZeros,

    LeadingOnes,
    LeadingZeros,

    TrailingOnes,
    TrailingZeros,

    // Bitwise Aggregations
    And,
    Or,
    Xor,
}

impl fmt::Display for BitwiseFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        use BitwiseFunction as B;

        let s = match self {
            B::CountOnes => "count_ones",
            B::CountZeros => "count_zeros",
            B::LeadingOnes => "leading_ones",
            B::LeadingZeros => "leading_zeros",
            B::TrailingOnes => "trailing_ones",
            B::TrailingZeros => "trailing_zeros",

            B::And => "and",
            B::Or => "or",
            B::Xor => "xor",
        };

        f.write_str(s)
    }
}
