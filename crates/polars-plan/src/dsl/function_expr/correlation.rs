#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Copy, Clone, PartialEq, Debug, Hash)]
pub enum CorrelationMethod {
    Pearson,
    #[cfg(all(feature = "rank", feature = "propagate_nans"))]
    SpearmanRank(bool),
    Covariance(u8),
}

impl Display for CorrelationMethod {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use CorrelationMethod::*;
        let s = match self {
            Pearson => "pearson",
            #[cfg(all(feature = "rank", feature = "propagate_nans"))]
            SpearmanRank(_) => "spearman_rank",
            Covariance(_) => return write!(f, "covariance"),
        };
        write!(f, "{s}_correlation")
    }
}
