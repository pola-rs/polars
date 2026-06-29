use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, PartialEq, Debug, Hash)]
pub enum IRCorrelationMethod {
    Pearson,
    #[cfg(all(feature = "rank", feature = "propagate_nans"))]
    SpearmanRank(bool),
    Covariance(u8),
}

impl Display for IRCorrelationMethod {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRCorrelationMethod::*;
        let s = match self {
            Pearson => "pearson",
            #[cfg(all(feature = "rank", feature = "propagate_nans"))]
            SpearmanRank(_) => "spearman_rank",
            Covariance(_) => return write!(f, "covariance"),
        };
        write!(f, "{s}_correlation")
    }
}
