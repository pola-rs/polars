use super::*;

#[derive(Clone, PartialEq, Debug, Hash)]
#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IRRollingFunction {
    Min,
    Max,
    Mean,
    Sum,
    Quantile,
    Var,
    Std,
    Rank,
    #[cfg(feature = "moment")]
    Skew,
    #[cfg(feature = "moment")]
    Kurtosis,
    #[cfg(feature = "cov")]
    CorrCov {
        corr_cov_options: RollingCovOptions,
        // Whether is Corr or Cov
        is_corr: bool,
    },
    Map(PlanCallback<Series, Series>),
}

impl Display for IRRollingFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRRollingFunction::*;

        let name = match self {
            Min => "min",
            Max => "max",
            Mean => "mean",
            Sum => "rsum",
            Quantile => "quantile",
            Var => "var",
            Std => "std",
            Rank => "rank",
            #[cfg(feature = "moment")]
            Skew => "skew",
            #[cfg(feature = "moment")]
            Kurtosis => "kurtosis",
            #[cfg(feature = "cov")]
            CorrCov { is_corr, .. } => {
                if *is_corr {
                    "corr"
                } else {
                    "cov"
                }
            },
            Map(_) => "map",
        };

        write!(f, "rolling_{name}")
    }
}
