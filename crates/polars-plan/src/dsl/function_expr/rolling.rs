use super::*;

#[derive(Clone, PartialEq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum RollingFunction {
    Min,
    Max,
    Mean,
    Sum,
    Quantile,
    Var,
    Std,
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
}

impl Display for RollingFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use RollingFunction::*;

        let name = match self {
            Min => "min",
            Max => "max",
            Mean => "mean",
            Sum => "rsum",
            Quantile => "quantile",
            Var => "var",
            Std => "std",
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
        };

        write!(f, "rolling_{name}")
    }
}
