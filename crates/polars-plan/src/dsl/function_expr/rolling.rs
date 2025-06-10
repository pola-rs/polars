use super::*;

#[derive(Clone, PartialEq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum RollingFunction {
    Min(RollingOptionsFixedWindow),
    Max(RollingOptionsFixedWindow),
    Mean(RollingOptionsFixedWindow),
    Sum(RollingOptionsFixedWindow),
    Quantile(RollingOptionsFixedWindow),
    Var(RollingOptionsFixedWindow),
    Std(RollingOptionsFixedWindow),
    #[cfg(feature = "moment")]
    Skew(RollingOptionsFixedWindow),
    #[cfg(feature = "moment")]
    Kurtosis(RollingOptionsFixedWindow),
    #[cfg(feature = "cov")]
    CorrCov {
        rolling_options: RollingOptionsFixedWindow,
        corr_cov_options: RollingCovOptions,
        // Whether is Corr or Cov
        is_corr: bool,
    },
}

impl Display for RollingFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use RollingFunction::*;

        let name = match self {
            Min(_) => "min",
            Max(_) => "max",
            Mean(_) => "mean",
            Sum(_) => "rsum",
            Quantile(_) => "quantile",
            Var(_) => "var",
            Std(_) => "std",
            #[cfg(feature = "moment")]
            Skew(..) => "skew",
            #[cfg(feature = "moment")]
            Kurtosis(..) => "kurtosis",
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
