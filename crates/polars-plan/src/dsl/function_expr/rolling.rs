use polars_time::chunkedarray::*;

use super::*;

#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RollingFunction {
    Min(RollingOptionsFixedWindow),
    Max(RollingOptionsFixedWindow),
    Mean(RollingOptionsFixedWindow),
    Sum(RollingOptionsFixedWindow),
    Quantile(RollingOptionsFixedWindow),
    Var(RollingOptionsFixedWindow),
    Std(RollingOptionsFixedWindow),
    #[cfg(feature = "moment")]
    Skew(usize, bool),
}

impl Display for RollingFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use RollingFunction::*;

        let name = match self {
            Min(_) => "rolling_min",
            Max(_) => "rolling_max",
            Mean(_) => "rolling_mean",
            Sum(_) => "rolling_sum",
            Quantile(_) => "rolling_quantile",
            Var(_) => "rolling_var",
            Std(_) => "rolling_std",
            #[cfg(feature = "moment")]
            Skew(..) => "rolling_skew",
        };

        write!(f, "{name}")
    }
}

impl Hash for RollingFunction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        use RollingFunction::*;

        std::mem::discriminant(self).hash(state);
        match self {
            #[cfg(feature = "moment")]
            Skew(window_size, bias) => {
                window_size.hash(state);
                bias.hash(state)
            },
            _ => {},
        }
    }
}

pub(super) fn rolling_min(s: &Series, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
    s.rolling_min(options)
}

pub(super) fn rolling_max(s: &Series, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
    s.rolling_max(options)
}

pub(super) fn rolling_mean(s: &Series, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
    s.rolling_mean(options)
}

pub(super) fn rolling_sum(s: &Series, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
    s.rolling_sum(options)
}

pub(super) fn rolling_quantile(
    s: &Series,
    options: RollingOptionsFixedWindow,
) -> PolarsResult<Series> {
    s.rolling_quantile(options)
}

pub(super) fn rolling_var(s: &Series, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
    s.rolling_var(options)
}

pub(super) fn rolling_std(s: &Series, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
    s.rolling_std(options)
}

#[cfg(feature = "moment")]
pub(super) fn rolling_skew(s: &Series, window_size: usize, bias: bool) -> PolarsResult<Series> {
    s.rolling_skew(window_size, bias)
}
