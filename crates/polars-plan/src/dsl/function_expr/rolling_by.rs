use polars_time::chunkedarray::*;

use super::*;

#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RollingFunctionBy {
    MinBy(RollingOptionsDynamicWindow),
    MaxBy(RollingOptionsDynamicWindow),
    MeanBy(RollingOptionsDynamicWindow),
    SumBy(RollingOptionsDynamicWindow),
    QuantileBy(RollingOptionsDynamicWindow),
    VarBy(RollingOptionsDynamicWindow),
    StdBy(RollingOptionsDynamicWindow),
}

impl Display for RollingFunctionBy {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use RollingFunctionBy::*;

        let name = match self {
            MinBy(_) => "rolling_min_by",
            MaxBy(_) => "rolling_max_by",
            MeanBy(_) => "rolling_mean_by",
            SumBy(_) => "rolling_sum_by",
            QuantileBy(_) => "rolling_quantile_by",
            VarBy(_) => "rolling_var_by",
            StdBy(_) => "rolling_std_by",
        };

        write!(f, "{name}")
    }
}

impl Hash for RollingFunctionBy {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
    }
}

pub(super) fn rolling_min_by(
    s: &[Series],
    options: RollingOptionsDynamicWindow,
) -> PolarsResult<Series> {
    s[0].rolling_min_by(&s[1], options)
}

pub(super) fn rolling_max_by(
    s: &[Series],
    options: RollingOptionsDynamicWindow,
) -> PolarsResult<Series> {
    s[0].rolling_max_by(&s[1], options)
}

pub(super) fn rolling_mean_by(
    s: &[Series],
    options: RollingOptionsDynamicWindow,
) -> PolarsResult<Series> {
    s[0].rolling_mean_by(&s[1], options)
}

pub(super) fn rolling_sum_by(
    s: &[Series],
    options: RollingOptionsDynamicWindow,
) -> PolarsResult<Series> {
    s[0].rolling_sum_by(&s[1], options)
}

pub(super) fn rolling_quantile_by(
    s: &[Series],
    options: RollingOptionsDynamicWindow,
) -> PolarsResult<Series> {
    s[0].rolling_quantile_by(&s[1], options)
}

pub(super) fn rolling_var_by(
    s: &[Series],
    options: RollingOptionsDynamicWindow,
) -> PolarsResult<Series> {
    s[0].rolling_var_by(&s[1], options)
}

pub(super) fn rolling_std_by(
    s: &[Series],
    options: RollingOptionsDynamicWindow,
) -> PolarsResult<Series> {
    s[0].rolling_std_by(&s[1], options)
}
