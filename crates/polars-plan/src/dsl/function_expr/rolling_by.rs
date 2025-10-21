use super::*;

#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum RollingFunctionBy {
    MinBy,
    MaxBy,
    MeanBy,
    SumBy,
    QuantileBy,
    VarBy,
    StdBy,
    RankBy,
}

impl Display for RollingFunctionBy {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use RollingFunctionBy::*;

        let name = match self {
            MinBy => "rolling_min_by",
            MaxBy => "rolling_max_by",
            MeanBy => "rolling_mean_by",
            SumBy => "rolling_sum_by",
            QuantileBy => "rolling_quantile_by",
            VarBy => "rolling_var_by",
            StdBy => "rolling_std_by",
            RankBy => "rolling_rank_by",
        };

        write!(f, "{name}")
    }
}

impl Hash for RollingFunctionBy {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
    }
}
