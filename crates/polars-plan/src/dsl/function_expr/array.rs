use std::fmt;

use polars_core::prelude::SortOptions;

use super::FunctionExpr;

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum ArrayFunction {
    Length,
    Slice(i64, i64),
    Min,
    Max,
    Sum,
    ToList,
    Unique(bool),
    NUnique,
    Std(u8),
    Var(u8),
    Mean,
    Median,
    #[cfg(feature = "array_any_all")]
    Any,
    #[cfg(feature = "array_any_all")]
    All,
    Sort(SortOptions),
    Reverse,
    ArgMin,
    ArgMax,
    Get(bool),
    Join(bool),
    #[cfg(feature = "is_in")]
    Contains {
        nulls_equal: bool,
    },
    #[cfg(feature = "array_count")]
    CountMatches,
    Shift,
    Explode {
        skip_empty: bool,
    },
    Concat,
    #[cfg(feature = "array_to_struct")]
    ToStruct(Option<super::DslNameGenerator>),
}

impl fmt::Display for ArrayFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        use ArrayFunction::*;
        let name = match self {
            Concat => "concat",
            Length => "length",
            Slice(_, _) => "slice",
            Min => "min",
            Max => "max",
            Sum => "sum",
            ToList => "to_list",
            Unique(_) => "unique",
            NUnique => "n_unique",
            Std(_) => "std",
            Var(_) => "var",
            Mean => "mean",
            Median => "median",
            #[cfg(feature = "array_any_all")]
            Any => "any",
            #[cfg(feature = "array_any_all")]
            All => "all",
            Sort(_) => "sort",
            Reverse => "reverse",
            ArgMin => "arg_min",
            ArgMax => "arg_max",
            Get(_) => "get",
            Join(_) => "join",
            #[cfg(feature = "is_in")]
            Contains { nulls_equal: _ } => "contains",
            #[cfg(feature = "array_count")]
            CountMatches => "count_matches",
            Shift => "shift",
            Explode { .. } => "explode",
            #[cfg(feature = "array_to_struct")]
            ToStruct(_) => "to_struct",
        };
        write!(f, "arr.{name}")
    }
}

impl From<ArrayFunction> for FunctionExpr {
    fn from(value: ArrayFunction) -> Self {
        Self::ArrayExpr(value)
    }
}
