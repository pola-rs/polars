use std::fmt;

use polars_core::prelude::SortOptions;

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum ArrayFunction {
    Length,
    Min,
    Max,
    Sum,
    ToList,
    Unique(bool),
    NUnique,
    Std(u8),
    Var(u8),
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
}

impl fmt::Display for ArrayFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        use ArrayFunction::*;
        let name = match self {
            Concat => "concat",
            Length => "length",
            Min => "min",
            Max => "max",
            Sum => "sum",
            ToList => "to_list",
            Unique(_) => "unique",
            NUnique => "n_unique",
            Std(_) => "std",
            Var(_) => "var",
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
        };
        write!(f, "arr.{name}")
    }
}
