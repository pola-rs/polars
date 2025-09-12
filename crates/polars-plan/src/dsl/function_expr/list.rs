use super::*;

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum ListFunction {
    Concat,
    #[cfg(feature = "is_in")]
    Contains {
        nulls_equal: bool,
    },
    #[cfg(feature = "list_drop_nulls")]
    DropNulls,
    #[cfg(feature = "list_sample")]
    Sample {
        is_fraction: bool,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    },
    Slice,
    Shift,
    Get(bool),
    #[cfg(feature = "list_gather")]
    Gather(bool),
    #[cfg(feature = "list_gather")]
    GatherEvery,
    #[cfg(feature = "list_count")]
    CountMatches,
    Sum,
    Length,
    Max,
    Min,
    Mean,
    Median,
    Std(u8),
    Var(u8),
    ArgMin,
    ArgMax,
    #[cfg(feature = "diff")]
    Diff {
        n: i64,
        null_behavior: NullBehavior,
    },
    Sort(SortOptions),
    Reverse,
    Unique(bool),
    NUnique,
    #[cfg(feature = "list_sets")]
    SetOperation(SetOperation),
    #[cfg(feature = "list_any_all")]
    Any,
    #[cfg(feature = "list_any_all")]
    All,
    Join(bool),
    #[cfg(feature = "dtype-array")]
    ToArray(usize),
    #[cfg(feature = "list_to_struct")]
    ToStruct(Arc<[PlSmallStr]>),
}

impl Display for ListFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use ListFunction::*;

        let name = match self {
            Concat => "concat",
            #[cfg(feature = "is_in")]
            Contains { nulls_equal: _ } => "contains",
            #[cfg(feature = "list_drop_nulls")]
            DropNulls => "drop_nulls",
            #[cfg(feature = "list_sample")]
            Sample { is_fraction, .. } => {
                if *is_fraction {
                    "sample_fraction"
                } else {
                    "sample_n"
                }
            },
            Slice => "slice",
            Shift => "shift",
            Get(_) => "get",
            #[cfg(feature = "list_gather")]
            Gather(_) => "gather",
            #[cfg(feature = "list_gather")]
            GatherEvery => "gather_every",
            #[cfg(feature = "list_count")]
            CountMatches => "count_matches",
            Sum => "sum",
            Min => "min",
            Max => "max",
            Mean => "mean",
            Median => "median",
            Std(_) => "std",
            Var(_) => "var",
            ArgMin => "arg_min",
            ArgMax => "arg_max",
            #[cfg(feature = "diff")]
            Diff { .. } => "diff",
            Length => "length",
            Sort(_) => "sort",
            Reverse => "reverse",
            Unique(is_stable) => {
                if *is_stable {
                    "unique_stable"
                } else {
                    "unique"
                }
            },
            NUnique => "n_unique",
            #[cfg(feature = "list_sets")]
            SetOperation(s) => return write!(f, "list.{s}"),
            #[cfg(feature = "list_any_all")]
            Any => "any",
            #[cfg(feature = "list_any_all")]
            All => "all",
            Join(_) => "join",
            #[cfg(feature = "dtype-array")]
            ToArray(_) => "to_array",
            #[cfg(feature = "list_to_struct")]
            ToStruct(_) => "to_struct",
        };
        write!(f, "list.{name}")
    }
}

impl From<ListFunction> for FunctionExpr {
    fn from(value: ListFunction) -> Self {
        Self::ListExpr(value)
    }
}
