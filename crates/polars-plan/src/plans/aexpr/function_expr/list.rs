use polars_core::utils::SuperTypeOptions;

use super::*;

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IRListFunction {
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

impl<'a> FieldsMapper<'a> {
    /// Validate that the dtype is a List.
    pub fn ensure_is_list(self) -> PolarsResult<Self> {
        let dt = self.args()[0].dtype();
        polars_ensure!(
            dt.is_list(),
            InvalidOperation:
            "expected List data type for list operation, got: {dt:?}"
        );
        Ok(self)
    }
}

impl IRListFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use IRListFunction::*;
        match self {
            Concat => mapper.map_to_list_supertype(),
            #[cfg(feature = "is_in")]
            Contains { nulls_equal: _ } => mapper.ensure_is_list()?.with_dtype(DataType::Boolean),
            #[cfg(feature = "list_drop_nulls")]
            DropNulls => mapper.ensure_is_list()?.with_same_dtype(),
            #[cfg(feature = "list_sample")]
            Sample { .. } => mapper.ensure_is_list()?.with_same_dtype(),
            Slice => mapper.ensure_is_list()?.with_same_dtype(),
            Shift => mapper.ensure_is_list()?.with_same_dtype(),
            Get(_) => mapper.ensure_is_list()?.map_to_list_and_array_inner_dtype(),
            #[cfg(feature = "list_gather")]
            Gather(_) => mapper.ensure_is_list()?.with_same_dtype(),
            #[cfg(feature = "list_gather")]
            GatherEvery => mapper.ensure_is_list()?.with_same_dtype(),
            #[cfg(feature = "list_count")]
            CountMatches => mapper.ensure_is_list()?.with_dtype(IDX_DTYPE),
            Sum => mapper.nested_sum_type(),
            Min => mapper.ensure_is_list()?.map_to_list_and_array_inner_dtype(),
            Max => mapper.ensure_is_list()?.map_to_list_and_array_inner_dtype(),
            Mean => mapper.nested_mean_median_type(),
            Median => mapper.nested_mean_median_type(),
            Std(_) => mapper.ensure_is_list()?.moment_dtype(), // Need to also have this sometimes marked as float32 or duration..
            Var(_) => mapper.ensure_is_list()?.var_dtype(),
            ArgMin => mapper.ensure_is_list()?.with_dtype(IDX_DTYPE),
            ArgMax => mapper.ensure_is_list()?.with_dtype(IDX_DTYPE),
            #[cfg(feature = "diff")]
            Diff { .. } => mapper.try_map_dtype(|dt| {
                let DataType::List(inner) = dt else {
                    polars_bail!(op = "list.diff", dt);
                };

                let inner_dt = match inner.as_ref() {
                    #[cfg(feature = "dtype-datetime")]
                    DataType::Datetime(tu, _) => DataType::Duration(*tu),
                    #[cfg(feature = "dtype-date")]
                    DataType::Date => DataType::Duration(TimeUnit::Microseconds),
                    #[cfg(feature = "dtype-time")]
                    DataType::Time => DataType::Duration(TimeUnit::Nanoseconds),
                    DataType::UInt64 | DataType::UInt32 => DataType::Int64,
                    DataType::UInt16 => DataType::Int32,
                    DataType::UInt8 => DataType::Int16,
                    inner_dt => inner_dt.clone(),
                };

                Ok(DataType::List(Box::new(inner_dt)))
            }),
            Sort(_) => mapper.ensure_is_list()?.with_same_dtype(),
            Reverse => mapper.ensure_is_list()?.with_same_dtype(),
            Unique(_) => mapper.ensure_is_list()?.with_same_dtype(),
            Length => mapper.ensure_is_list()?.with_dtype(IDX_DTYPE),
            #[cfg(feature = "list_sets")]
            SetOperation(_) => mapper.ensure_is_list()?.with_same_dtype(),
            #[cfg(feature = "list_any_all")]
            Any => mapper.ensure_is_list()?.with_dtype(DataType::Boolean),
            #[cfg(feature = "list_any_all")]
            All => mapper.ensure_is_list()?.with_dtype(DataType::Boolean),
            Join(_) => mapper.try_map_dtype(|dtype| {
                let DataType::List(inner_dtype) = dtype else {
                    polars_bail!(
                        InvalidOperation:
                        "attempted list to_struct on non-list dtype: {dtype}",
                    );
                };
                let inner_dtype = inner_dtype.as_ref();
                polars_ensure!(inner_dtype.is_string(), InvalidOperation:
                            "attempted list join with non-string dtype: {dtype}",);
                Ok(DataType::String)
            }),
            #[cfg(feature = "dtype-array")]
            ToArray(width) => mapper
                .ensure_is_list()?
                .try_map_dtype(|dt| map_list_dtype_to_array_dtype(dt, *width)),
            NUnique => mapper.ensure_is_list()?.with_dtype(IDX_DTYPE),
            #[cfg(feature = "list_to_struct")]
            ToStruct(names) => mapper.try_map_dtype(|dtype| {
                let DataType::List(inner_dtype) = dtype else {
                    polars_bail!(
                        InvalidOperation:
                        "attempted list to_struct on non-list dtype: {dtype}",
                    );
                };
                let inner_dtype = inner_dtype.as_ref();

                Ok(DataType::Struct(
                    names
                        .iter()
                        .map(|x| Field::new(x.clone(), inner_dtype.clone()))
                        .collect::<Vec<_>>(),
                ))
            }),
        }
    }

    pub fn function_options(&self) -> FunctionOptions {
        use IRListFunction as L;
        match self {
            L::Concat => FunctionOptions::elementwise()
                .with_flags(|f| f | FunctionFlags::INPUT_WILDCARD_EXPANSION),
            #[cfg(feature = "is_in")]
            L::Contains { nulls_equal: _ } => FunctionOptions::elementwise(),
            #[cfg(feature = "list_sample")]
            L::Sample { .. } => FunctionOptions::elementwise(),
            #[cfg(feature = "list_gather")]
            L::Gather(_) => FunctionOptions::elementwise(),
            #[cfg(feature = "list_gather")]
            L::GatherEvery => FunctionOptions::elementwise(),
            #[cfg(feature = "list_sets")]
            L::SetOperation(_) => FunctionOptions::elementwise()
                .with_casting_rules(CastingRules::Supertype(SuperTypeOptions {
                    flags: SuperTypeFlags::default() | SuperTypeFlags::ALLOW_IMPLODE_LIST,
                }))
                .with_flags(|f| f & !FunctionFlags::RETURNS_SCALAR),
            #[cfg(feature = "diff")]
            L::Diff { .. } => FunctionOptions::elementwise(),
            #[cfg(feature = "list_drop_nulls")]
            L::DropNulls => FunctionOptions::elementwise(),
            #[cfg(feature = "list_count")]
            L::CountMatches => FunctionOptions::elementwise(),
            L::Sum
            | L::Slice
            | L::Shift
            | L::Get(_)
            | L::Length
            | L::Max
            | L::Min
            | L::Mean
            | L::Median
            | L::Std(_)
            | L::Var(_)
            | L::ArgMin
            | L::ArgMax
            | L::Sort(_)
            | L::Reverse
            | L::Unique(_)
            | L::Join(_)
            | L::NUnique => FunctionOptions::elementwise(),
            #[cfg(feature = "list_any_all")]
            L::Any | L::All => FunctionOptions::elementwise(),
            #[cfg(feature = "dtype-array")]
            L::ToArray(_) => FunctionOptions::elementwise(),
            #[cfg(feature = "list_to_struct")]
            L::ToStruct(_) => FunctionOptions::elementwise(),
        }
    }
}

#[cfg(feature = "dtype-array")]
fn map_list_dtype_to_array_dtype(datatype: &DataType, width: usize) -> PolarsResult<DataType> {
    if let DataType::List(inner) = datatype {
        Ok(DataType::Array(inner.clone(), width))
    } else {
        polars_bail!(ComputeError: "expected List dtype")
    }
}

impl Display for IRListFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRListFunction::*;

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
