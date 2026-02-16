use polars_core::utils::slice_offsets;
use polars_ops::chunked_array::array::*;

use super::*;

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IRArrayFunction {
    Length,
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
    Explode(ExplodeOptions),
    Concat,
    Slice(i64, i64),
    #[cfg(feature = "array_to_struct")]
    ToStruct(Option<DslNameGenerator>),
}

impl<'a> FieldsMapper<'a> {
    /// Validate that the dtype is an array.
    pub fn ensure_is_array(self) -> PolarsResult<Self> {
        let dt = self.args()[0].dtype();
        polars_ensure!(
            dt.is_array(),
            InvalidOperation:
            "expected Array datatype for array operation, got: {dt:?}"
        );
        Ok(self)
    }
}

impl IRArrayFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use IRArrayFunction::*;

        match self {
            Concat => Ok(Field::new(
                mapper
                    .args()
                    .first()
                    .map_or(PlSmallStr::EMPTY, |x| x.name.clone()),
                concat_arr_output_dtype(
                    &mut mapper.args().iter().map(|x| (x.name.as_str(), &x.dtype)),
                )?,
            )),
            Length => mapper.ensure_is_array()?.with_dtype(IDX_DTYPE),
            Min | Max => mapper
                .ensure_is_array()?
                .map_to_list_and_array_inner_dtype(),
            Sum => mapper.ensure_is_array()?.nested_sum_type(),
            ToList => mapper
                .ensure_is_array()?
                .try_map_dtype(map_array_dtype_to_list_dtype),
            Unique(_) => mapper
                .ensure_is_array()?
                .try_map_dtype(map_array_dtype_to_list_dtype),
            NUnique => mapper.ensure_is_array()?.with_dtype(IDX_DTYPE),
            Std(_) => mapper.ensure_is_array()?.moment_dtype(),
            Var(_) => mapper.ensure_is_array()?.var_dtype(),
            Mean => mapper.ensure_is_array()?.moment_dtype(),
            Median => mapper.ensure_is_array()?.moment_dtype(),
            #[cfg(feature = "array_any_all")]
            Any | All => mapper.ensure_is_array()?.with_dtype(DataType::Boolean),
            Sort(_) => mapper.ensure_is_array()?.with_same_dtype(),
            Reverse => mapper.ensure_is_array()?.with_same_dtype(),
            ArgMin | ArgMax => mapper.ensure_is_array()?.with_dtype(IDX_DTYPE),
            Get(_) => mapper
                .ensure_is_array()?
                .map_to_list_and_array_inner_dtype(),
            Join(_) => mapper.ensure_is_array()?.with_dtype(DataType::String),
            #[cfg(feature = "is_in")]
            Contains { nulls_equal: _ } => mapper.ensure_is_array()?.with_dtype(DataType::Boolean),
            #[cfg(feature = "array_count")]
            CountMatches => mapper.ensure_is_array()?.with_dtype(IDX_DTYPE),
            Shift => mapper.ensure_is_array()?.with_same_dtype(),
            Explode { .. } => mapper.ensure_is_array()?.try_map_to_array_inner_dtype(),
            Slice(offset, length) => mapper
                .ensure_is_array()?
                .try_map_dtype(map_to_array_fixed_length(offset, length)),
            #[cfg(feature = "array_to_struct")]
            ToStruct(name_generator) => mapper.ensure_is_array()?.try_map_dtype(|dtype| {
                let DataType::Array(inner, width) = dtype else {
                    polars_bail!(InvalidOperation: "expected Array type, got: {dtype}")
                };

                (0..*width)
                    .map(|i| {
                        let name = match name_generator {
                            None => arr_default_struct_name_gen(i),
                            Some(ng) => PlSmallStr::from_string(ng.call(i)?),
                        };
                        Ok(Field::new(name, inner.as_ref().clone()))
                    })
                    .collect::<PolarsResult<Vec<Field>>>()
                    .map(DataType::Struct)
            }),
        }
    }

    pub fn function_options(&self) -> FunctionOptions {
        use IRArrayFunction as A;
        match self {
            #[cfg(feature = "array_any_all")]
            A::Any | A::All => FunctionOptions::elementwise(),
            #[cfg(feature = "is_in")]
            A::Contains { nulls_equal: _ } => FunctionOptions::elementwise(),
            #[cfg(feature = "array_count")]
            A::CountMatches => FunctionOptions::elementwise(),
            A::Concat => FunctionOptions::elementwise()
                .with_flags(|f| f | FunctionFlags::INPUT_WILDCARD_EXPANSION),
            A::Length
            | A::Min
            | A::Max
            | A::Sum
            | A::ToList
            | A::Unique(_)
            | A::NUnique
            | A::Std(_)
            | A::Var(_)
            | A::Mean
            | A::Median
            | A::Sort(_)
            | A::Reverse
            | A::ArgMin
            | A::ArgMax
            | A::Get(_)
            | A::Join(_)
            | A::Shift
            | A::Slice(_, _) => FunctionOptions::elementwise(),
            A::Explode { .. } => FunctionOptions::row_separable(),
            #[cfg(feature = "array_to_struct")]
            A::ToStruct(_) => FunctionOptions::elementwise(),
        }
    }
}

fn map_array_dtype_to_list_dtype(datatype: &DataType) -> PolarsResult<DataType> {
    if let DataType::Array(inner, _) = datatype {
        Ok(DataType::List(inner.clone()))
    } else {
        polars_bail!(ComputeError: "expected array dtype")
    }
}

fn map_to_array_fixed_length(
    offset: &i64,
    length: &i64,
) -> impl FnOnce(&DataType) -> PolarsResult<DataType> {
    move |datatype: &DataType| {
        if let DataType::Array(inner, array_len) = datatype {
            let length: usize = if *length < 0 {
                (*array_len as i64 + *length).max(0)
            } else {
                *length
            }.try_into().map_err(|_| {
                polars_err!(OutOfBounds: "length must be a non-negative integer, got: {}", length)
            })?;
            let (_, slice_offset) = slice_offsets(*offset, length, *array_len);
            Ok(DataType::Array(inner.clone(), slice_offset))
        } else {
            polars_bail!(ComputeError: "expected array dtype, got {}", datatype);
        }
    }
}

impl Display for IRArrayFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRArrayFunction::*;
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
            Slice(_, _) => "slice",
            Explode { .. } => "explode",
            #[cfg(feature = "array_to_struct")]
            ToStruct(_) => "to_struct",
        };
        write!(f, "arr.{name}")
    }
}

/// Determine the output dtype of a `concat_arr` operation. Also performs validation to ensure input
/// dtypes are compatible.
fn concat_arr_output_dtype(
    inputs: &mut dyn ExactSizeIterator<Item = (&str, &DataType)>,
) -> PolarsResult<DataType> {
    #[allow(clippy::len_zero)]
    if inputs.len() == 0 {
        // should not be reachable - we did not set ALLOW_EMPTY_INPUTS
        panic!();
    }

    let mut inputs = inputs.map(|(name, dtype)| {
        let (inner_dtype, width) = match dtype {
            DataType::Array(inner, width) => (inner.as_ref(), *width),
            dt => (dt, 1),
        };
        (name, dtype, inner_dtype, width)
    });
    let (first_name, first_dtype, first_inner_dtype, mut out_width) = inputs.next().unwrap();

    for (col_name, dtype, inner_dtype, width) in inputs {
        out_width += width;

        if inner_dtype != first_inner_dtype {
            polars_bail!(
                SchemaMismatch:
                "concat_arr dtype mismatch: expected {} or array[{}] dtype to match dtype of first \
                input column (name: {}, dtype: {}), got {} instead for column {}",
                first_inner_dtype, first_inner_dtype, first_name, first_dtype, dtype, col_name,
            )
        }
    }

    Ok(DataType::Array(
        Box::new(first_inner_dtype.clone()),
        out_width,
    ))
}
