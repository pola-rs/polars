use polars_ops::chunked_array::array::*;

use super::*;
use crate::{map, map_as_slice};

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

impl ArrayFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use ArrayFunction::*;
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
            Length => mapper.with_dtype(IDX_DTYPE),
            Min | Max => mapper.map_to_list_and_array_inner_dtype(),
            Sum => mapper.nested_sum_type(),
            ToList => mapper.try_map_dtype(map_array_dtype_to_list_dtype),
            Unique(_) => mapper.try_map_dtype(map_array_dtype_to_list_dtype),
            NUnique => mapper.with_dtype(IDX_DTYPE),
            Std(_) => mapper.map_to_float_dtype(),
            Var(_) => mapper.map_to_float_dtype(),
            Median => mapper.map_to_float_dtype(),
            #[cfg(feature = "array_any_all")]
            Any | All => mapper.with_dtype(DataType::Boolean),
            Sort(_) => mapper.with_same_dtype(),
            Reverse => mapper.with_same_dtype(),
            ArgMin | ArgMax => mapper.with_dtype(IDX_DTYPE),
            Get(_) => mapper.map_to_list_and_array_inner_dtype(),
            Join(_) => mapper.with_dtype(DataType::String),
            #[cfg(feature = "is_in")]
            Contains { nulls_equal: _ } => mapper.with_dtype(DataType::Boolean),
            #[cfg(feature = "array_count")]
            CountMatches => mapper.with_dtype(IDX_DTYPE),
            Shift => mapper.with_same_dtype(),
            Explode { .. } => mapper.try_map_to_array_inner_dtype(),
        }
    }

    pub fn function_options(&self) -> FunctionOptions {
        use ArrayFunction as A;
        match self {
            #[cfg(feature = "array_any_all")]
            A::Any | A::All => FunctionOptions::elementwise(),
            #[cfg(feature = "is_in")]
            A::Contains { nulls_equal: _ } => FunctionOptions::elementwise(),
            #[cfg(feature = "array_count")]
            A::CountMatches => FunctionOptions::elementwise(),
            A::Length
            | A::Min
            | A::Max
            | A::Sum
            | A::ToList
            | A::Unique(_)
            | A::NUnique
            | A::Std(_)
            | A::Var(_)
            | A::Median
            | A::Sort(_)
            | A::Reverse
            | A::ArgMin
            | A::ArgMax
            | A::Concat
            | A::Get(_)
            | A::Join(_)
            | A::Shift => FunctionOptions::elementwise(),
            A::Explode { .. } => FunctionOptions::row_separable(),
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

impl Display for ArrayFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
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

impl From<ArrayFunction> for SpecialEq<Arc<dyn ColumnsUdf>> {
    fn from(func: ArrayFunction) -> Self {
        use ArrayFunction::*;
        match func {
            Concat => map_as_slice!(concat_arr),
            Length => map!(length),
            Min => map!(min),
            Max => map!(max),
            Sum => map!(sum),
            ToList => map!(to_list),
            Unique(stable) => map!(unique, stable),
            NUnique => map!(n_unique),
            Std(ddof) => map!(std, ddof),
            Var(ddof) => map!(var, ddof),
            Median => map!(median),
            #[cfg(feature = "array_any_all")]
            Any => map!(any),
            #[cfg(feature = "array_any_all")]
            All => map!(all),
            Sort(options) => map!(sort, options),
            Reverse => map!(reverse),
            ArgMin => map!(arg_min),
            ArgMax => map!(arg_max),
            Get(null_on_oob) => map_as_slice!(get, null_on_oob),
            Join(ignore_nulls) => map_as_slice!(join, ignore_nulls),
            #[cfg(feature = "is_in")]
            Contains { nulls_equal } => map_as_slice!(contains, nulls_equal),
            #[cfg(feature = "array_count")]
            CountMatches => map_as_slice!(count_matches),
            Shift => map_as_slice!(shift),
            Explode { skip_empty } => map_as_slice!(explode, skip_empty),
        }
    }
}

pub(super) fn length(s: &Column) -> PolarsResult<Column> {
    let array = s.array()?;
    let width = array.width();
    let width = IdxSize::try_from(width)
        .map_err(|_| polars_err!(bigidx, ctx = "array length", size = width))?;

    let mut c = Column::new_scalar(array.name().clone(), width.into(), array.len());
    if let Some(validity) = array.rechunk_validity() {
        let mut series = c.into_materialized_series().clone();

        // SAFETY: We keep datatypes intact and call compute_len afterwards.
        let chunks = unsafe { series.chunks_mut() };
        assert_eq!(chunks.len(), 1);

        chunks[0] = chunks[0].with_validity(Some(validity));

        series.compute_len();
        c = series.into_column();
    }

    Ok(c)
}

pub(super) fn max(s: &Column) -> PolarsResult<Column> {
    Ok(s.array()?.array_max().into())
}

pub(super) fn min(s: &Column) -> PolarsResult<Column> {
    Ok(s.array()?.array_min().into())
}

pub(super) fn sum(s: &Column) -> PolarsResult<Column> {
    s.array()?.array_sum().map(Column::from)
}

pub(super) fn std(s: &Column, ddof: u8) -> PolarsResult<Column> {
    s.array()?.array_std(ddof).map(Column::from)
}

pub(super) fn var(s: &Column, ddof: u8) -> PolarsResult<Column> {
    s.array()?.array_var(ddof).map(Column::from)
}
pub(super) fn median(s: &Column) -> PolarsResult<Column> {
    s.array()?.array_median().map(Column::from)
}

pub(super) fn unique(s: &Column, stable: bool) -> PolarsResult<Column> {
    let ca = s.array()?;
    let out = if stable {
        ca.array_unique_stable()
    } else {
        ca.array_unique()
    };
    out.map(|ca| ca.into_column())
}

pub(super) fn n_unique(s: &Column) -> PolarsResult<Column> {
    Ok(s.array()?.array_n_unique()?.into_column())
}

pub(super) fn to_list(s: &Column) -> PolarsResult<Column> {
    let list_dtype = map_array_dtype_to_list_dtype(s.dtype())?;
    s.cast(&list_dtype)
}

#[cfg(feature = "array_any_all")]
pub(super) fn any(s: &Column) -> PolarsResult<Column> {
    s.array()?.array_any().map(Column::from)
}

#[cfg(feature = "array_any_all")]
pub(super) fn all(s: &Column) -> PolarsResult<Column> {
    s.array()?.array_all().map(Column::from)
}

pub(super) fn sort(s: &Column, options: SortOptions) -> PolarsResult<Column> {
    Ok(s.array()?.array_sort(options)?.into_column())
}

pub(super) fn reverse(s: &Column) -> PolarsResult<Column> {
    Ok(s.array()?.array_reverse().into_column())
}

pub(super) fn arg_min(s: &Column) -> PolarsResult<Column> {
    Ok(s.array()?.array_arg_min().into_column())
}

pub(super) fn arg_max(s: &Column) -> PolarsResult<Column> {
    Ok(s.array()?.array_arg_max().into_column())
}

pub(super) fn get(s: &[Column], null_on_oob: bool) -> PolarsResult<Column> {
    let ca = s[0].array()?;
    let index = s[1].cast(&DataType::Int64)?;
    let index = index.i64().unwrap();
    ca.array_get(index, null_on_oob).map(Column::from)
}

pub(super) fn join(s: &[Column], ignore_nulls: bool) -> PolarsResult<Column> {
    let ca = s[0].array()?;
    let separator = s[1].str()?;
    ca.array_join(separator, ignore_nulls).map(Column::from)
}

#[cfg(feature = "is_in")]
pub(super) fn contains(s: &[Column], nulls_equal: bool) -> PolarsResult<Column> {
    let array = &s[0];
    let item = &s[1];
    polars_ensure!(matches!(array.dtype(), DataType::Array(_, _)),
        SchemaMismatch: "invalid series dtype: expected `Array`, got `{}`", array.dtype(),
    );
    let mut ca = is_in(
        item.as_materialized_series(),
        array.as_materialized_series(),
        nulls_equal,
    )?;
    ca.rename(array.name().clone());
    Ok(ca.into_column())
}

#[cfg(feature = "array_count")]
pub(super) fn count_matches(args: &[Column]) -> PolarsResult<Column> {
    let s = &args[0];
    let element = &args[1];
    polars_ensure!(
        element.len() == 1,
        ComputeError: "argument expression in `arr.count_matches` must produce exactly one element, got {}",
        element.len()
    );
    let ca = s.array()?;
    ca.array_count_matches(element.get(0).unwrap())
        .map(Column::from)
}

pub(super) fn shift(s: &[Column]) -> PolarsResult<Column> {
    let ca = s[0].array()?;
    let n = &s[1];

    ca.array_shift(n.as_materialized_series()).map(Column::from)
}

fn explode(c: &[Column], skip_empty: bool) -> PolarsResult<Column> {
    c[0].explode(skip_empty)
}

fn concat_arr(args: &[Column]) -> PolarsResult<Column> {
    let dtype = concat_arr_output_dtype(&mut args.iter().map(|c| (c.name().as_str(), c.dtype())))?;

    polars_ops::series::concat_arr::concat_arr(args, &dtype)
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
