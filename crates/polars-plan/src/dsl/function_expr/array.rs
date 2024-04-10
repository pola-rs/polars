use polars_ops::chunked_array::array::*;

use super::*;
use crate::{map, map_as_slice};

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ArrayFunction {
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
    Contains,
    #[cfg(feature = "array_count")]
    CountMatches,
    Shift,
}

impl ArrayFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use ArrayFunction::*;
        match self {
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
            Contains => mapper.with_dtype(DataType::Boolean),
            #[cfg(feature = "array_count")]
            CountMatches => mapper.with_dtype(IDX_DTYPE),
            Shift => mapper.with_same_dtype(),
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
            Contains => "contains",
            #[cfg(feature = "array_count")]
            CountMatches => "count_matches",
            Shift => "shift",
        };
        write!(f, "arr.{name}")
    }
}

impl From<ArrayFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: ArrayFunction) -> Self {
        use ArrayFunction::*;
        match func {
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
            Contains => map_as_slice!(contains),
            #[cfg(feature = "array_count")]
            CountMatches => map_as_slice!(count_matches),
            Shift => map_as_slice!(shift),
        }
    }
}

pub(super) fn max(s: &Series) -> PolarsResult<Series> {
    Ok(s.array()?.array_max())
}

pub(super) fn min(s: &Series) -> PolarsResult<Series> {
    Ok(s.array()?.array_min())
}

pub(super) fn sum(s: &Series) -> PolarsResult<Series> {
    s.array()?.array_sum()
}

pub(super) fn std(s: &Series, ddof: u8) -> PolarsResult<Series> {
    s.array()?.array_std(ddof)
}

pub(super) fn var(s: &Series, ddof: u8) -> PolarsResult<Series> {
    s.array()?.array_var(ddof)
}
pub(super) fn median(s: &Series) -> PolarsResult<Series> {
    s.array()?.array_median()
}

pub(super) fn unique(s: &Series, stable: bool) -> PolarsResult<Series> {
    let ca = s.array()?;
    let out = if stable {
        ca.array_unique_stable()
    } else {
        ca.array_unique()
    };
    out.map(|ca| ca.into_series())
}

pub(super) fn n_unique(s: &Series) -> PolarsResult<Series> {
    Ok(s.array()?.array_n_unique()?.into_series())
}

pub(super) fn to_list(s: &Series) -> PolarsResult<Series> {
    let list_dtype = map_array_dtype_to_list_dtype(s.dtype())?;
    s.cast(&list_dtype)
}

#[cfg(feature = "array_any_all")]
pub(super) fn any(s: &Series) -> PolarsResult<Series> {
    s.array()?.array_any()
}

#[cfg(feature = "array_any_all")]
pub(super) fn all(s: &Series) -> PolarsResult<Series> {
    s.array()?.array_all()
}

pub(super) fn sort(s: &Series, options: SortOptions) -> PolarsResult<Series> {
    Ok(s.array()?.array_sort(options)?.into_series())
}

pub(super) fn reverse(s: &Series) -> PolarsResult<Series> {
    Ok(s.array()?.array_reverse().into_series())
}

pub(super) fn arg_min(s: &Series) -> PolarsResult<Series> {
    Ok(s.array()?.array_arg_min().into_series())
}

pub(super) fn arg_max(s: &Series) -> PolarsResult<Series> {
    Ok(s.array()?.array_arg_max().into_series())
}

pub(super) fn get(s: &[Series], null_on_oob: bool) -> PolarsResult<Series> {
    let ca = s[0].array()?;
    let index = s[1].cast(&DataType::Int64)?;
    let index = index.i64().unwrap();
    ca.array_get(index, null_on_oob)
}

pub(super) fn join(s: &[Series], ignore_nulls: bool) -> PolarsResult<Series> {
    let ca = s[0].array()?;
    let separator = s[1].str()?;
    ca.array_join(separator, ignore_nulls)
}

#[cfg(feature = "is_in")]
pub(super) fn contains(s: &[Series]) -> PolarsResult<Series> {
    let array = &s[0];
    let item = &s[1];
    polars_ensure!(matches!(array.dtype(), DataType::Array(_, _)),
        SchemaMismatch: "invalid series dtype: expected `Array`, got `{}`", array.dtype(),
    );
    Ok(is_in(item, array)?.with_name(array.name()).into_series())
}

#[cfg(feature = "array_count")]
pub(super) fn count_matches(args: &[Series]) -> PolarsResult<Series> {
    let s = &args[0];
    let element = &args[1];
    polars_ensure!(
        element.len() == 1,
        ComputeError: "argument expression in `arr.count_matches` must produce exactly one element, got {}",
        element.len()
    );
    let ca = s.array()?;
    ca.array_count_matches(element.get(0).unwrap())
}

pub(super) fn shift(s: &[Series]) -> PolarsResult<Series> {
    let ca = s[0].array()?;
    let n = &s[1];

    ca.array_shift(n)
}
