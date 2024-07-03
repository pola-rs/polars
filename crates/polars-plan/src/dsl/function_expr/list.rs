use arrow::legacy::utils::CustomIterTools;
use polars_ops::chunked_array::list::*;

use super::*;
use crate::{map, map_as_slice, wrap};

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ListFunction {
    Concat,
    #[cfg(feature = "is_in")]
    Contains,
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
}

impl ListFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use ListFunction::*;
        match self {
            Concat => mapper.map_to_list_supertype(),
            #[cfg(feature = "is_in")]
            Contains => mapper.with_dtype(DataType::Boolean),
            #[cfg(feature = "list_drop_nulls")]
            DropNulls => mapper.with_same_dtype(),
            #[cfg(feature = "list_sample")]
            Sample { .. } => mapper.with_same_dtype(),
            Slice => mapper.with_same_dtype(),
            Shift => mapper.with_same_dtype(),
            Get(_) => mapper.map_to_list_and_array_inner_dtype(),
            #[cfg(feature = "list_gather")]
            Gather(_) => mapper.with_same_dtype(),
            #[cfg(feature = "list_gather")]
            GatherEvery => mapper.with_same_dtype(),
            #[cfg(feature = "list_count")]
            CountMatches => mapper.with_dtype(IDX_DTYPE),
            Sum => mapper.nested_sum_type(),
            Min => mapper.map_to_list_and_array_inner_dtype(),
            Max => mapper.map_to_list_and_array_inner_dtype(),
            Mean => mapper.with_dtype(DataType::Float64),
            Median => mapper.map_to_float_dtype(),
            Std(_) => mapper.map_to_float_dtype(), // Need to also have this sometimes marked as float32 or duration..
            Var(_) => mapper.map_to_float_dtype(),
            ArgMin => mapper.with_dtype(IDX_DTYPE),
            ArgMax => mapper.with_dtype(IDX_DTYPE),
            #[cfg(feature = "diff")]
            Diff { .. } => mapper.with_same_dtype(),
            Sort(_) => mapper.with_same_dtype(),
            Reverse => mapper.with_same_dtype(),
            Unique(_) => mapper.with_same_dtype(),
            Length => mapper.with_dtype(IDX_DTYPE),
            #[cfg(feature = "list_sets")]
            SetOperation(_) => mapper.with_same_dtype(),
            #[cfg(feature = "list_any_all")]
            Any => mapper.with_dtype(DataType::Boolean),
            #[cfg(feature = "list_any_all")]
            All => mapper.with_dtype(DataType::Boolean),
            Join(_) => mapper.with_dtype(DataType::String),
            #[cfg(feature = "dtype-array")]
            ToArray(width) => mapper.try_map_dtype(|dt| map_list_dtype_to_array_dtype(dt, *width)),
            NUnique => mapper.with_dtype(IDX_DTYPE),
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

impl Display for ListFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use ListFunction::*;

        let name = match self {
            Concat => "concat",
            #[cfg(feature = "is_in")]
            Contains => "contains",
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
        };
        write!(f, "list.{name}")
    }
}

impl From<ListFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: ListFunction) -> Self {
        use ListFunction::*;
        match func {
            Concat => wrap!(concat),
            #[cfg(feature = "is_in")]
            Contains => wrap!(contains),
            #[cfg(feature = "list_drop_nulls")]
            DropNulls => map!(drop_nulls),
            #[cfg(feature = "list_sample")]
            Sample {
                is_fraction,
                with_replacement,
                shuffle,
                seed,
            } => {
                if is_fraction {
                    map_as_slice!(sample_fraction, with_replacement, shuffle, seed)
                } else {
                    map_as_slice!(sample_n, with_replacement, shuffle, seed)
                }
            },
            Slice => wrap!(slice),
            Shift => map_as_slice!(shift),
            Get(null_on_oob) => wrap!(get, null_on_oob),
            #[cfg(feature = "list_gather")]
            Gather(null_on_oob) => map_as_slice!(gather, null_on_oob),
            #[cfg(feature = "list_gather")]
            GatherEvery => map_as_slice!(gather_every),
            #[cfg(feature = "list_count")]
            CountMatches => map_as_slice!(count_matches),
            Sum => map!(sum),
            Length => map!(length),
            Max => map!(max),
            Min => map!(min),
            Mean => map!(mean),
            Median => map!(median),
            Std(ddof) => map!(std, ddof),
            Var(ddof) => map!(var, ddof),
            ArgMin => map!(arg_min),
            ArgMax => map!(arg_max),
            #[cfg(feature = "diff")]
            Diff { n, null_behavior } => map!(diff, n, null_behavior),
            Sort(options) => map!(sort, options),
            Reverse => map!(reverse),
            Unique(is_stable) => map!(unique, is_stable),
            #[cfg(feature = "list_sets")]
            SetOperation(s) => map_as_slice!(set_operation, s),
            #[cfg(feature = "list_any_all")]
            Any => map!(lst_any),
            #[cfg(feature = "list_any_all")]
            All => map!(lst_all),
            Join(ignore_nulls) => map_as_slice!(join, ignore_nulls),
            #[cfg(feature = "dtype-array")]
            ToArray(width) => map!(to_array, width),
            NUnique => map!(n_unique),
        }
    }
}

#[cfg(feature = "is_in")]
pub(super) fn contains(args: &mut [Series]) -> PolarsResult<Option<Series>> {
    let list = &args[0];
    let item = &args[1];
    polars_ensure!(matches!(list.dtype(), DataType::List(_)),
        SchemaMismatch: "invalid series dtype: expected `List`, got `{}`", list.dtype(),
    );
    polars_ops::prelude::is_in(item, list).map(|mut ca| {
        ca.rename(list.name());
        Some(ca.into_series())
    })
}

#[cfg(feature = "list_drop_nulls")]
pub(super) fn drop_nulls(s: &Series) -> PolarsResult<Series> {
    let list = s.list()?;

    Ok(list.lst_drop_nulls().into_series())
}

#[cfg(feature = "list_sample")]
pub(super) fn sample_n(
    s: &[Series],
    with_replacement: bool,
    shuffle: bool,
    seed: Option<u64>,
) -> PolarsResult<Series> {
    let list = s[0].list()?;
    let n = &s[1];
    list.lst_sample_n(n, with_replacement, shuffle, seed)
        .map(|ok| ok.into_series())
}

#[cfg(feature = "list_sample")]
pub(super) fn sample_fraction(
    s: &[Series],
    with_replacement: bool,
    shuffle: bool,
    seed: Option<u64>,
) -> PolarsResult<Series> {
    let list = s[0].list()?;
    let fraction = &s[1];
    list.lst_sample_fraction(fraction, with_replacement, shuffle, seed)
        .map(|ok| ok.into_series())
}

fn check_slice_arg_shape(slice_len: usize, ca_len: usize, name: &str) -> PolarsResult<()> {
    polars_ensure!(
        slice_len == ca_len,
        ComputeError:
        "shape of the slice '{}' argument: {} does not match that of the list column: {}",
        name, slice_len, ca_len
    );
    Ok(())
}

pub(super) fn shift(s: &[Series]) -> PolarsResult<Series> {
    let list = s[0].list()?;
    let periods = &s[1];

    list.lst_shift(periods).map(|ok| ok.into_series())
}

pub(super) fn slice(args: &mut [Series]) -> PolarsResult<Option<Series>> {
    let s = &args[0];
    let list_ca = s.list()?;
    let offset_s = &args[1];
    let length_s = &args[2];

    let mut out: ListChunked = match (offset_s.len(), length_s.len()) {
        (1, 1) => {
            let offset = offset_s.get(0).unwrap().try_extract::<i64>()?;
            let slice_len = length_s
                .get(0)
                .unwrap()
                .extract::<usize>()
                .unwrap_or(usize::MAX);
            return Ok(Some(list_ca.lst_slice(offset, slice_len).into_series()));
        },
        (1, length_slice_len) => {
            check_slice_arg_shape(length_slice_len, list_ca.len(), "length")?;
            let offset = offset_s.get(0).unwrap().try_extract::<i64>()?;
            // cast to i64 as it is more likely that it is that dtype
            // instead of usize/u64 (we never need that max length)
            let length_ca = length_s.cast(&DataType::Int64)?;
            let length_ca = length_ca.i64().unwrap();

            list_ca
                .amortized_iter()
                .zip(length_ca)
                .map(|(opt_s, opt_length)| match (opt_s, opt_length) {
                    (Some(s), Some(length)) => Some(s.as_ref().slice(offset, length as usize)),
                    _ => None,
                })
                .collect_trusted()
        },
        (offset_len, 1) => {
            check_slice_arg_shape(offset_len, list_ca.len(), "offset")?;
            let length_slice = length_s
                .get(0)
                .unwrap()
                .extract::<usize>()
                .unwrap_or(usize::MAX);
            let offset_ca = offset_s.cast(&DataType::Int64)?;
            let offset_ca = offset_ca.i64().unwrap();
            list_ca
                .amortized_iter()
                .zip(offset_ca)
                .map(|(opt_s, opt_offset)| match (opt_s, opt_offset) {
                    (Some(s), Some(offset)) => Some(s.as_ref().slice(offset, length_slice)),
                    _ => None,
                })
                .collect_trusted()
        },
        _ => {
            check_slice_arg_shape(offset_s.len(), list_ca.len(), "offset")?;
            check_slice_arg_shape(length_s.len(), list_ca.len(), "length")?;
            let offset_ca = offset_s.cast(&DataType::Int64)?;
            let offset_ca = offset_ca.i64()?;
            // cast to i64 as it is more likely that it is that dtype
            // instead of usize/u64 (we never need that max length)
            let length_ca = length_s.cast(&DataType::Int64)?;
            let length_ca = length_ca.i64().unwrap();

            list_ca
                .amortized_iter()
                .zip(offset_ca)
                .zip(length_ca)
                .map(
                    |((opt_s, opt_offset), opt_length)| match (opt_s, opt_offset, opt_length) {
                        (Some(s), Some(offset), Some(length)) => {
                            Some(s.as_ref().slice(offset, length as usize))
                        },
                        _ => None,
                    },
                )
                .collect_trusted()
        },
    };
    out.rename(s.name());
    Ok(Some(out.into_series()))
}

pub(super) fn concat(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    let mut first = std::mem::take(&mut s[0]);
    let other = &s[1..];

    let mut first_ca = match first.list().ok() {
        Some(ca) => ca,
        None => {
            first = first.reshape_list(&[-1, 1]).unwrap();
            first.list().unwrap()
        },
    }
    .clone();

    if first_ca.len() == 1 && !other.is_empty() {
        let max_len = other.iter().map(|s| s.len()).max().unwrap();
        if max_len > 1 {
            first_ca = first_ca.new_from_index(0, max_len)
        }
    }

    first_ca.lst_concat(other).map(|ca| Some(ca.into_series()))
}

pub(super) fn get(s: &mut [Series], null_on_oob: bool) -> PolarsResult<Option<Series>> {
    let ca = s[0].list()?;
    let index = s[1].cast(&DataType::Int64)?;
    let index = index.i64().unwrap();

    match index.len() {
        1 => {
            let index = index.get(0);
            if let Some(index) = index {
                ca.lst_get(index, null_on_oob).map(Some)
            } else {
                Ok(Some(Series::full_null(
                    ca.name(),
                    ca.len(),
                    ca.inner_dtype(),
                )))
            }
        },
        len if len == ca.len() => {
            let ca = ca.rechunk();
            let arr = ca.downcast_iter().next().unwrap();
            let offsets = arr.offsets().as_slice();
            let take_by = if ca.null_count() == 0 {
                index
                    .iter()
                    .enumerate()
                    .map(|(i, opt_idx)| match opt_idx {
                        Some(idx) => {
                            let (start, end) = unsafe {
                                (*offsets.get_unchecked(i), *offsets.get_unchecked(i + 1))
                            };
                            let offset = if idx >= 0 { start + idx } else { end + idx };
                            if offset >= end || offset < start || start == end {
                                if null_on_oob {
                                    Ok(None)
                                } else {
                                    polars_bail!(ComputeError: "get index is out of bounds");
                                }
                            } else {
                                Ok(Some(offset as IdxSize))
                            }
                        },
                        None => Ok(None),
                    })
                    .collect::<Result<IdxCa, _>>()?
            } else {
                index
                    .iter()
                    .zip(arr.validity().unwrap())
                    .enumerate()
                    .map(|(i, (opt_idx, valid))| match (valid, opt_idx) {
                        (true, Some(idx)) => {
                            let (start, end) = unsafe {
                                (*offsets.get_unchecked(i), *offsets.get_unchecked(i + 1))
                            };
                            let offset = if idx >= 0 { start + idx } else { end + idx };
                            if offset >= end || offset < start || start == end {
                                if null_on_oob {
                                    Ok(None)
                                } else {
                                    polars_bail!(ComputeError: "get index is out of bounds");
                                }
                            } else {
                                Ok(Some(offset as IdxSize))
                            }
                        },
                        _ => Ok(None),
                    })
                    .collect::<Result<IdxCa, _>>()?
            };
            let s = Series::try_from((ca.name(), arr.values().clone())).unwrap();
            unsafe { s.take_unchecked(&take_by) }
                .cast(ca.inner_dtype())
                .map(Some)
        },
        len => polars_bail!(
            ComputeError:
            "`list.get` expression got an index array of length {} while the list has {} elements",
            len, ca.len()
        ),
    }
}

#[cfg(feature = "list_gather")]
pub(super) fn gather(args: &[Series], null_on_oob: bool) -> PolarsResult<Series> {
    let ca = &args[0];
    let idx = &args[1];
    let ca = ca.list()?;

    if idx.len() == 1 && null_on_oob {
        // fast path
        let idx = idx.get(0)?.try_extract::<i64>()?;
        let out = ca.lst_get(idx, null_on_oob)?;
        // make sure we return a list
        out.reshape_list(&[-1, 1])
    } else {
        ca.lst_gather(idx, null_on_oob)
    }
}

#[cfg(feature = "list_gather")]
pub(super) fn gather_every(args: &[Series]) -> PolarsResult<Series> {
    let ca = &args[0];
    let n = &args[1].strict_cast(&IDX_DTYPE)?;
    let offset = &args[2].strict_cast(&IDX_DTYPE)?;

    ca.list()?.lst_gather_every(n.idx()?, offset.idx()?)
}

#[cfg(feature = "list_count")]
pub(super) fn count_matches(args: &[Series]) -> PolarsResult<Series> {
    let s = &args[0];
    let element = &args[1];
    polars_ensure!(
        element.len() == 1,
        ComputeError: "argument expression in `list.count_matches` must produce exactly one element, got {}",
        element.len()
    );
    let ca = s.list()?;
    list_count_matches(ca, element.get(0).unwrap())
}

pub(super) fn sum(s: &Series) -> PolarsResult<Series> {
    s.list()?.lst_sum()
}

pub(super) fn length(s: &Series) -> PolarsResult<Series> {
    Ok(s.list()?.lst_lengths().into_series())
}

pub(super) fn max(s: &Series) -> PolarsResult<Series> {
    s.list()?.lst_max()
}

pub(super) fn min(s: &Series) -> PolarsResult<Series> {
    s.list()?.lst_min()
}

pub(super) fn mean(s: &Series) -> PolarsResult<Series> {
    Ok(s.list()?.lst_mean())
}

pub(super) fn median(s: &Series) -> PolarsResult<Series> {
    Ok(s.list()?.lst_median())
}

pub(super) fn std(s: &Series, ddof: u8) -> PolarsResult<Series> {
    Ok(s.list()?.lst_std(ddof))
}

pub(super) fn var(s: &Series, ddof: u8) -> PolarsResult<Series> {
    Ok(s.list()?.lst_var(ddof))
}

pub(super) fn arg_min(s: &Series) -> PolarsResult<Series> {
    Ok(s.list()?.lst_arg_min().into_series())
}

pub(super) fn arg_max(s: &Series) -> PolarsResult<Series> {
    Ok(s.list()?.lst_arg_max().into_series())
}

#[cfg(feature = "diff")]
pub(super) fn diff(s: &Series, n: i64, null_behavior: NullBehavior) -> PolarsResult<Series> {
    Ok(s.list()?.lst_diff(n, null_behavior)?.into_series())
}

pub(super) fn sort(s: &Series, options: SortOptions) -> PolarsResult<Series> {
    Ok(s.list()?.lst_sort(options)?.into_series())
}

pub(super) fn reverse(s: &Series) -> PolarsResult<Series> {
    Ok(s.list()?.lst_reverse().into_series())
}

pub(super) fn unique(s: &Series, is_stable: bool) -> PolarsResult<Series> {
    if is_stable {
        Ok(s.list()?.lst_unique_stable()?.into_series())
    } else {
        Ok(s.list()?.lst_unique()?.into_series())
    }
}

#[cfg(feature = "list_sets")]
pub(super) fn set_operation(s: &[Series], set_type: SetOperation) -> PolarsResult<Series> {
    let s0 = &s[0];
    let s1 = &s[1];

    if s0.len() == 0 || s1.len() == 0 {
        return match set_type {
            SetOperation::Intersection => {
                if s0.len() == 0 {
                    Ok(s0.clone())
                } else {
                    Ok(s1.clone().with_name(s0.name()))
                }
            },
            SetOperation::Difference => Ok(s0.clone()),
            SetOperation::Union | SetOperation::SymmetricDifference => {
                if s0.len() == 0 {
                    Ok(s1.clone().with_name(s0.name()))
                } else {
                    Ok(s0.clone())
                }
            },
        };
    }

    list_set_operation(s0.list()?, s1.list()?, set_type).map(|ca| ca.into_series())
}

#[cfg(feature = "list_any_all")]
pub(super) fn lst_any(s: &Series) -> PolarsResult<Series> {
    s.list()?.lst_any()
}

#[cfg(feature = "list_any_all")]
pub(super) fn lst_all(s: &Series) -> PolarsResult<Series> {
    s.list()?.lst_all()
}

pub(super) fn join(s: &[Series], ignore_nulls: bool) -> PolarsResult<Series> {
    let ca = s[0].list()?;
    let separator = s[1].str()?;
    Ok(ca.lst_join(separator, ignore_nulls)?.into_series())
}

#[cfg(feature = "dtype-array")]
pub(super) fn to_array(s: &Series, width: usize) -> PolarsResult<Series> {
    let array_dtype = map_list_dtype_to_array_dtype(s.dtype(), width)?;
    s.cast(&array_dtype)
}

pub(super) fn n_unique(s: &Series) -> PolarsResult<Series> {
    Ok(s.list()?.lst_n_unique()?.into_series())
}
