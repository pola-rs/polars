use std::sync::Arc;

use polars_core::error::{PolarsResult, polars_bail, polars_ensure};
use polars_core::prelude::{
    ChunkExpandAtIndex, Column, DataType, IDX_DTYPE, IntoColumn, ListChunked, SortOptions,
};
use polars_core::utils::CustomIterTools;
use polars_ops::prelude::ListNameSpaceImpl;
use polars_plan::dsl::{ColumnsUdf, ReshapeDimension, SpecialEq};
use polars_plan::plans::IRListFunction;
use polars_utils::pl_str::PlSmallStr;

pub fn function_expr_to_udf(func: IRListFunction) -> SpecialEq<Arc<dyn ColumnsUdf>> {
    use IRListFunction::*;
    match func {
        Concat => wrap!(concat),
        #[cfg(feature = "is_in")]
        Contains { nulls_equal } => map_as_slice!(contains, nulls_equal),
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
        #[cfg(feature = "list_to_struct")]
        ToStruct(names) => map!(to_struct, &names),
    }
}

#[cfg(feature = "is_in")]
pub(super) fn contains(args: &mut [Column], nulls_equal: bool) -> PolarsResult<Column> {
    let list = &args[0];
    let item = &args[1];
    polars_ensure!(matches!(list.dtype(), DataType::List(_)),
        SchemaMismatch: "invalid series dtype: expected `List`, got `{}`", list.dtype(),
    );
    let mut ca = polars_ops::prelude::is_in(
        item.as_materialized_series(),
        list.as_materialized_series(),
        nulls_equal,
    )?;
    ca.rename(list.name().clone());
    Ok(ca.into_column())
}

#[cfg(feature = "list_drop_nulls")]
pub(super) fn drop_nulls(s: &Column) -> PolarsResult<Column> {
    let list = s.list()?;
    Ok(list.lst_drop_nulls().into_column())
}

#[cfg(feature = "list_sample")]
pub(super) fn sample_n(
    s: &[Column],
    with_replacement: bool,
    shuffle: bool,
    seed: Option<u64>,
) -> PolarsResult<Column> {
    let list = s[0].list()?;
    let n = &s[1];
    list.lst_sample_n(n.as_materialized_series(), with_replacement, shuffle, seed)
        .map(|ok| ok.into_column())
}

#[cfg(feature = "list_sample")]
pub(super) fn sample_fraction(
    s: &[Column],
    with_replacement: bool,
    shuffle: bool,
    seed: Option<u64>,
) -> PolarsResult<Column> {
    let list = s[0].list()?;
    let fraction = &s[1];
    list.lst_sample_fraction(
        fraction.as_materialized_series(),
        with_replacement,
        shuffle,
        seed,
    )
    .map(|ok| ok.into_column())
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

pub(super) fn shift(s: &[Column]) -> PolarsResult<Column> {
    let list = s[0].list()?;
    let periods = &s[1];

    list.lst_shift(periods).map(|ok| ok.into_column())
}

pub(super) fn slice(args: &mut [Column]) -> PolarsResult<Column> {
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
            return Ok(list_ca.lst_slice(offset, slice_len).into_column());
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
    out.rename(s.name().clone());
    Ok(out.into_column())
}

pub(super) fn concat(s: &mut [Column]) -> PolarsResult<Column> {
    let mut first = std::mem::take(&mut s[0]);
    let other = &s[1..];

    // TODO! don't auto cast here, but implode beforehand.
    let mut first_ca = match first.try_list() {
        Some(ca) => ca,
        None => {
            first = first
                .reshape_list(&[ReshapeDimension::Infer, ReshapeDimension::new_dimension(1)])
                .unwrap();
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

    first_ca.lst_concat(other).map(IntoColumn::into_column)
}

pub(super) fn get(s: &mut [Column], null_on_oob: bool) -> PolarsResult<Column> {
    let ca = s[0].list()?;
    let index = s[1].cast(&DataType::Int64)?;
    let index = index.i64().unwrap();

    polars_ops::prelude::lst_get(ca, index, null_on_oob)
}

#[cfg(feature = "list_gather")]
pub(super) fn gather(args: &[Column], null_on_oob: bool) -> PolarsResult<Column> {
    let ca = &args[0];
    let idx = &args[1];
    let ca = ca.list()?;

    if idx.len() == 1 && idx.dtype().is_primitive_numeric() && null_on_oob {
        // fast path
        let idx = idx.get(0)?.try_extract::<i64>()?;
        let out = ca.lst_get(idx, null_on_oob).map(Column::from)?;
        // make sure we return a list
        out.reshape_list(&[ReshapeDimension::Infer, ReshapeDimension::new_dimension(1)])
    } else {
        ca.lst_gather(idx.as_materialized_series(), null_on_oob)
            .map(Column::from)
    }
}

#[cfg(feature = "list_gather")]
pub(super) fn gather_every(args: &[Column]) -> PolarsResult<Column> {
    let ca = &args[0];
    let n = &args[1].strict_cast(&IDX_DTYPE)?;
    let offset = &args[2].strict_cast(&IDX_DTYPE)?;

    ca.list()?
        .lst_gather_every(n.idx()?, offset.idx()?)
        .map(Column::from)
}

#[cfg(feature = "list_count")]
pub(super) fn count_matches(args: &[Column]) -> PolarsResult<Column> {
    let s = &args[0];
    let element = &args[1];
    polars_ensure!(
        element.len() == 1,
        ComputeError: "argument expression in `list.count_matches` must produce exactly one element, got {}",
        element.len()
    );
    let ca = s.list()?;
    polars_ops::prelude::list_count_matches(ca, element.get(0).unwrap()).map(Column::from)
}

pub(super) fn sum(s: &Column) -> PolarsResult<Column> {
    s.list()?.lst_sum().map(Column::from)
}

pub(super) fn length(s: &Column) -> PolarsResult<Column> {
    Ok(s.list()?.lst_lengths().into_column())
}

pub(super) fn max(s: &Column) -> PolarsResult<Column> {
    s.list()?.lst_max().map(Column::from)
}

pub(super) fn min(s: &Column) -> PolarsResult<Column> {
    s.list()?.lst_min().map(Column::from)
}

pub(super) fn mean(s: &Column) -> PolarsResult<Column> {
    Ok(s.list()?.lst_mean().into())
}

pub(super) fn median(s: &Column) -> PolarsResult<Column> {
    Ok(s.list()?.lst_median().into())
}

pub(super) fn std(s: &Column, ddof: u8) -> PolarsResult<Column> {
    Ok(s.list()?.lst_std(ddof).into())
}

pub(super) fn var(s: &Column, ddof: u8) -> PolarsResult<Column> {
    Ok(s.list()?.lst_var(ddof)?.into())
}

pub(super) fn arg_min(s: &Column) -> PolarsResult<Column> {
    Ok(s.list()?.lst_arg_min().into_column())
}

pub(super) fn arg_max(s: &Column) -> PolarsResult<Column> {
    Ok(s.list()?.lst_arg_max().into_column())
}

#[cfg(feature = "diff")]
pub(super) fn diff(
    s: &Column,
    n: i64,
    null_behavior: polars_core::series::ops::NullBehavior,
) -> PolarsResult<Column> {
    Ok(s.list()?.lst_diff(n, null_behavior)?.into_column())
}

pub(super) fn sort(s: &Column, options: SortOptions) -> PolarsResult<Column> {
    Ok(s.list()?.lst_sort(options)?.into_column())
}

pub(super) fn reverse(s: &Column) -> PolarsResult<Column> {
    Ok(s.list()?.lst_reverse().into_column())
}

pub(super) fn unique(s: &Column, is_stable: bool) -> PolarsResult<Column> {
    if is_stable {
        Ok(s.list()?.lst_unique_stable()?.into_column())
    } else {
        Ok(s.list()?.lst_unique()?.into_column())
    }
}

#[cfg(feature = "list_sets")]
pub(super) fn set_operation(
    s: &[Column],
    set_type: polars_ops::prelude::SetOperation,
) -> PolarsResult<Column> {
    let s0 = &s[0];
    let s1 = &s[1];

    if s0.is_empty() || s1.is_empty() {
        use polars_ops::prelude::SetOperation;

        return match set_type {
            SetOperation::Intersection => {
                if s0.is_empty() {
                    Ok(s0.clone())
                } else {
                    Ok(s1.clone().with_name(s0.name().clone()))
                }
            },
            SetOperation::Difference => Ok(s0.clone()),
            SetOperation::Union | SetOperation::SymmetricDifference => {
                if s0.is_empty() {
                    Ok(s1.clone().with_name(s0.name().clone()))
                } else {
                    Ok(s0.clone())
                }
            },
        };
    }

    polars_ops::prelude::list_set_operation(s0.list()?, s1.list()?, set_type)
        .map(|ca| ca.into_column())
}

#[cfg(feature = "list_any_all")]
pub(super) fn lst_any(s: &Column) -> PolarsResult<Column> {
    s.list()?.lst_any().map(Column::from)
}

#[cfg(feature = "list_any_all")]
pub(super) fn lst_all(s: &Column) -> PolarsResult<Column> {
    s.list()?.lst_all().map(Column::from)
}

pub(super) fn join(s: &[Column], ignore_nulls: bool) -> PolarsResult<Column> {
    let ca = s[0].list()?;
    let separator = s[1].str()?;
    Ok(ca.lst_join(separator, ignore_nulls)?.into_column())
}

#[cfg(feature = "dtype-array")]
pub(super) fn to_array(s: &Column, width: usize) -> PolarsResult<Column> {
    if let DataType::List(inner) = s.dtype() {
        s.cast(&DataType::Array(inner.clone(), width))
    } else {
        polars_bail!(ComputeError: "expected List dtype")
    }
}

#[cfg(feature = "list_to_struct")]
pub(super) fn to_struct(s: &Column, names: &Arc<[PlSmallStr]>) -> PolarsResult<Column> {
    use polars_ops::prelude::ToStruct;

    let args = polars_ops::prelude::ListToStructArgs::FixedWidth(names.clone());
    Ok(s.list()?.to_struct(&args)?.into_column())
}

pub(super) fn n_unique(s: &Column) -> PolarsResult<Column> {
    Ok(s.list()?.lst_n_unique()?.into_column())
}
