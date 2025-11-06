use polars_core::error::{PolarsResult, polars_bail, polars_ensure, polars_err};
use polars_core::prelude::{Column, DataType, IntoColumn, SortOptions};
use polars_ops::prelude::array::ArrayNameSpace;
#[cfg(feature = "array_to_struct")]
use polars_plan::dsl::DslNameGenerator;
use polars_plan::dsl::{ColumnsUdf, SpecialEq};
use polars_plan::plans::IRArrayFunction;
use polars_utils::pl_str::PlSmallStr;

use super::*;

pub fn function_expr_to_udf(func: IRArrayFunction) -> SpecialEq<Arc<dyn ColumnsUdf>> {
    use IRArrayFunction::*;
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
        Mean => map!(mean),
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
        Slice(offset, length) => map!(slice, offset, length),
        #[cfg(feature = "array_to_struct")]
        ToStruct(ng) => map!(arr_to_struct, ng.clone()),
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

pub(super) fn mean(s: &Column) -> PolarsResult<Column> {
    s.array()?.array_mean().map(Column::from)
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
    if let DataType::Array(inner, _) = s.dtype() {
        s.cast(&DataType::List(inner.clone()))
    } else {
        polars_bail!(ComputeError: "expected array dtype")
    }
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
    let mut ca = polars_ops::series::is_in(
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

pub(super) fn slice(s: &Column, offset: i64, length: i64) -> PolarsResult<Column> {
    let ca = s.array()?;
    ca.array_slice(offset, length).map(Column::from)
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

#[cfg(feature = "array_to_struct")]
fn arr_to_struct(s: &Column, name_generator: Option<DslNameGenerator>) -> PolarsResult<Column> {
    use polars_ops::prelude::array::ToStruct;

    let name_generator =
        name_generator.map(|f| Arc::new(move |i| f.call(i).map(PlSmallStr::from)) as Arc<_>);
    s.array()?
        .to_struct(name_generator)
        .map(IntoColumn::into_column)
}
